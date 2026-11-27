"""Per-stage timeline of an eligible stream's first two chunks, and the bisect bursts, for one tree on one card.

Boots TREE on card N with the frame probe, then:
- bursts (opening_burst.run_burst) at each COUNT: capped x3, override x2, normal x1;
- per load (name:concurrency:requests), a closed loop where every request is eligible (Ryan, English, streamed, default
  sampling) with request events recorded for the whole pass. The client stamps send and first byte in wall-clock ns.
Per pass it also reads per-thread CPU ticks of the server's process group and the card's memory.
Writes OUT/bursts/<count>_<burst>_r<i>, OUT/<load>/{events,client.jsonl,threads.json}, OUT/trace.txt.

usage: python3 opening_stage_trace.py --tree TREE --out DIR --port P [--counts 48 16] [--loads c1:1:100 c64:64:1088]
"""

import argparse
import collections
import concurrent.futures
import json
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from eligible_load import audible_onset
from opening_burst import run_burst
from seedtts_pair import serve, stop
from served_baseline import MODELS, wait_healthy
from slot_wait_serving import codec_state

HERE = Path(__file__).resolve().parent
SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
SAMPLE_RATE = 24000
EVENT_CAPACITY = 4000000
REQUEST_TIMEOUT_S = 300
WARMUP_REQUESTS = 64
BURSTS = (
    ("capped", {"max_new_tokens": 3}, True, 3),
    ("override", {"max_new_tokens": 8, "initial_codec_chunk_frames": 4}, True, 2),
    ("normal", {}, False, 1),
)
# the first decode's events carry decoded_chunks 0 until its commit, which carries 1
STAGES = (
    "admission", "prefill_start", "prefill_end", "frame1", "frame2", "enqueued", "dispatched", "launched", "resolved",
    "committed", "coordinator",
)


def run_request(client: httpx.Client, url: str, text: str, extra: dict) -> dict:
    body = {
        "model": MODELS["customvoice"][0], "input": text, "voice": "Ryan", "language": "English",
        "response_format": "pcm", "stream": True, **extra,
    }
    send_ns = time.time_ns()
    start = time.monotonic()
    arrivals: list[tuple[float, int]] = []
    audio = bytearray()
    with client.stream("POST", f"{url}/v1/audio/speech", json=body) as response:
        if response.status_code != 200:
            return {"status": response.status_code}
        else:
            pass
        for piece in response.iter_bytes():
            arrivals.append((time.monotonic() - start, len(audio) // 2))
            audio.extend(piece)
    latency_s = time.monotonic() - start
    samples = np.frombuffer(bytes(audio[: len(audio) // 2 * 2]), dtype=np.int16).astype(np.float32) / 32768.0
    if not arrivals or samples.size == 0:
        return {"status": 200, "samples": 0}
    else:
        pass
    first_byte_s = arrivals[0][0]
    stalls = [
        seconds - (first_byte_s + before / SAMPLE_RATE)
        for seconds, before in arrivals[1:]
        if seconds > first_byte_s + before / SAMPLE_RATE
    ]
    # margin of the second piece: how long before the first piece's audio runs out it arrived (negative is an underrun)
    second = next(((seconds, before) for seconds, before in arrivals[1:] if before > 0), None)
    onset = audible_onset(samples)
    return {
        "status": 200, "samples": int(samples.size), "send_ns": send_ns, "first_byte_ms": first_byte_s * 1000.0,
        "first_piece_samples": arrivals[1][1] if len(arrivals) > 1 else int(samples.size),
        "second_margin_ms": None if second is None else (first_byte_s + second[1] / SAMPLE_RATE - second[0]) * 1000.0,
        "audible_ms": None if onset is None else (first_byte_s + onset / SAMPLE_RATE) * 1000.0,
        "leading_silence_ms": None if onset is None else 1000.0 * onset / SAMPLE_RATE,
        "latency_s": latency_s, "audio_s": samples.size / SAMPLE_RATE, "stalls": len(stalls), "stall_ms": 1000.0 * sum(stalls),
    }


def run_pass(url: str, texts: list[str], concurrency: int, extra: dict) -> tuple[list[dict], float]:
    work: queue.Queue = queue.Queue()
    for text in texts:
        work.put(text)
    rows: list[dict] = []
    lock = threading.Lock()

    def worker() -> None:
        with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
            while True:
                try:
                    text = work.get_nowait()
                except queue.Empty:
                    return
                row = run_request(client, url, text, extra)
                with lock:
                    rows.append(row)

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for future in [pool.submit(worker) for _ in range(concurrency)]:
            future.result()
    return rows, time.monotonic() - start


def thread_ticks(pgid: int) -> dict[str, int]:
    """utime + stime per thread name over every process in the server's process group."""
    ticks: dict[str, int] = collections.defaultdict(int)
    pids = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True).stdout.split()
    for pid in pids:
        for task in Path(f"/proc/{pid}/task").glob("*"):
            try:
                stat = (task / "stat").read_text()
            except OSError:
                continue
            name = stat[stat.index("(") + 1 : stat.rindex(")")]
            fields = stat[stat.rindex(")") + 2 :].split()
            ticks[name] += int(fields[11]) + int(fields[12])
    return dict(ticks)


def quantiles(values: list[float]) -> str:
    if not values:
        return "none"
    else:
        pass
    values = sorted(values)
    picks = [values[min(len(values) - 1, int(fraction * len(values)))] for fraction in (0.1, 0.5, 0.9)]
    return "/".join(f"{value:.1f}" for value in picks) + f" mean {statistics.mean(values):.1f}"


def timelines(event_dir: Path) -> list[dict]:
    by_request: dict[str, list[dict]] = collections.defaultdict(list)
    for path in event_dir.glob("events_*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            by_request[event["request_id"]].append(event)
    rows = []
    for request_id, events in by_request.items():
        events.sort(key=lambda event: event["timestamp_ns"])
        row: dict = {"request_id": request_id}
        for event in events:
            metadata = event.get("metadata") or {}
            name = event["event_name"]
            chunks = metadata.get("decoded_chunks")
            stamp = event["timestamp_ns"]
            if name == "request_admission":
                row.setdefault("admission", stamp)
            elif name == "scheduler_prefill_start":
                row.setdefault("prefill_start", stamp)
            elif name == "scheduler_prefill_end":
                row.setdefault("prefill_end", stamp)
            elif name == "qwen3_tts_frames_ingested":
                frames = metadata.get("generated_frames", 0)
                for count in range(1, min(frames, 6) + 1):
                    row.setdefault(f"frame{count}", stamp)
            elif name == "qwen3_tts_vocoder_decode_enqueued" and chunks in (0, 1):
                row.setdefault("enqueued" if chunks == 0 else "enqueued2", stamp)
            elif name == "qwen3_tts_vocoder_decode_dispatched" and chunks in (0, 1):
                if ("dispatched" if chunks == 0 else "dispatched2") not in row:
                    row["dispatched" if chunks == 0 else "dispatched2"] = stamp
                    row["collected" if chunks == 0 else "collected2"] = metadata.get("collected")
                else:
                    pass
            elif name == "qwen3_tts_vocoder_decode_launched" and chunks in (0, 1):
                if ("launched" if chunks == 0 else "launched2") not in row:
                    row["launched" if chunks == 0 else "launched2"] = stamp
                    row["width" if chunks == 0 else "width2"] = metadata.get("fresh_frames", metadata.get("path"))
                    row["cohort" if chunks == 0 else "cohort2"] = metadata.get("cohort_size")
                else:
                    pass
            elif name == "qwen3_tts_vocoder_decode_resolved" and chunks in (0, 1):
                row.setdefault("resolved" if chunks == 0 else "resolved2", stamp)
            elif name == "qwen3_tts_vocoder_decode_committed" and chunks in (1, 2):
                if ("committed" if chunks == 1 else "committed2") not in row:
                    row["committed" if chunks == 1 else "committed2"] = stamp
                    row["samples" if chunks == 1 else "samples2"] = metadata.get("samples")
                else:
                    pass
            elif name == "coordinator_stream_received":
                if "coordinator" not in row:
                    row["coordinator"] = stamp
                elif "coordinator2" not in row:
                    row["coordinator2"] = stamp
                else:
                    pass
            elif name == "stage_complete" and event.get("stage") == "tts_engine":
                row.setdefault("engine_done", stamp)
            else:
                pass
        if "admission" in row and "committed" in row:
            rows.append(row)
        else:
            pass
    rows.sort(key=lambda row: row["admission"])
    return rows


def digest(event_dir: Path, client_rows: list[dict], skip: int) -> str:
    rows = timelines(event_dir)[skip:]
    if not rows:
        return "no stream recorded"
    else:
        pass
    milliseconds = 1e6
    lines = [
        f"streams {len(rows)} (p10/p50/p90 mean, ms), first decode widths "
        f"{dict(collections.Counter(row.get('width') for row in rows).most_common(4))}, first commit samples "
        f"{dict(collections.Counter(row.get('samples') for row in rows).most_common(4))}, second widths "
        f"{dict(collections.Counter(row.get('width2') for row in rows).most_common(4))}",
        f"first decode cohort size {quantiles([row['cohort'] for row in rows if row.get('cohort')])}, "
        f"collected {quantiles([row['collected'] for row in rows if row.get('collected')])}",
    ]
    for before, after in zip(STAGES, STAGES[1:]):
        gaps = [(row[after] - row[before]) / milliseconds for row in rows if before in row and after in row]
        lines.append(f"{before:13s} -> {after:13s} {quantiles(gaps)}")
    for stage in STAGES[1:]:
        gaps = [(row[stage] - row["admission"]) / milliseconds for row in rows if stage in row]
        lines.append(f"admission -> {stage:13s} {quantiles(gaps)}")
    for before, after in (
        ("frame1", "enqueued"), ("frame2", "enqueued"), ("frame1", "committed"), ("frame2", "committed"),
        ("dispatched", "resolved"), ("frame2", "frame3"), ("frame3", "frame4"), ("frame4", "frame5"),
        ("committed", "enqueued2"), ("enqueued2", "launched2"), ("launched2", "resolved2"), ("resolved2", "committed2"),
        ("committed", "committed2"), ("coordinator", "coordinator2"),
    ):
        gaps = [(row[after] - row[before]) / milliseconds for row in rows if before in row and after in row]
        lines.append(f"{before:13s} => {after:13s} {quantiles(gaps)}")
    by_width: dict = collections.defaultdict(list)
    for row in rows:
        if "launched" in row and "resolved" in row:
            by_width[(row.get("width"), min(row.get("cohort") or 0, 64))].append((row["resolved"] - row["launched"]) / milliseconds)
        else:
            pass
    for (width, cohort), gaps in sorted(by_width.items(), key=lambda entry: str(entry[0])):
        if len(gaps) >= 5:
            lines.append(f"launch to resolve, width {width} cohort {cohort}: {quantiles(gaps)} n {len(gaps)}")
        else:
            pass
    late = [row for row in rows if "engine_done" in row and "dispatched" in row and row["engine_done"] < row["dispatched"]]
    lines.append(f"engine done before first dispatch {len(late)}")
    served = [row for row in client_rows if row["status"] == 200 and row.get("samples")]
    served.sort(key=lambda row: row["send_ns"])
    served = served[skip:]
    lines.append(f"client n {len(served)} first byte {quantiles([row['first_byte_ms'] for row in served])}")
    lines.append(f"client audible {quantiles([row['audible_ms'] for row in served if row['audible_ms'] is not None])}")
    lines.append(f"client first piece samples {dict(collections.Counter(row['first_piece_samples'] for row in served).most_common(4))}")
    lines.append(f"client second margin {quantiles([row['second_margin_ms'] for row in served if row['second_margin_ms'] is not None])}")
    margins = sorted(row["second_margin_ms"] for row in served if row["second_margin_ms"] is not None)
    if margins:
        lines.append(f"client second margin p1 {margins[len(margins) // 100]:.1f} min {margins[0]:.1f} negative {sum(1 for value in margins if value < 0)}")
    else:
        pass
    lines.append(
        f"client underrun requests {sum(1 for row in served if row['stalls'])} stall ms {sum(row['stall_ms'] for row in served):.0f} "
        f"rtf {statistics.mean(row['latency_s'] / row['audio_s'] for row in served):.4f} "
        f"leading silence >= 80 ms {sum(1 for row in served if (row['leading_silence_ms'] or 0) >= 80)}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--counts", type=int, nargs="*", default=[48, 16])
    parser.add_argument("--loads", nargs="*", default=["c1:1:100", "c16:16:400", "c64:64:1088"])
    parser.add_argument("--extra", default="{}", help="JSON of extra request fields for the loads")
    args = parser.parse_args()
    sys.path.insert(0, str(args.tree))
    from benchmarks.dataset.seedtts import load_seedtts_samples

    texts = [sample.target_text for sample in load_seedtts_samples(SEEDTTS_META, split="en")]
    args.out.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD", "HEAD:sglang_omni"], cwd=args.tree, text=True).split()
    report = [f"tree {args.tree} {head[0]} sglang_omni {head[1]}"]
    print(report[0], flush=True)
    url = f"http://127.0.0.1:{args.port}"
    extra = json.loads(args.extra)
    server = serve(
        args.tree, MODELS["customvoice"][0], args.card, args.port, args.out / "serve.log",
        {"PYTHONPATH": f"{HERE / 'frame_probe'}:{args.tree}", "QWEN3_TTS_FRAME_PROBE": "1"},
    )
    (args.out / "serve.pgid").write_text(str(server.pid))
    try:
        wait_healthy(args.port, server)
        run_pass(url, texts[:WARMUP_REQUESTS], WARMUP_REQUESTS, {})
        for count in args.counts:
            for name, burst_extra, is_short, rounds in BURSTS:
                for index in range(rounds):
                    line = run_burst(url, args.port, name, burst_extra, is_short, count, args.out / "bursts" / f"{count}_{name}_r{index}")
                    report.append(f"burst count {count} {line}")
                    print(f"{time.strftime('%T')} {report[-1]}", flush=True)
        for load in args.loads:
            name, concurrency, requests = load.split(":")
            concurrency, requests = int(concurrency), int(requests)
            out = args.out / name
            out.mkdir(parents=True, exist_ok=True)
            run_pass(url, texts[-max(concurrency, 8):], concurrency, extra)
            profile = {"run_id": name, "event_dir": str(out / "events"), "event_buffer_capacity": EVENT_CAPACITY}
            httpx.post(f"{url}/start_request_profile", json=profile, timeout=60).raise_for_status()
            before = thread_ticks(server.pid)
            rows, wall_s = run_pass(url, texts[:requests], concurrency, extra)
            after = thread_ticks(server.pid)
            time.sleep(2)
            httpx.post(f"{url}/stop_request_profile", json={"run_id": name}, timeout=120).raise_for_status()
            time.sleep(3)
            with (out / "client.jsonl").open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            ticks = {thread: after.get(thread, 0) - before.get(thread, 0) for thread in after}
            ticks = dict(sorted(ticks.items(), key=lambda entry: -entry[1])[:14])
            (out / "threads.json").write_text(json.dumps({"wall_s": wall_s, "ticks": ticks}))
            memory = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader", "-i", str(args.card)],
                capture_output=True, text=True,
            ).stdout.strip()
            report.append(
                f"== {name}: concurrency {concurrency}, {requests} requests in {wall_s:.1f} s ({requests / wall_s:.2f} req/s), "
                f"card memory {memory}, CPU seconds per wall second by thread "
                f"{ {thread: round(value / os.sysconf('SC_CLK_TCK') / wall_s, 3) for thread, value in ticks.items()} }"
            )
            report.append(digest(out / "events", rows, 2 * concurrency if concurrency > 1 else 2))
            print(f"{time.strftime('%T')} {report[-2]}\n{report[-1]}", flush=True)
        report.append(f"codec counters {json.dumps(codec_state(url))}")
        print(report[-1], flush=True)
    finally:
        stop(server)
    (args.out / "trace.txt").write_text("\n".join(report) + "\n")
    print(f"{time.strftime('%T')} TRACE_DONE", flush=True)


if __name__ == "__main__":
    main()
