"""Where a slow first chunk spends its time. One tree on one card, SeedTTS EN streaming at 8 in flight with request events on,
beside the harness's long non-streaming requests at 8 in flight when --long-requests is above 0. Prints the client's first
chunk, then for each first-audio gap its median over all streaming requests and its mean over the slowest 1 % and 5 %, and what
the vocoder's first worker launched: each launched group's rows, width and launched-to-resolved span.

usage: python3 first_chunk_tail.py --tree TREE --card 0 --port 10000 --root DIR --harness TREE [--long-requests 32]
"""

import argparse
import collections
import concurrent.futures
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "next_20261031"))
sys.path.insert(0, str(HERE.parent / "profile_20261022"))

from first_audio_chain import CHAIN  # noqa: E402
from long_stream_client import start_server  # noqa: E402
from seeded_tail import bench, identity, stop_group, wait_healthy  # noqa: E402

CONCURRENCY_PER_CLIENT = 8
EVENT_BUFFER_CAPACITY = 4_000_000
EVENT_WRITE_TIMEOUT_S = 300
FIRST_CHUNK_SENT = ("vocoder", "stage_first_stream_chunk_sent")
LAUNCHED = "qwen3_tts_vocoder_decode_launched"
RESOLVED = "qwen3_tts_vocoder_decode_resolved"


def run_long(harness: str, port: int, requests: int, out: Path) -> str:
    done = subprocess.run(
        [sys.executable, str(HERE / "long_stream_client.py"), "--port", str(port), "--concurrency", str(CONCURRENCY_PER_CLIENT),
         "--requests", str(requests), "--harness", harness, "--out", str(out), "--label", "long", "--no-stream"],
        capture_output=True, text=True, check=False,
    )
    lines = [line for line in done.stdout.splitlines() if " of " in line and " ok;" in line]
    return lines[-1] if lines else f"long client rc {done.returncode}: {done.stderr[-300:]}"


def wait_for_event_files(event_dir: Path) -> None:
    """A stop does not wait for the writers, and each process writes its status file after its last event."""
    deadline = time.monotonic() + EVENT_WRITE_TIMEOUT_S
    while time.monotonic() < deadline:
        if len(list(event_dir.glob("run_status_*.json"))) >= len(list(event_dir.glob("events_*.jsonl"))):
            break
        else:
            time.sleep(1)
    for path in sorted(event_dir.glob("run_status_*.json")):
        status = json.loads(path.read_text())
        print(f"{path.name}: {status['written_events']} written, {status['dropped_events']} dropped, stages {status['stages']}", flush=True)


def read_events(event_dir: Path) -> list[dict]:
    events = []
    for path in sorted(event_dir.glob("events_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


def mean_of_slowest(values_by_request: dict[str, float], requests: list[str]) -> float:
    present = [values_by_request[request] for request in requests if request in values_by_request]
    return statistics.mean(present) if present else 0.0


def report(events: list[dict]) -> None:
    first: dict[str, dict[tuple[str, str], int]] = {}
    for event in events:
        milestones = first.setdefault(event["request_id"], {})
        key = (event["stage"], event["event_name"])
        milestones[key] = min(milestones.get(key, event["timestamp_ns"]), event["timestamp_ns"])
    streams = {request: milestones for request, milestones in first.items() if FIRST_CHUNK_SENT in milestones}
    totals = {
        request: (milestones[CHAIN[-1]] - milestones[CHAIN[0]]) / 1e6
        for request, milestones in streams.items()
        if CHAIN[0] in milestones and CHAIN[-1] in milestones
    }
    ranked = sorted(totals, key=totals.get, reverse=True)
    slowest_1 = ranked[: max(1, len(ranked) // 100)]
    slowest_5 = ranked[: max(1, len(ranked) // 20)]
    print(f"streams {len(streams)}, of which timed {len(totals)}; total ms median {statistics.median(totals.values()):.1f}, "
          f"slowest 5 % mean {mean_of_slowest(totals, slowest_5):.1f}, slowest 1 % mean {mean_of_slowest(totals, slowest_1):.1f}, "
          f"worst {max(totals.values()):.1f}")
    gaps: dict[str, dict[str, float]] = {}
    for request, milestones in streams.items():
        present = [key for key in CHAIN if key in milestones]
        for earlier, later in zip(present, present[1:]):
            name = f"{earlier[0]}.{earlier[1]} -> {later[0]}.{later[1]}"
            gaps.setdefault(name, {})[request] = (milestones[later] - milestones[earlier]) / 1e6
    print(f"{'median':>8} {'slow 5 %':>8} {'slow 1 %':>8}  gap, ms")
    for name, values in gaps.items():
        print(f"{statistics.median(values.values()):8.2f} {mean_of_slowest(values, slowest_5):8.2f} "
              f"{mean_of_slowest(values, slowest_1):8.2f}  {name}")
    launched: dict[str, list[dict]] = collections.defaultdict(list)
    resolved: dict[str, list[dict]] = collections.defaultdict(list)
    for event in events:
        if event["stage"] == "vocoder" and event["event_name"] == LAUNCHED:
            launched[event["request_id"]].append(event)
        elif event["stage"] == "vocoder" and event["event_name"] == RESOLVED:
            resolved[event["request_id"]].append(event)
        else:
            pass
    groups: dict[tuple[str, int], dict] = {}
    for request, entries in launched.items():
        for entry, done in zip(sorted(entries, key=lambda item: item["timestamp_ns"]),
                               sorted(resolved.get(request, []), key=lambda item: item["timestamp_ns"])):
            metadata = entry.get("metadata", {})
            if not str(metadata.get("worker", "")).startswith("qwen3-tts-initial") and "initial" not in str(metadata.get("worker", "")):
                continue
            else:
                pass
            groups[(request, entry["timestamp_ns"])] = {
                "launch": entry["timestamp_ns"],
                "span": (done["timestamp_ns"] - entry["timestamp_ns"]) / 1e6,
                "rows": metadata.get("cohort_size"),
                "width": metadata.get("fresh_frames"),
            }
    by_shape: dict[tuple[object, object], list[float]] = collections.defaultdict(list)
    for group in groups.values():
        by_shape[(group["width"], group["rows"])].append(group["span"])
    print("first worker groups by row, width and rows: count, launched-to-resolved ms median, max")
    for (width, rows), spans in sorted(by_shape.items(), key=lambda item: -sum(item[1]))[:16]:
        print(f"  {width}x{rows}: {len(spans)}, {statistics.median(spans):.2f}, {max(spans):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--long-requests", type=int, default=0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    print(f"tree {args.tree} {identity(args.tree)} card {args.card} long {args.long_requests}", flush=True)
    server = start_server(args.tree, args.card, args.port, Path("/"), args.root / "serve.log")
    long_line = "no long requests"
    try:
        wait_healthy(args.port, server)
        httpx.post(f"http://127.0.0.1:{args.port}/start_profile", timeout=60, trust_env=False, json={
            "run_id": "tail", "event_dir": str(args.root / "events"), "enable_torch": False,
            "event_buffer_capacity": EVENT_BUFFER_CAPACITY,
        }).raise_for_status()
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            stream_job = pool.submit(bench, args.tree, args.port, args.root / "stream", CONCURRENCY_PER_CLIENT, None, True)
            long_job = (
                pool.submit(run_long, args.harness, args.port, args.long_requests, args.root / "long")
                if args.long_requests
                else None
            )
            stream_job.result()
            long_line = long_job.result() if long_job is not None else long_line
        httpx.post(f"http://127.0.0.1:{args.port}/stop_profile", json={"run_id": "tail"}, timeout=600,
                   trust_env=False).raise_for_status()
        wait_for_event_files(args.root / "events")
    finally:
        stop_group(server, args.port)
    stream = json.loads((args.root / "stream" / "bench" / "speed_results.json").read_text())["summary"]
    log = (args.root / "serve.log").read_text(errors="replace")
    print(f"stream done {stream['completed_requests']}/{stream['total_requests']} first chunk mean "
          f"{1000 * stream['audio_ttfp_mean_s']:.1f} p50 {1000 * stream['audio_ttfp_median_s']:.1f} p95 "
          f"{1000 * stream['audio_ttfp_p95_s']:.1f} p99 {1000 * stream['audio_ttfp_p99_s']:.1f} ms RTF {stream['rtf_mean']:.4f}; "
          f"tracebacks {log.count('Traceback')}, left context {log.count('left-context decoder')}")
    print(long_line, flush=True)
    report(read_events(args.root / "events"))


if __name__ == "__main__":
    main()
