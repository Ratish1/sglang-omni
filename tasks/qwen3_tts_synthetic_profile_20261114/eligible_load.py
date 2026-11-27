"""Closed-loop load where every request is eligible for the bootstrap trim, two trees on one card.

Per order (a then b, then b then a), each tree boots alone on card N and serves, per concurrency, ROUNDS passes
over every SeedTTS EN text: streamed PCM, Ryan, English, default sampling, CONCURRENCY threads each sending its
next request when its last one ends. Above 24 live streams main never trims and the fix always does, so the
difference is the trim's cost and gain at load. Per request: first byte, first audible sound as played from the
first byte (DC-removed RMS over 20 ms windows, 10 ms hop, -45 dBFS for two windows in a row), audio seconds,
latency, and playback underruns (a piece arriving after the audio before it has played out). Writes
OUT/order<k>/<arm>/c<n>_r<i>.jsonl and prints one line per pass and a pooled table.

usage: python3 eligible_load.py --a-tree MAIN --b-tree FIX --out DIR [--card 0] [--concurrency 32 64] [--rounds 2]
"""

import argparse
import concurrent.futures
import json
import queue
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from seedtts_pair import serve, stop
from served_baseline import MODELS, wait_healthy

SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
SAMPLE_RATE = 24000
AUDIBLE_DBFS = -45.0
WINDOW_SAMPLES = 480
HOP_SAMPLES = 240
REQUEST_TIMEOUT_S = 300
WARMUP_REQUESTS = 64


def audible_onset(samples: np.ndarray) -> int | None:
    previous = False
    for start in range(0, max(0, samples.size - WINDOW_SAMPLES + 1), HOP_SAMPLES):
        window = samples[start : start + WINDOW_SAMPLES]
        window = window - float(window.mean())
        rms = float(np.sqrt(np.mean(window * window)))
        is_audible = rms > 0 and 20.0 * np.log10(rms) >= AUDIBLE_DBFS
        if is_audible and previous:
            return start - HOP_SAMPLES
        else:
            previous = is_audible
    return None


def run_request(client: httpx.Client, url: str, text: str) -> dict:
    body = {
        "model": MODELS["customvoice"][0], "input": text, "voice": "Ryan", "language": "English",
        "response_format": "pcm", "stream": True,
    }
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
    onset = audible_onset(samples)
    return {
        "status": 200, "samples": int(samples.size), "first_byte_ms": first_byte_s * 1000.0,
        "audible_ms": None if onset is None else (first_byte_s + onset / SAMPLE_RATE) * 1000.0,
        "leading_silence_ms": None if onset is None else 1000.0 * onset / SAMPLE_RATE,
        "latency_s": latency_s, "audio_s": samples.size / SAMPLE_RATE,
        "stalls": len(stalls), "stall_ms": 1000.0 * sum(stalls),
    }


def run_pass(url: str, texts: list[str], concurrency: int) -> tuple[list[dict], float]:
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
                row = run_request(client, url, text)
                with lock:
                    rows.append(row)

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for future in [pool.submit(worker) for _ in range(concurrency)]:
            future.result()
    return rows, time.monotonic() - start


def summarize(rows: list[dict], wall_s: float) -> dict:
    served = [row for row in rows if row["status"] == 200 and row.get("samples")]
    first = [row["first_byte_ms"] for row in served]
    audible = [row["audible_ms"] for row in served if row["audible_ms"] is not None]
    return {
        "ok": len(served), "n": len(rows), "req_s": len(served) / wall_s,
        "rtf": statistics.mean(row["latency_s"] / row["audio_s"] for row in served),
        "first_byte_mean": statistics.mean(first), "first_byte_median": statistics.median(first),
        "audible_mean": statistics.mean(audible), "audible_median": statistics.median(audible),
        "underrun_requests": sum(1 for row in served if row["stalls"]),
        "stall_ms": sum(row["stall_ms"] for row in served),
        "leading_silence_80": sum(1 for row in served if (row["leading_silence_ms"] or 0) >= 80),
    }


def line(summary: dict) -> str:
    return (
        f"ok {summary['ok']}/{summary['n']} req/s {summary['req_s']:.2f} rtf {summary['rtf']:.4f} "
        f"first byte mean/median {summary['first_byte_mean']:.1f}/{summary['first_byte_median']:.1f} ms "
        f"audible mean/median {summary['audible_mean']:.1f}/{summary['audible_median']:.1f} ms "
        f"underrun requests {summary['underrun_requests']} stall {summary['stall_ms']:.0f} ms "
        f"leading silence >= 80 ms {summary['leading_silence_80']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tree", type=Path, required=True)
    parser.add_argument("--b-tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--port", type=int, default=25000)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.a_tree))
    from benchmarks.dataset.seedtts import load_seedtts_samples

    texts = [sample.target_text for sample in load_seedtts_samples(SEEDTTS_META, split="en")]
    trees = {"a": args.a_tree, "b": args.b_tree}
    pooled: dict[tuple[str, int], list[dict]] = {}
    for order, arms in enumerate((("a", "b"), ("b", "a"))):
        for arm in arms:
            out = args.out / f"order{order}" / arm
            out.mkdir(parents=True, exist_ok=True)
            port = args.port + 400 * (2 * order + arms.index(arm))
            url = f"http://127.0.0.1:{port}"
            server = serve(trees[arm], MODELS["customvoice"][0], args.card, port, out / "serve.log", {})
            try:
                wait_healthy(port, server)
                run_pass(url, texts[:WARMUP_REQUESTS], WARMUP_REQUESTS)
                for concurrency in args.concurrency:
                    for index in range(args.rounds):
                        rows, wall_s = run_pass(url, texts, concurrency)
                        with (out / f"c{concurrency}_r{index}.jsonl").open("w") as handle:
                            for row in rows:
                                handle.write(json.dumps(row) + "\n")
                        summary = summarize(rows, wall_s)
                        pooled.setdefault((arm, concurrency), []).append(summary)
                        print(f"{time.strftime('%T')} order {order} {arm} c{concurrency} r{index} {line(summary)}", flush=True)
            finally:
                stop(server)
    keys = ("req_s", "rtf", "first_byte_mean", "first_byte_median", "audible_mean", "audible_median", "underrun_requests", "stall_ms")
    for concurrency in args.concurrency:
        for key in keys:
            values = {arm: [summary[key] for summary in pooled[(arm, concurrency)]] for arm in ("a", "b")}
            print(
                f"pooled c{concurrency} {key:18s} a {' '.join(f'{v:.4g}' for v in values['a'])} | "
                f"b {' '.join(f'{v:.4g}' for v in values['b'])} | means {statistics.mean(values['a']):.4g} -> "
                f"{statistics.mean(values['b']):.4g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
