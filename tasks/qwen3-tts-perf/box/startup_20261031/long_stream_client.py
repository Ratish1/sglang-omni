"""Long streaming requests against a running server, timed with a client that only counts bytes, so its own event loop is
not the bound. The serving stress harness validates and keeps audio, which stalls its loop for seconds at concurrency 64
(block 106: 64 starts held 6.7 s after a wave ended, while decode queue wait was 2.3 ms), so it stays the contract check
and this is the clock. Each request is the harness's: the 4,096 character stress text, a 2,048 frame budget, the file
reference, streamed PCM. Prints first chunk, latency, frames a second over the busy span and the largest gap inside a
request. Runs under an existing server, or boots one with --tree.

usage: python3 long_stream_client.py --port 26000 --concurrency 64 --requests 128 --harness TREE --out DIR
       [--tree TREE --card 2 --barrier DIR --parties 2]
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from seeded_tail import MODEL, identity, stop_group, wait_healthy  # noqa: E402

SAMPLE_BYTES = 2
SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
BARRIER_TIMEOUT_S = 1200.0


def request_body(harness: str, seed: int, text: str, reference: str, reference_text: str, stream: bool = True) -> dict:
    return {
        "model": MODEL,
        "input": text,
        "voice": "default",
        "task_type": "Base",
        "ref_audio": reference,
        "ref_text": reference_text,
        "response_format": "pcm",
        "stream": stream,
        "speed": 1.0,
        "max_new_tokens": 2048,
        "seed": seed,
    }


async def one_request(session: aiohttp.ClientSession, url: str, body: dict) -> dict:
    start = time.perf_counter()
    first = None
    last = start
    largest_gap = 0.0
    received = 0
    async with session.post(url, json=body) as response:
        status = response.status
        async for chunk in response.content.iter_any():
            now = time.perf_counter()
            if first is None:
                first = now - start
            else:
                largest_gap = max(largest_gap, now - last)
            last = now
            received += len(chunk)
    return {"status": status, "start": start, "end": last, "first": first, "latency": last - start,
            "gap": largest_gap, "bytes": received}


async def run_load(args: argparse.Namespace, bodies: list[dict]) -> list[dict]:
    url = f"http://127.0.0.1:{args.port}/v1/audio/speech"
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=0)

    async def guarded(session: aiohttp.ClientSession, body: dict) -> dict:
        async with semaphore:
            return await one_request(session, url, body)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        return await asyncio.gather(*(guarded(session, body) for body in bodies))


def quantile(values: list[float], share: float) -> float:
    return sorted(values)[min(len(values) - 1, int(share * len(values)))]


def stress_bodies(harness: str, requests: int, stream: bool = True) -> list[dict]:
    """The harness's own long texts, reference and seeds, read from its scenario builder in its own tree."""
    script = (
        "import json,sys;"
        "from benchmarks.tts_serving.spec import load_spec;"
        "from benchmarks.tts_serving.scenarios import build_scenarios;"
        "s=load_spec('benchmarks/tts_serving/examples/stress_qwen3_tts_long_c64.json');"
        "print(json.dumps([x.payload['input'] for x in build_scenarios(s)]));"
        "print(json.dumps([s.params.file_ref_text]))"
    )
    env = dict(os.environ, PYTHONPATH=harness, CUDA_VISIBLE_DEVICES="")
    lines = subprocess.run([sys.executable, "-c", script], cwd=harness, env=env, check=True, capture_output=True,
                           text=True).stdout.splitlines()
    texts = json.loads(lines[-2])
    reference_text = json.loads(lines[-1])[0]
    reference = f"file://{Path(harness) / 'docs/_static/audio/male-voice.wav'}"
    return [request_body(harness, index, texts[index % len(texts)], reference, reference_text, stream) for index in range(requests)]


def start_server(tree: str, card: int, port: int, media_root: Path, log: Path) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=tree, SGLANG_OMNI_STRICT_PORT="1")
    command = [sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", MODEL, "--host", "127.0.0.1",
               "--port", str(port), "--tts_engine.gpu", str(card), "--vocoder.gpu", str(card),
               "--allowed-local-media-path", str(media_root)]
    with open(log, "w", encoding="utf-8") as handle:
        return subprocess.Popen(command, cwd=tree, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def wait_at_barrier(barrier: Path | None, parties: int, label: str) -> None:
    if barrier is None:
        return
    else:
        pass
    barrier.mkdir(parents=True, exist_ok=True)
    (barrier / label).write_text("", encoding="utf-8")
    deadline = time.monotonic() + BARRIER_TIMEOUT_S
    while len(list(barrier.iterdir())) < parties and time.monotonic() < deadline:
        time.sleep(0.05)


def report(label: str, rows: list[dict], out: Path) -> None:
    (out / "requests.json").write_text(json.dumps(rows), encoding="utf-8")
    good = [row for row in rows if row["status"] == 200 and row["first"] is not None]
    span = max(row["end"] for row in good) - min(row["start"] for row in good)
    samples = sum(row["bytes"] for row in good) / SAMPLE_BYTES
    frames = samples / FRAME_SAMPLES
    firsts = [row["first"] * 1000 for row in good]
    latencies = [row["latency"] for row in good]
    gaps = [row["gap"] * 1000 for row in good]
    rtf = [row["latency"] / (row["bytes"] / SAMPLE_BYTES / SAMPLE_RATE) for row in good]
    print(
        f"{label}: {len(good)} of {len(rows)} ok; span {span:.2f} s; {frames / span:.0f} frames/s; RTF mean {statistics.mean(rtf):.4f}; "
        f"first chunk ms mean {statistics.mean(firsts):.1f} p50 {quantile(firsts, 0.5):.1f} p95 {quantile(firsts, 0.95):.1f}; "
        f"latency s p50 {quantile(latencies, 0.5):.2f} p95 {quantile(latencies, 0.95):.2f} max {max(latencies):.2f}; "
        f"largest gap ms p50 {quantile(gaps, 0.5):.0f} p95 {quantile(gaps, 0.95):.0f} max {max(gaps):.0f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--requests", type=int, required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tree", default=None)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--label", default="run")
    parser.add_argument("--barrier", type=Path, default=None)
    parser.add_argument("--parties", type=int, default=2)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bodies = stress_bodies(args.harness, args.requests, not args.no_stream)
    server = None
    if args.tree is not None:
        print(f"{args.label}: tree {args.tree} {identity(args.tree)} card {args.card}", flush=True)
        server = start_server(args.tree, args.card, args.port, Path(args.harness) / "docs/_static/audio", args.out / "serve.log")
        wait_healthy(args.port, server)
    else:
        pass
    try:
        wait_at_barrier(args.barrier, args.parties, args.label)
        report(args.label, asyncio.run(run_load(args, bodies)), args.out)
    finally:
        if server is not None:
            stop_group(server, args.port)
            log = (args.out / "serve.log").read_text(errors="replace")
            print(f"{args.label}: Tracebacks {log.count('Traceback')}", flush=True)
        else:
            pass


if __name__ == "__main__":
    main()
