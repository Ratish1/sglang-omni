"""A real server through one injected replay failure on one incremental Codec runner. Three rounds of short and long requests,
streamed then not, each read after the last: requests that completed, audio bytes, the runners' state from model_info, whether the
server still answers, and how the log names the failure. A runner sharing a capture stream with the failed one must not replay
freed memory: on a tree without the fix the next replay of a sibling kills the vocoder.

usage: cd TREE && python3 graph_fault_read.py --tree TREE --harness TREE --card 0 --port 4100 --mode tail --replay 3 --out DIR
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "next_20261031"))

from seeded_tail import MODEL, identity, stop_group, wait_healthy  # noqa: E402

ROUNDS = ("before", "after", "again")
REQUESTS_PER_TEXT = 4
CONCURRENCY = 4
REQUEST_TIMEOUT_S = 300
FAULT_PATH = Path(__file__).resolve().parent / "graph_fault"
TEXTS = (
    "Hi.",
    "A short sentence to decode.",
    "This is a longer sentence, written so that the request keeps decoding well past its first chunk of audio, and then for a while.",
)


async def one_request(session: aiohttp.ClientSession, url: str, body: dict) -> dict:
    received = 0
    try:
        async with session.post(url, json=body) as response:
            status = response.status
            async for chunk in response.content.iter_any():
                received += len(chunk)
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        return {
            "status": type(exc).__name__,
            "bytes": received,
            "stream": body["stream"],
        }
    return {"status": status, "bytes": received, "stream": body["stream"]}


async def run_round(port: int, bodies: list[dict]) -> list[dict]:
    url = f"http://127.0.0.1:{port}/v1/audio/speech"
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def guarded(session: aiohttp.ClientSession, body: dict) -> dict:
        async with semaphore:
            return await one_request(session, url, body)

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=0)
    ) as session:
        return await asyncio.gather(*(guarded(session, body) for body in bodies))


def runner_states(port: int) -> dict:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/model_info", timeout=30
        ) as response:
            info = json.loads(response.read())
    except (OSError, ValueError) as exc:
        return {"unreachable": f"{type(exc).__name__}: {exc}"[:160]}
    found: dict = {}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for name, value in node.items():
                if name == "cuda_graphs":
                    found.update(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        else:
            pass

    walk(info)
    states: dict = {}
    for name, stats in found.items():
        for index, one in enumerate(stats if isinstance(stats, list) else [stats]):
            label = f"{name}{index}" if isinstance(stats, list) else name
            runtime = one.get("runtime", {})
            states[label] = {
                "enabled": one.get("enabled"),
                "reason": (one.get("disable_reason") or "")[:60],
                "replays": runtime.get("replays"),
                "failures": runtime.get("replay_failures"),
                "graphs": len(one.get("build", {}).get("captured_keys", [])),
            }
    return states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("cold", "window", "tail", "warm"), required=True
    )
    parser.add_argument("--replay", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=f"{args.tree}:{FAULT_PATH}",
        SGLANG_OMNI_STRICT_PORT="1",
        QTTS_GRAPH_FAULT=f"{args.mode}:{args.replay}",
    )
    audio = Path(args.harness) / "docs/_static/audio"
    reference_script = (
        "import json;from benchmarks.tts_serving.spec import load_spec;"
        "print(json.dumps(load_spec('benchmarks/tts_serving/examples/stress_qwen3_tts_long_c64.json').params.file_ref_text))"
    )
    reference_text = json.loads(
        subprocess.run(
            [sys.executable, "-c", reference_script],
            cwd=args.harness,
            check=True,
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONPATH=args.harness, CUDA_VISIBLE_DEVICES=""),
        ).stdout.splitlines()[-1]
    )
    log = args.out / "serve.log"
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tts_engine.gpu",
        str(args.card),
        "--vocoder.gpu",
        str(args.card),
        "--allowed-local-media-path",
        str(audio),
    ]
    with open(log, "w", encoding="utf-8") as handle:
        server = subprocess.Popen(
            command,
            cwd=args.tree,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    row = {"tree": identity(args.tree), "mode": args.mode, "replay": args.replay}
    try:
        wait_healthy(args.port, server)
        row["runners_boot"] = runner_states(args.port)
        for index, name in enumerate(ROUNDS):
            bodies = [
                {
                    "model": MODEL,
                    "input": text,
                    "voice": "default",
                    "task_type": "Base",
                    "ref_audio": f"file://{audio / 'male-voice.wav'}",
                    "ref_text": reference_text,
                    "response_format": "pcm",
                    "stream": stream,
                    "speed": 1.0,
                    "max_new_tokens": 2048,
                    "seed": seed + 10 * index,
                }
                for stream in (True, False)
                for seed, text in enumerate(TEXTS * REQUESTS_PER_TEXT)
            ]
            started = time.monotonic()
            results = asyncio.run(run_round(args.port, bodies))
            statuses: dict = {}
            for result in results:
                label = (
                    f"{'stream' if result['stream'] else 'whole'}_{result['status']}"
                )
                statuses[label] = statuses.get(label, 0) + 1
            empty = sum(
                1
                for result in results
                if result["status"] == 200 and result["bytes"] == 0
            )
            row[name] = {
                "statuses": statuses,
                "empty_200": empty,
                "bytes": sum(result["bytes"] for result in results),
                "seconds": round(time.monotonic() - started, 1),
                "runners": runner_states(args.port),
                "server_exit": server.poll(),
            }
            print(
                "ROUND " + json.dumps({"mode": args.mode, "round": name, **row[name]}),
                flush=True,
            )
    finally:
        stop_group(server, args.port)
    text = log.read_text(errors="replace")
    row["log"] = {
        "tracebacks": text.count("Traceback"),
        "injected": text.count("injected replay failure"),
        "illegal": text.count("ILLEGAL_ADDRESS"),
        "sticky": text.count("CUDA_ERROR"),
        "left_context_fallbacks": text.count("left-context"),
    }
    print("FAULT " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
