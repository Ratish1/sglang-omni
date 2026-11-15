"""Served SeedTTS EN at c1, c16 and c64 on one card: one fresh server per model, every
cell over the full 1,088 prompts, seeded, generation only; one summary JSON per run."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CONCURRENCIES = (1, 16, 64)
SEED = 1234
HEALTH_TIMEOUT_S = 1800
CELL_TIMEOUT_S = 3600
MODELS = {
    "base": (
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        ["--ref-format", "references"],
    ),
    "customvoice": (
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ["--no-ref-audio", "--voice", "Ryan"],
    ),
}


def wait_healthy(port: int, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server on port {port} exited with {server.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise TimeoutError(f"server on port {port} not healthy after {HEALTH_TIMEOUT_S} s")


def run_cell(
    tree: Path, model: str, extra: list[str], port: int, concurrency: int,
    stream: bool, directory: Path,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts",
        "--meta", "zhaochenyang20/seed-tts-eval-arrow", "--model", model,
        "--use-existing-server", "--host", "127.0.0.1", "--port", str(port),
        "--generate-only", "--lang", "en", "--warmup", str(concurrency),
        "--max-concurrency", str(concurrency), "--seed", str(SEED),
        "--output-dir", str(directory / "bench"), *extra,
    ]
    if stream:
        command.append("--stream")
    else:
        pass
    with (directory / "benchmark.log").open("w") as log:
        subprocess.run(
            command, cwd=tree, env={**os.environ, "PYTHONPATH": str(tree)},
            stdout=log, stderr=subprocess.STDOUT, check=True, timeout=CELL_TIMEOUT_S,
        )
    return json.loads((directory / "bench/speed_results.json").read_text())["summary"]


def serve_model(tree: Path, name: str, port: int, out: Path, cells: list[tuple[int, bool]]) -> None:
    model, extra = MODELS[name]
    command = [
        sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", model,
        "--host", "127.0.0.1", "--port", str(port),
        "--tts_engine.gpu", "0", "--vocoder.gpu", "0",
    ]
    with (out / f"serve_{name}.log").open("w") as log:
        server = subprocess.Popen(
            command, cwd=tree,
            env={**os.environ, "PYTHONPATH": str(tree), "SGLANG_OMNI_STRICT_PORT": "1"},
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    (out / f"serve_{name}.pgid").write_text(str(server.pid))
    try:
        wait_healthy(port, server)
        print(f"{time.strftime('%T')} healthy {name}", flush=True)
        for concurrency, stream in cells:
            cell = f"{name}_{'stream' if stream else 'whole'}_c{concurrency}"
            summary = run_cell(tree, model, extra, port, concurrency, stream, out / cell)
            (out / f"{cell}.json").write_text(json.dumps(summary, indent=2))
            print(f"{time.strftime('%T')} {cell} {json.dumps(summary)}", flush=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        else:
            pass
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=23500)
    parser.add_argument("--models", default="base,customvoice")
    parser.add_argument("--stream-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD", "HEAD:sglang_omni"], cwd=args.tree, text=True
    ).split()
    print(f"{time.strftime('%T')} tree {head[0]} sglang_omni {head[1]}", flush=True)
    for name in args.models.split(","):
        cells = [(concurrency, True) for concurrency in CONCURRENCIES]
        if name == "base" and not args.stream_only:
            cells += [(concurrency, False) for concurrency in CONCURRENCIES]
        else:
            pass
        serve_model(args.tree, name, args.port, args.out, cells)
    print(f"{time.strftime('%T')} served_baseline done", flush=True)


if __name__ == "__main__":
    main()
