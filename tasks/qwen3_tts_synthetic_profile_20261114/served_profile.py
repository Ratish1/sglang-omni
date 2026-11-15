"""One served SeedTTS EN cell under nsys: the server boots under `nsys launch` with one of
synthetic_driver.py's collection modes, the benchmark client drives the cell, and once the
load is steady one window of WINDOW_S seconds is recorded per concurrency."""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from served_baseline import MODELS, wait_healthy
from synthetic_driver import LAUNCH_OPTIONS, START_OPTIONS

HERE = Path(__file__).resolve().parent
WARMUP_S = 5.0
WINDOW_S = 3.0
# note (ratish): one pass of the 1,088 prompts lasts about 16 s at c64, so the client repeats it to outlast the window.
CLIENT_PASSES = 3
PYSPY_MODE = "pyspy"
PYSPY_WINDOW_S = 30
PYSPY_RATE_HZ = 100


def launch(tree: Path, mode: str, session: str, model: str, port: int, out: Path) -> subprocess.Popen:
    if mode == PYSPY_MODE:
        profiler = []
    else:
        profiler = ["nsys", "launch", f"--session-new={session}", *LAUNCH_OPTIONS[mode]]
    if mode == "attribution":
        merged = out / "marks_attribution.json"
        merged.write_text(
            json.dumps(
                json.loads((HERE / "marks.json").read_text())
                + json.loads((HERE / "marks_capture.json").read_text())
            )
        )
        profiler.append(f"--python-functions-trace={merged}")
    else:
        pass
    command = [
        *profiler, sys.executable, "-m", "sglang_omni.cli", "serve", "--model-path", model,
        "--host", "127.0.0.1", "--port", str(port), "--tts_engine.gpu", "0", "--vocoder.gpu", "0",
    ]
    with (out / "server.log").open("w") as log:
        server = subprocess.Popen(
            command, cwd=tree,
            env={
                **os.environ,
                "PYTHONPATH": f"{HERE / 'probe'}:{tree}",
                "QWEN3_TTS_PROBE_LABELS": "1",
                "SGLANG_OMNI_STRICT_PORT": "1",
            },
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    (out / "server.pgid").write_text(str(server.pid))
    return server


def client(tree: Path, model: str, extra: list[str], port: int, concurrency: int, directory: Path) -> subprocess.Popen:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-m", "benchmarks.eval.benchmark_tts_seedtts",
        "--meta", "zhaochenyang20/seed-tts-eval-arrow", "--model", model,
        "--use-existing-server", "--host", "127.0.0.1", "--port", str(port),
        "--generate-only", "--lang", "en", "--warmup", str(concurrency),
        "--max-concurrency", str(concurrency), "--stream",
        "--repeats", str(CLIENT_PASSES),
        "--output-dir", str(directory / "bench"), *extra,
    ]
    with (directory / "benchmark.log").open("w") as log:
        return subprocess.Popen(
            command, cwd=tree, env={**os.environ, "PYTHONPATH": str(tree)},
            stdout=log, stderr=subprocess.STDOUT,
        )


def record_window(session: str, mode: str, output: Path, server_pid: int) -> None:
    if mode == PYSPY_MODE:
        # note (ratish): --gil keeps only samples whose thread holds the interpreter lock, in every process of the server.
        subprocess.run(
            [
                "py-spy", "record", "--pid", str(server_pid), "--subprocesses", "--gil",
                "--nonblocking", "--threads", "--rate", str(PYSPY_RATE_HZ),
                "--duration", str(PYSPY_WINDOW_S), "--format", "raw",
                "--output", f"{output}.pyspy.txt",
            ],
            check=True,
        )
        return
    else:
        pass
    subprocess.run(
        ["nsys", "start", f"--session={session}", f"--output={output}", "--force-overwrite=true", *START_OPTIONS[mode]],
        check=True,
    )
    time.sleep(WINDOW_S)
    subprocess.run(["nsys", "stop", f"--session={session}"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=[*sorted(LAUNCH_OPTIONS), PYSPY_MODE], required=True)
    parser.add_argument("--model", choices=sorted(MODELS), default="base")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[64, 16])
    parser.add_argument("--port", type=int, default=24300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model, extra = MODELS[args.model]
    session = f"served-{args.mode}"
    server = launch(args.tree, args.mode, session, model, args.port, args.out)
    try:
        wait_healthy(args.port, server)
        print(f"{time.strftime('%T')} healthy", flush=True)
        for concurrency in args.concurrencies:
            cell = f"{args.model}_stream_c{concurrency}"
            load = client(args.tree, model, extra, args.port, concurrency, args.out / cell)
            time.sleep(WARMUP_S)
            if load.poll() is not None:
                raise RuntimeError(f"{cell} client ended before the window")
            else:
                pass
            record_window(session, args.mode, args.out / cell, server.pid)
            print(f"{time.strftime('%T')} {cell} window recorded", flush=True)
            load.wait(timeout=3600)
            print(f"{time.strftime('%T')} {cell} client rc {load.returncode}", flush=True)
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
        else:
            pass
    print(f"{time.strftime('%T')} served_profile done", flush=True)


if __name__ == "__main__":
    main()
