"""Run unmodified PR bases and heads through the repository benchmarks on H100."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

MODEL_IDS = {
    "moss": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "higgs": "bosonai/higgs-audio-v3-tts-4b",
    "fish": "fishaudio/s2-pro",
}
STAGES = {
    "moss": ("asr",),
    "higgs": ("audio_encoder", "tts_engine", "vocoder"),
    "fish": ("tts_engine", "vocoder"),
}


def stop_server(server: subprocess.Popen[bytes]) -> None:
    if server.poll() is not None:
        return
    else:
        pass
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=30)


def wait_ready(server: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + 2400
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server port {port} exited with {server.returncode}")
        else:
            pass
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if response.status_code == 200:
                return
            else:
                pass
        except requests.RequestException:
            pass
        time.sleep(2)
    raise TimeoutError(f"server port {port} did not become ready")


def run_arm(
    model: str,
    arm: str,
    source: Path,
    output_root: Path,
    gpu_index: int,
    concurrencies: list[int],
    repeats: int,
    stream: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    port = 28980 + gpu_index
    command = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        MODEL_IDS[model],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    for stage in STAGES[model]:
        command.extend([f"--{stage}.gpu", str(gpu_index)])
    if model == "fish" and arm == "head":
        command.extend(["--preprocessing.gpu", str(gpu_index)])
    else:
        pass
    metadata = {
        "model": model,
        "arm": arm,
        "gpu": gpu_index,
        "command": command,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "source": str(source),
        "concurrencies": concurrencies,
        "repeats": repeats,
        "stream": stream,
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    with (output_root / "serve.log").open("wb") as serve_log:
        started = time.monotonic()
        server = subprocess.Popen(
            command,
            cwd=source,
            stdout=serve_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_ready(server, port)
            metadata["startup_seconds"] = time.monotonic() - started
            metadata["ready_memory"] = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.used,memory.total",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
            for concurrency in concurrencies:
                for repeat in range(repeats):
                    point = output_root / f"c{concurrency}-r{repeat + 1}"
                    point.mkdir(exist_ok=True)
                    if model == "moss":
                        benchmark = [
                            sys.executable,
                            "-m",
                            "benchmarks.eval.benchmark_asr_seedtts",
                            "--port",
                            str(port),
                            "--model-path",
                            MODEL_IDS[model],
                            "--concurrencies",
                            str(concurrency),
                            "--repeats",
                            "1",
                            "--output",
                            str(point / "results.json"),
                            "--lang",
                            "en",
                        ]
                    else:
                        benchmark = [
                            sys.executable,
                            "-m",
                            "benchmarks.eval.benchmark_tts_seedtts",
                            "--use-existing-server",
                            "--generate-only",
                            "--port",
                            str(port),
                            "--model",
                            MODEL_IDS[model],
                            "--max-concurrency",
                            str(concurrency),
                            "--output-dir",
                            str(point),
                            "--lang",
                            "en",
                        ]
                        if model == "higgs":
                            benchmark.extend(["--ref-format", "references"])
                        else:
                            pass
                        if stream:
                            benchmark.append("--stream")
                        else:
                            pass
                    with (point / "bench.log").open("wb") as bench_log:
                        subprocess.run(
                            benchmark,
                            cwd=source,
                            stdout=bench_log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    print(
                        f"completed {model} {arm} c{concurrency} r{repeat + 1}",
                        flush=True,
                    )
        finally:
            stop_server(server)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--arm", choices=("base", "head"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--concurrencies", default="16")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    arguments = parser.parse_args()
    run_arm(
        arguments.model,
        arguments.arm,
        arguments.source,
        arguments.output_root,
        arguments.gpu,
        [int(value) for value in arguments.concurrencies.split(",")],
        arguments.repeats,
        arguments.stream,
    )


if __name__ == "__main__":
    main()
