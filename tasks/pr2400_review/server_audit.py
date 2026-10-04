import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(port: int, route: str, payload: dict[str, str]) -> bytes:
    response = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(response, timeout=60) as stream:
        return stream.read()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=("default", "ci"), required=True)
    parser.add_argument("--slack", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--base", action="store_true")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--port", type=int, default=18400)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", arguments.port))
    variant = "Base" if arguments.base else "CustomVoice"
    model = f"Qwen/Qwen3-TTS-12Hz-1.7B-{variant}"
    server = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(arguments.port),
        "--vocoder.factory.followup_urgent_slack_ms",
        str(arguments.slack),
    ]
    if arguments.layout == "ci":
        server += [
            "--vocoder.process",
            "vocoder",
            "--tts_engine.gpu_memory_fraction",
            "0.85",
            "--vocoder.gpu_memory_fraction",
            "0.10",
        ]
        if not arguments.base:
            server += [
                "--tts_engine.engine.max_running_requests",
                "64",
                "--tts_engine.engine.cuda_graph_max_bs",
                "64",
                "--tts_engine.engine.torch_compile_max_bs",
                "64",
            ]
    benchmark = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--generate-only",
        "--use-existing-server",
        "--stream",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(arguments.port),
        "--lang",
        "en",
        "--seed",
        "42",
        "--arrival-seed",
        "731",
        "--disable-tqdm",
    ]
    if arguments.base:
        benchmark += ["--ref-format", "references"]
    else:
        benchmark += ["--no-ref-audio", "--voice", "Ryan", "--task-type", "CustomVoice"]
    session = arguments.output.name
    launch = server
    if arguments.profile:
        launch = [
            "nsys",
            "launch",
            f"--session-new={session}",
            "--trace=cuda,nvtx,osrt",
            "--cuda-graph-trace=node",
            "--python-functions-trace=tasks/pr2400_review/nsys_functions.json",
            *server,
        ]
    environment = os.environ.copy()
    environment["SGLANG_OMNI_STRICT_PORT"] = "1"
    metadata = {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "server": server,
        "launch": launch,
        "benchmark": benchmark,
        "arguments": {**vars(arguments), "output": str(arguments.output)},
    }
    (arguments.output / "invocation.json").write_text(json.dumps(metadata, indent=2))
    with (arguments.output / "server.log").open("w") as server_log:
        process = subprocess.Popen(
            launch,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() not in (None, 0):
                    raise RuntimeError(f"server launch exited {process.returncode}")
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{arguments.port}/health", timeout=2
                    ):
                        break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server readiness deadline")
                    time.sleep(1)
            with (arguments.output / "warmup.log").open("w") as warmup_log:
                subprocess.run(
                    [
                        *benchmark,
                        "--max-samples",
                        "8",
                        "--concurrency",
                        "4",
                        "--output-dir",
                        str(arguments.output / "warmup"),
                    ],
                    stdout=warmup_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=600,
                )
            if arguments.profile:
                subprocess.run(
                    [
                        "nsys",
                        "start",
                        f"--session={session}",
                        "--sample=none",
                        "--cpuctxsw=process-tree",
                        "--output",
                        str(arguments.output / "trace"),
                    ],
                    check=True,
                )
                request(
                    arguments.port,
                    "start_request_profile",
                    {"run_id": session, "event_dir": str(arguments.output / "events")},
                )
                time.sleep(1)
            with (arguments.output / "benchmark.log").open("w") as benchmark_log:
                workload = [
                    *benchmark,
                    "--max-samples",
                    str(arguments.samples),
                    "--concurrency",
                    "64" if arguments.layout == "ci" else "16",
                    "--request-rate",
                    "44" if arguments.layout == "ci" else "inf",
                    "--output-dir",
                    str(arguments.output / "benchmark"),
                ]
                subprocess.run(
                    workload,
                    stdout=benchmark_log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=1200,
                )
            if arguments.profile:
                request(arguments.port, "stop_request_profile", {"run_id": session})
                subprocess.run(
                    ["nsys", "stop", f"--session={session}"], check=True, timeout=300
                )
        finally:
            if arguments.profile:
                subprocess.run(
                    ["nsys", "shutdown", f"--session={session}", "--kill=sigterm"],
                    timeout=60,
                )
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            deadline = time.monotonic() + 60
            while True:
                with socket.socket() as probe:
                    try:
                        probe.bind(("127.0.0.1", arguments.port))
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("server port was not released")
                        time.sleep(1)


if __name__ == "__main__":
    main()
