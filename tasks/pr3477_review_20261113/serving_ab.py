"""Run pinned base/head serving comparisons on the two assigned container GPUs."""

import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUN_DIRECTORY = Path("/data/pr3477-review-20261113")
REPOSITORY = Path("/workspace/sglang-omni")
BASE_COMMIT = "103b9fbf9a55c4f19c5c4ede569e0938d8857377"
HEAD_COMMIT = "a0888506bff67ff1318ac7224f8697fd1d3688a1"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def wait_healthy(port: int, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"Server on port {port} exited: {server.returncode}")
        else:
            pass
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except (OSError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"Server on port {port} did not become healthy")


def benchmark(
    tree: Path, port: int, directory: Path, concurrency: int, samples: int, stream: bool
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--model",
        MODEL,
        "--use-existing-server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--generate-only",
        "--lang",
        "en",
        "--warmup",
        str(concurrency),
        "--max-concurrency",
        str(concurrency),
        "--max-samples",
        str(samples),
        "--seed",
        "1234",
        "--output-dir",
        str(directory / "bench"),
    ]
    if stream:
        command.append("--stream")
    else:
        pass
    with (directory / "benchmark.log").open("w") as log:
        subprocess.run(
            command,
            cwd=tree,
            env={**os.environ, "PYTHONPATH": str(tree)},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1200,
        )
    return json.loads((directory / "bench/speed_results.json").read_text())["summary"]


def main() -> None:
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    trees = {"base": RUN_DIRECTORY / "base", "head": REPOSITORY}
    if not trees["base"].exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(trees["base"]), BASE_COMMIT],
            cwd=REPOSITORY,
            check=True,
        )
    else:
        pass
    for name, commit in (("base", BASE_COMMIT), ("head", HEAD_COMMIT)):
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=trees[name], text=True
        ).strip()
        assert actual == commit, (name, actual, commit)
    summaries = {}
    cells = [
        ("nonstream_c16", 16, 256, False),
        ("nonstream_c64", 64, 256, False),
        ("nonstream_c1", 1, 32, False),
        ("stream_c16", 16, 256, True),
    ]
    for order in (0, 1):
        servers = []
        try:
            for name in ("base", "head"):
                card = order if name == "base" else 1 - order
                port = 23477 + card
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
                    str(port),
                    "--tts_engine.gpu",
                    str(card),
                    "--vocoder.gpu",
                    str(card),
                ]
                with (RUN_DIRECTORY / f"serve_{order}_{name}.log").open("w") as log:
                    server = subprocess.Popen(
                        command,
                        cwd=trees[name],
                        env={
                            **os.environ,
                            "PYTHONPATH": str(trees[name]),
                            "SGLANG_OMNI_STRICT_PORT": "1",
                        },
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                servers.append((name, port, server))
            for name, port, server in servers:
                wait_healthy(port, server)
                print(f"healthy order={order} arm={name} port={port}", flush=True)
            for cell, concurrency, samples, stream in cells:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {
                        name: executor.submit(
                            benchmark,
                            trees[name],
                            port,
                            RUN_DIRECTORY / f"order{order}/{name}/{cell}",
                            concurrency,
                            samples,
                            stream,
                        )
                        for name, port, _ in servers
                    }
                    for name, future in futures.items():
                        summary = future.result()
                        summaries[f"{order}/{name}/{cell}"] = summary
                        print(
                            json.dumps(
                                {
                                    "order": order,
                                    "arm": name,
                                    "cell": cell,
                                    "summary": summary,
                                }
                            ),
                            flush=True,
                        )
                        (RUN_DIRECTORY / "serving_summary.json").write_text(
                            json.dumps(summaries, indent=2)
                        )
        finally:
            for _, _, server in servers:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                else:
                    pass
            for _, _, server in servers:
                try:
                    server.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=15)


if __name__ == "__main__":
    main()
