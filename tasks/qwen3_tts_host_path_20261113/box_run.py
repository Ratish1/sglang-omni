"""Unit suites, seeded byte identity and same-time pairs for the Qwen3-TTS host path arms."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUN_DIRECTORY = Path("/data/qwen3-tts-host-path-20261113")
REPOSITORY = Path("/workspace/sglang-omni")
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ARMS = {
    "main": "df7bc688cd75e882ee47fd005d93777dec6cfcc0",
    "a1": "46d17389ae67dad11154830bfc65f2148a655e7d",
    "m1": "78abe3af5f7eec60cb909ef76389343d56e5063e",
    "m2": "76ffc18324287001e69b01800d2a4680a7da7059",
    "m3": "312690bd81db2e7c14bdcef2136dde7e05ee991f",
    "stack": "3254d8151279e10feb846e6eb3ed5f777243aada",
}
UNIT_SUITES = [
    "tests/unit_test/qwen3_tts",
    "tests/unit_test/scheduling/test_session_row_admission.py",
    "tests/unit_test/pipeline/test_scheduler.py",
]
IDENTITY_SAMPLES = 24
PAIRS = [
    ("main", "main"),
    ("main", "stack"),
    ("main", "m1"),
    ("m1", "m2"),
    ("main", "m3"),
    ("main", "a1"),
]
PAIR_CELLS = [("stream_c64", 64, 256, True), ("stream_c16", 16, 128, True)]
NONSTREAM_CELL = ("nonstream_c64", 64, 256, False)


def tree_for(arm: str) -> Path:
    tree = RUN_DIRECTORY / "trees" / arm
    if not tree.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(tree), ARMS[arm]],
            cwd=REPOSITORY,
            check=True,
        )
    else:
        pass
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tree, text=True
    ).strip()
    assert actual == ARMS[arm], (arm, actual)
    return tree


def run_unit_suites(arms: list[str]) -> None:
    for arm in arms:
        tree = tree_for(arm)
        directory = RUN_DIRECTORY / "unit" / arm
        directory.mkdir(parents=True, exist_ok=True)
        suites = [suite for suite in UNIT_SUITES if (tree / suite).exists()]
        with (directory / "pytest.log").open("w") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    *suites,
                ],
                cwd=tree,
                env={**os.environ, "PYTHONPATH": str(tree)},
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=3600,
            )
        print(json.dumps({"unit": arm, "returncode": completed.returncode}), flush=True)


def start_server(arm: str, card: int, log_path: Path) -> tuple[subprocess.Popen, int]:
    port = 23600 + card
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [
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
            ],
            cwd=tree_for(arm),
            env={
                **os.environ,
                "PYTHONPATH": str(tree_for(arm)),
                "SGLANG_OMNI_STRICT_PORT": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (log_path.parent / f"{log_path.stem}.pgid").write_text(str(server.pid))
    return server, port


def wait_healthy(port: int, server: subprocess.Popen) -> None:
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server on port {port} exited: {server.returncode}")
        else:
            pass
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return
        except (OSError, TimeoutError):
            time.sleep(1)
    raise TimeoutError(f"server on port {port} did not become healthy")


def stop_servers(servers: list[subprocess.Popen]) -> None:
    for server in servers:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
        else:
            pass
    for server in servers:
        try:
            server.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait(timeout=15)


def benchmark(
    arm: str,
    port: int,
    directory: Path,
    concurrency: int,
    samples: int,
    stream: bool,
    seed: int | None,
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
        "--output-dir",
        str(directory / "bench"),
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    else:
        pass
    if stream:
        command.append("--stream")
    else:
        pass
    with (directory / "benchmark.log").open("w") as log:
        subprocess.run(
            command,
            cwd=tree_for(arm),
            env={**os.environ, "PYTHONPATH": str(tree_for(arm))},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1800,
        )
    return json.loads((directory / "bench/speed_results.json").read_text())["summary"]


def audio_digests(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((directory / "bench/audio").glob("*.wav"))
    }


def run_identity_modes(arm: str, port: int) -> None:
    for mode, stream in (("whole", False), ("stream", True)):
        benchmark(
            arm,
            port,
            RUN_DIRECTORY / "identity" / arm / mode,
            1,
            IDENTITY_SAMPLES,
            stream,
            1234,
        )


def run_identity(arms: list[str]) -> None:
    for first in range(0, len(arms), 2):
        batch = arms[first : first + 2]
        servers = []
        try:
            ports = {}
            for card, arm in enumerate(batch):
                server, port = start_server(
                    arm, card, RUN_DIRECTORY / "identity" / arm / "serve.log"
                )
                servers.append(server)
                ports[arm] = port
            for arm, server in zip(batch, servers):
                wait_healthy(ports[arm], server)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(run_identity_modes, arm, ports[arm])
                    for arm in batch
                ]
                for future in futures:
                    future.result()
        finally:
            stop_servers(servers)
    report = {}
    for mode in ("whole", "stream"):
        reference = audio_digests(RUN_DIRECTORY / "identity" / "main" / mode)
        for arm in arms:
            digests = audio_digests(RUN_DIRECTORY / "identity" / arm / mode)
            report[f"{arm}/{mode}"] = {
                "files": len(digests),
                "equal_to_main": sum(
                    digests.get(name) == digest for name, digest in reference.items()
                ),
            }
    (RUN_DIRECTORY / "identity" / "report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps({"identity": report}), flush=True)


def run_pairs(
    pairs: list[tuple[str, str]], cells: list[tuple[str, int, int, bool]]
) -> None:
    summaries_path = RUN_DIRECTORY / "pairs" / "summary.json"
    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = (
        json.loads(summaries_path.read_text()) if summaries_path.exists() else {}
    )
    for base, head in pairs:
        pair = f"{base}_vs_{head}"
        for order in (0, 1):
            servers = []
            try:
                ports = {}
                for side, arm in (("base", base), ("head", head)):
                    card = order if side == "base" else 1 - order
                    server, port = start_server(
                        arm,
                        card,
                        RUN_DIRECTORY
                        / "pairs"
                        / pair
                        / f"order{order}"
                        / f"serve_{side}.log",
                    )
                    servers.append(server)
                    ports[side] = (arm, port)
                for (arm, port), server in zip(ports.values(), servers):
                    wait_healthy(port, server)
                for cell, concurrency, samples, stream in cells:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=2
                    ) as executor:
                        futures = {
                            side: executor.submit(
                                benchmark,
                                arm,
                                port,
                                RUN_DIRECTORY
                                / "pairs"
                                / pair
                                / f"order{order}"
                                / side
                                / cell,
                                concurrency,
                                samples,
                                stream,
                                None,
                            )
                            for side, (arm, port) in ports.items()
                        }
                        for side, future in futures.items():
                            summary = future.result()
                            key = f"{pair}/order{order}/{side}/{cell}"
                            summaries[key] = summary
                            print(json.dumps({key: summary}), flush=True)
                            summaries_path.write_text(json.dumps(summaries, indent=2))
            finally:
                stop_servers(servers)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["unit", "identity", "pairs", "nonstream"])
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    arguments = parser.parse_args()
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if arguments.phase == "unit":
        run_unit_suites(arguments.arms)
    elif arguments.phase == "identity":
        run_identity(arguments.arms)
    elif arguments.phase == "pairs":
        run_pairs(PAIRS, PAIR_CELLS)
    else:
        run_pairs([("main", "stack"), ("m1", "m2")], [NONSTREAM_CELL])


if __name__ == "__main__":
    main()
