"""Unit suites, seeded byte identity and same-time pairs for the Qwen3-TTS host path arms."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

RUN_DIRECTORY = Path("/data/qwen3-tts-host-path-20261113")
REPOSITORY = Path("/workspace/sglang-omni")
PROBE_DIRECTORY = Path(__file__).resolve().parent / "probe"
ABORT_CLIENT = Path(__file__).resolve().parent / "abort_client.py"
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
S2PRO_MODEL = "fishaudio/s2-pro"
QWEN3_TTS_STAGES = ("tts_engine", "vocoder")
S2PRO_STAGES = ("preprocessing", "tts_engine", "vocoder")
S2PRO_CONFIG = ["--config", "examples/configs/s2pro_tts.yaml"]
EVENT_BUFFER_CAPACITY = 2_000_000
QUIET_S = 15
# Same Poisson arrival schedule for both arms of a fixed-rate cell.
ARRIVAL_SEED = 7
REQUEST_RATE_BY_CELL = {"stream_rate55": 55.0, "stream_rate45": 45.0}
ARMS = {
    "main": "df7bc688cd75e882ee47fd005d93777dec6cfcc0",
    "a1": "46d17389ae67dad11154830bfc65f2148a655e7d",
    "m1": "c843356795309ffe881117a5a0b2c4c6e0e8cef7",
    "m2": "4f3c2ab657a04f8ff554ea4e5c8b6a4fd1b305e6",
    "m3": "312690bd81db2e7c14bdcef2136dde7e05ee991f",
    "stack": "2d753d64d0279d0a579dac00704efd1d63525957",
    "main13": "13364eb45fd2ff09393fb0031576b2783a0edcf8",
    "pr": "2395d08957d87301e0fdd1284154051caba2cf79",
    "waits": "50fb3e3a8d5a86a9d55c73c4fc6cf987c86c45a0",
    "table_on_waits": "3a1c5b27948da1c211818132b02b17ee9d0209b9",
}
UNIT_SUITES = [
    "tests/unit_test/qwen3_tts",
    "tests/unit_test/preprocessing/test_cache_key.py",
    "tests/unit_test/scheduling/test_session_row_admission.py",
    "tests/unit_test/pipeline/test_scheduler.py",
]
WAITS_CELLS = [("stream_c64", 64, None, True), ("stream_c16", 16, None, True)]
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
    tree = RUN_DIRECTORY / "trees" / f"{arm}-{ARMS[arm][:9]}"
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


def start_server(
    arm: str,
    card: int,
    log_path: Path,
    *,
    model: str = MODEL,
    stages: tuple[str, ...] = QWEN3_TTS_STAGES,
    extra_args: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen, int]:
    port = 23600 + card
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stage_gpus = [
        argument for stage in stages for argument in (f"--{stage}.gpu", str(card))
    ]
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--model-path",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                *stage_gpus,
                *extra_args,
            ],
            cwd=tree_for(arm),
            env={
                **os.environ,
                "PYTHONPATH": str(tree_for(arm)),
                "SGLANG_OMNI_STRICT_PORT": "1",
                **(extra_env or {}),
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
    samples: int | None,
    stream: bool,
    seed: int | None,
    warmup: int | None = None,
    request_rate: float | None = None,
    model: str = MODEL,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_tts_seedtts",
        "--model",
        model,
        "--use-existing-server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--generate-only",
        "--lang",
        "en",
        "--warmup",
        str(concurrency if warmup is None else warmup),
        "--max-concurrency",
        str(concurrency),
        "--output-dir",
        str(directory / "bench"),
    ]
    if samples is not None:
        command.extend(["--max-samples", str(samples)])
    else:
        pass
    if request_rate is not None:
        command.extend(
            ["--request-rate", str(request_rate), "--arrival-seed", str(ARRIVAL_SEED)]
        )
    else:
        pass
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
            timeout=7200,
        )
    return json.loads((directory / "bench/speed_results.json").read_text())["summary"]


def probe_step_path(
    server: subprocess.Popen, directory: Path, stop: threading.Event
) -> None:
    """py-spy the server's stage process from the cell's start until its benchmarks end."""
    listing = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
    stage_pid = next(
        line.split()[0]
        for line in listing.splitlines()[1:]
        if line.split()[1] == str(server.pid) and "spawn_main" in line
    )
    dumps = []
    while not stop.is_set():
        dumps.append(
            subprocess.run(
                ["py-spy", "dump", "--pid", stage_pid, "--nonblocking"],
                capture_output=True,
                text=True,
            ).stdout
        )
        time.sleep(0.2)
    text = "\n".join(dumps)
    (directory / "step_probe_dumps.txt").write_text(text)
    counts = {
        name: text.count(f" {name} (")
        for name in (
            "handle_stream_step",
            "take_stream_step",
            "emit_stream_step",
            "write_step_frames",
            "gather_table_codes",
            "handle_stream_chunk",
            "stream_output_builder",
            "post_process_outputs",
            "sleep_during_idle",
        )
    }
    (directory / "step_probe.json").write_text(json.dumps(counts, indent=2))
    print(json.dumps({"step_probe": counts}), flush=True)


GIL_RECORD_DELAY_S = 8
GIL_RECORD_DURATION_S = 12


def stage_process_id(server: subprocess.Popen) -> str:
    listing = subprocess.check_output(["ps", "-eo", "pid,ppid,args"], text=True)
    return next(
        line.split()[0]
        for line in listing.splitlines()[1:]
        if line.split()[1] == str(server.pid) and "spawn_main" in line
    )


def record_gil_owners(server: subprocess.Popen, path: Path) -> None:
    """py-spy the stage process's GIL holders, by thread, from past the opening burst."""
    time.sleep(GIL_RECORD_DELAY_S)
    subprocess.run(
        [
            "py-spy",
            "record",
            "--pid",
            stage_process_id(server),
            "--gil",
            "--nonblocking",
            "--threads",
            "--format",
            "raw",
            "--duration",
            str(GIL_RECORD_DURATION_S),
            "--output",
            str(path),
        ],
        capture_output=True,
        check=True,
    )


def run_gil_owner_pairs(base: str, head: str, root: str) -> None:
    """Same-time stream c64 pairs in both card orders, recording each stage process's GIL holders."""
    for order in (0, 1):
        directory = RUN_DIRECTORY / root / f"order{order}"
        servers = []
        try:
            ports = {}
            for side, arm in (("base", base), ("head", head)):
                card = order if side == "base" else 1 - order
                server, port = start_server(arm, card, directory / f"serve_{side}.log")
                servers.append(server)
                ports[side] = (arm, port, server)
            for arm, port, server in ports.values():
                wait_healthy(port, server)
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                recordings = [
                    executor.submit(
                        record_gil_owners, server, directory / f"gil_{side}.txt"
                    )
                    for side, (_, _, server) in ports.items()
                ]
                futures = {
                    side: executor.submit(
                        benchmark,
                        arm,
                        port,
                        directory / side / "stream_c64",
                        64,
                        None,
                        True,
                        None,
                        1,
                    )
                    for side, (arm, port, _) in ports.items()
                }
                for side, future in futures.items():
                    key = f"{root}/order{order}/{side}/stream_c64"
                    print(json.dumps({key: future.result()}), flush=True)
                for recording in recordings:
                    recording.result()
        finally:
            stop_servers(servers)


def probe_env(arm: str, **values: str) -> dict[str, str]:
    """The arm's tree first, then the boot probe, whose switches are values."""
    return {"PYTHONPATH": f"{tree_for(arm)}:{PROBE_DIRECTORY}", **values}


def post_json(port: int, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read())


def run_event_pairs(
    base: str,
    head: str,
    root: str,
    cells: list[tuple[str, int, int | None, bool]],
    server_env: dict[str, str] | None = None,
) -> None:
    """Same-time pairs in both card orders, each request's first events recorded on both
    servers; server_env switches the boot probe on both, with a probe directory per side.
    """
    for order in (0, 1):
        directory = RUN_DIRECTORY / root / f"order{order}"
        servers = []
        try:
            ports = {}
            for side, arm in (("base", base), ("head", head)):
                card = order if side == "base" else 1 - order
                switches = {"QWEN3_TTS_FIRST_EVENTS": "1"}
                if server_env is not None:
                    switches.update(server_env)
                    switches["QWEN3_TTS_PROBE_DIR"] = str(directory / side / "probe")
                else:
                    pass
                server, port = start_server(
                    arm,
                    card,
                    directory / f"serve_{side}.log",
                    extra_env=probe_env(arm, **switches),
                )
                servers.append(server)
                ports[side] = (arm, port, server)
            for arm, port, server in ports.values():
                wait_healthy(port, server)
            for cell, concurrency, samples, stream in cells:
                for side, (arm, port, _) in ports.items():
                    post_json(
                        port,
                        "/start_request_profile",
                        {
                            "run_id": f"{root}-order{order}-{side}-{cell}",
                            "event_dir": str(directory / side / cell / "events"),
                            "event_buffer_capacity": EVENT_BUFFER_CAPACITY,
                        },
                    )
                with (directory / f"{cell}_dmon.txt").open("w") as dmon_log:
                    dmon = subprocess.Popen(
                        ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-o", "T"],
                        stdout=dmon_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                try:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=2
                    ) as executor:
                        futures = {
                            side: executor.submit(
                                benchmark,
                                arm,
                                port,
                                directory / side / cell,
                                concurrency,
                                samples,
                                stream,
                                None,
                                1,
                                REQUEST_RATE_BY_CELL.get(cell),
                            )
                            for side, (arm, port, _) in ports.items()
                        }
                        for side, future in futures.items():
                            key = f"{root}/order{order}/{side}/{cell}"
                            print(json.dumps({key: future.result()}), flush=True)
                finally:
                    os.killpg(dmon.pid, signal.SIGTERM)
                    dmon.wait(timeout=15)
                for arm, port, _ in ports.values():
                    post_json(port, "/stop_request_profile", {})
                time.sleep(QUIET_S)
        finally:
            stop_servers(servers)


def run_abort_rows(arms: list[str], root: str) -> None:
    """Drop most streams at set points on every arm at once, then a full c64 pass; the probe
    samples the code table's rows throughout and after."""
    directory = RUN_DIRECTORY / root
    probe = directory / "probe"
    probe.mkdir(parents=True, exist_ok=True)
    servers = []
    try:
        ports = {}
        for card, arm in enumerate(arms):
            server, port = start_server(
                arm,
                card,
                directory / arm / "serve.log",
                extra_env=probe_env(arm, QWEN3_TTS_PROBE_DIR=str(probe)),
            )
            servers.append(server)
            ports[arm] = (port, server)
        for port, server in ports.values():
            wait_healthy(port, server)

        def drop_then_serve(arm: str) -> None:
            port = ports[arm][0]
            with (directory / arm / "aborts.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        str(ABORT_CLIENT),
                        "--port",
                        str(port),
                        "--model",
                        MODEL,
                        "--output",
                        str(directory / arm / "aborts.json"),
                    ],
                    cwd=tree_for(arm),
                    env={**os.environ, "PYTHONPATH": str(tree_for(arm))},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=3600,
                )
            print(
                json.dumps(
                    {
                        f"{root}/{arm}/aborts": json.loads(
                            (directory / arm / "aborts.json").read_text()
                        )
                    }
                ),
                flush=True,
            )
            time.sleep(QUIET_S)
            summary = benchmark(
                arm, port, directory / arm / "after_aborts", 64, None, True, None, 1
            )
            print(json.dumps({f"{root}/{arm}/after_aborts": summary}), flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for future in [executor.submit(drop_then_serve, arm) for arm in arms]:
                future.result()
        time.sleep(QUIET_S)
    finally:
        stop_servers(servers)
    for path in sorted(probe.glob("rows_*.jsonl")):
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        print(
            json.dumps(
                {
                    "rows_probe": path.name,
                    "samples": len(samples),
                    "min_free": min(sample["free"] for sample in samples),
                    "last": samples[-1],
                }
            ),
            flush=True,
        )


CACHE_SAMPLES = 64
CACHE_PASSES = (("whole_a", False), ("whole_b", False), ("stream_a", True))


def run_cache_history(arm: str, root: str) -> None:
    """Seeded c1 whole, whole again, then stream, on a server with the engine's prefix cache
    (card 0) and one without (card 1)."""
    variants = {
        "radix": (),
        "no_radix": ("--tts_engine.engine.disable_radix_cache", "true"),
    }
    servers = []
    try:
        ports = {}
        for card, (name, extra_args) in enumerate(variants.items()):
            server, port = start_server(
                arm,
                card,
                RUN_DIRECTORY / root / name / "serve.log",
                extra_args=extra_args,
            )
            servers.append(server)
            ports[name] = (port, server)
        for port, server in ports.values():
            wait_healthy(port, server)

        def run_passes(name: str) -> None:
            for label, stream in CACHE_PASSES:
                benchmark(
                    arm,
                    ports[name][0],
                    RUN_DIRECTORY / root / name / label,
                    1,
                    CACHE_SAMPLES,
                    stream,
                    1234,
                    1,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for future in [executor.submit(run_passes, name) for name in variants]:
                future.result()
    finally:
        stop_servers(servers)
    report = {}
    for name in variants:
        digests = {
            label: audio_digests(RUN_DIRECTORY / root / name / label)
            for label, _ in CACHE_PASSES
        }
        frames = {
            label: {
                record["id"]: record["completion_tokens"]
                for record in json.loads(
                    (
                        RUN_DIRECTORY / root / name / label / "bench/speed_results.json"
                    ).read_text()
                )["per_request"]
            }
            for label, _ in CACHE_PASSES
        }
        report[name] = {
            "whole_a_vs_whole_b_bytes": sum(
                digests["whole_b"].get(file) == digest
                for file, digest in digests["whole_a"].items()
            ),
            "whole_a_vs_stream_a_frames": sum(
                frames["stream_a"][key] == count
                for key, count in frames["whole_a"].items()
            ),
            "whole_b_vs_stream_a_frames": sum(
                frames["stream_a"][key] == count
                for key, count in frames["whole_b"].items()
            ),
            "samples": len(digests["whole_a"]),
        }
    (RUN_DIRECTORY / root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"cache_history": report}), flush=True)


def run_other_model_identity(
    arms: list[str], root: str, extra_args: tuple[str, ...] = ()
) -> None:
    """Seeded c1 stream over the full corpus on S2-Pro; each arm boots after the previous is
    healthy, so both read one warm inductor cache."""
    servers = []
    try:
        ports = {}
        for card, arm in enumerate(arms):
            server, port = start_server(
                arm,
                card,
                RUN_DIRECTORY / root / arm / "serve.log",
                model=S2PRO_MODEL,
                stages=S2PRO_STAGES,
                extra_args=(*S2PRO_CONFIG, *extra_args),
            )
            servers.append(server)
            ports[arm] = port
            wait_healthy(port, server)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                arm: executor.submit(
                    benchmark,
                    arm,
                    ports[arm],
                    RUN_DIRECTORY / root / arm / "stream",
                    1,
                    None,
                    True,
                    1234,
                    1,
                    None,
                    S2PRO_MODEL,
                )
                for arm in arms
            }
            for arm, future in futures.items():
                print(json.dumps({f"{root}/{arm}": future.result()}), flush=True)
    finally:
        stop_servers(servers)
    digests = {
        arm: audio_digests(RUN_DIRECTORY / root / arm / "stream") for arm in arms
    }
    report = {
        arm: {
            "files": len(digests[arm]),
            "equal_to_first": sum(
                digests[arm].get(name) == digest
                for name, digest in digests[arms[0]].items()
            ),
        }
        for arm in arms
    }
    (RUN_DIRECTORY / root / "identity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"other_model_identity": report}), flush=True)


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
    pairs: list[tuple[str, str]],
    cells: list[tuple[str, int, int | None, bool]],
    root: str = "pairs",
    warmup: int | None = None,
    probe_arm: str | None = None,
    server_env: dict[str, str] | None = None,
) -> None:
    summaries_path = RUN_DIRECTORY / root / "summary.json"
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
                processes = {}
                for side, arm in (("base", base), ("head", head)):
                    card = order if side == "base" else 1 - order
                    server, port = start_server(
                        arm,
                        card,
                        RUN_DIRECTORY
                        / root
                        / pair
                        / f"order{order}"
                        / f"serve_{side}.log",
                        extra_env=(
                            None if server_env is None else probe_env(arm, **server_env)
                        ),
                    )
                    servers.append(server)
                    ports[side] = (arm, port)
                    processes[side] = server
                for (arm, port), server in zip(ports.values(), servers):
                    wait_healthy(port, server)
                for cell, concurrency, samples, stream in cells:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=3
                    ) as executor:
                        probe_stop = threading.Event()
                        for side, (arm, _) in ports.items():
                            if arm == probe_arm and cell == "stream_c64":
                                executor.submit(
                                    probe_step_path,
                                    processes[side],
                                    RUN_DIRECTORY / root / pair / f"order{order}",
                                    probe_stop,
                                )
                            else:
                                pass
                        futures = {
                            side: executor.submit(
                                benchmark,
                                arm,
                                port,
                                RUN_DIRECTORY
                                / root
                                / pair
                                / f"order{order}"
                                / side
                                / cell,
                                concurrency,
                                samples,
                                stream,
                                None,
                                warmup,
                                REQUEST_RATE_BY_CELL.get(cell),
                            )
                            for side, (arm, port) in ports.items()
                        }
                        try:
                            for side, future in futures.items():
                                summary = future.result()
                                key = f"{pair}/order{order}/{side}/{cell}"
                                summaries[key] = summary
                                print(json.dumps({key: summary}), flush=True)
                                summaries_path.write_text(
                                    json.dumps(summaries, indent=2)
                                )
                        finally:
                            probe_stop.set()
            finally:
                stop_servers(servers)


def run_seedtts_c1(arms: list[str], root: str) -> None:
    """Seeded c1 over the full SeedTTS EN corpus, arm i on card i, whole then stream."""
    servers = []
    try:
        ports = {}
        for card, arm in enumerate(arms):
            server, port = start_server(
                arm, card, RUN_DIRECTORY / root / arm / "serve.log"
            )
            servers.append(server)
            ports[arm] = port
        for arm, server in zip(arms, servers):
            wait_healthy(ports[arm], server)

        def run_modes(arm: str) -> None:
            for mode, stream in (("whole", False), ("stream", True)):
                summary = benchmark(
                    arm,
                    ports[arm],
                    RUN_DIRECTORY / root / arm / mode,
                    1,
                    None,
                    stream,
                    1234,
                    1,
                )
                print(json.dumps({f"{root}/{arm}/{mode}": summary}), flush=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            for future in [executor.submit(run_modes, arm) for arm in arms]:
                future.result()
    finally:
        stop_servers(servers)
    report = {}
    for mode in ("whole", "stream"):
        reference = audio_digests(RUN_DIRECTORY / root / arms[0] / mode)
        for arm in arms:
            digests = audio_digests(RUN_DIRECTORY / root / arm / mode)
            report[f"{arm}/{mode}"] = {
                "files": len(digests),
                "equal_to_first": sum(
                    digests.get(name) == digest for name, digest in reference.items()
                ),
            }
    (RUN_DIRECTORY / root / "identity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"seedtts_identity": report}), flush=True)


def score_generated(roots: list[str]) -> None:
    """WER on a Qwen3-ASR server on card 0 and SIM on card 1, for every generated run."""
    directories = sorted(
        path.parent
        for root in roots
        for path in (RUN_DIRECTORY / root).rglob("generated.json")
    )
    log_path = RUN_DIRECTORY / "score" / "asr_serve.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    port = 23700
    with log_path.open("w") as log:
        asr_server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang_omni.cli",
                "serve",
                "--model-path",
                ASR_MODEL,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=tree_for("main"),
            env={
                **os.environ,
                "PYTHONPATH": str(tree_for("main")),
                "SGLANG_OMNI_STRICT_PORT": "1",
            },
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (log_path.parent / "asr_serve.pgid").write_text(str(asr_server.pid))

    def score_one(directory: Path, scorer: list[str], log_name: str) -> None:
        with (directory / log_name).open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "benchmarks.eval.benchmark_tts_seedtts",
                    "--model",
                    MODEL,
                    "--lang",
                    "en",
                    "--output-dir",
                    str(directory),
                    *scorer,
                ],
                cwd=tree_for("main"),
                env={**os.environ, "PYTHONPATH": str(tree_for("main"))},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=7200,
            )
        print(json.dumps({"scored": str(directory), "by": log_name}), flush=True)

    try:
        wait_healthy(port, asr_server)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            transcription = executor.submit(
                lambda: [
                    score_one(
                        directory,
                        [
                            "--transcribe-only",
                            "--use-existing-server",
                            "--host",
                            "127.0.0.1",
                            "--port",
                            str(port),
                            "--asr-concurrency",
                            "64",
                        ],
                        "transcribe.log",
                    )
                    for directory in directories
                ]
            )
            similarity = executor.submit(
                lambda: [
                    score_one(
                        directory,
                        ["--similarity-only", "--device", "cuda:1"],
                        "similarity.log",
                    )
                    for directory in directories
                ]
            )
            transcription.result()
            similarity.result()
    finally:
        stop_servers([asr_server])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=[
            "unit",
            "identity",
            "pairs",
            "nonstream",
            "on_m2",
            "seedtts_c1",
            "seedtts_c1_swapped",
            "seedtts_pairs",
            "score",
            "quality_draws",
            "score_draws",
            "quality_draws_c64",
            "rate_probe",
            "events_c64",
            "retract",
            "abort_rows",
            "cache_history",
            "other_model",
            "other_model_eager",
            "events_c64_timed",
            "events_c64_priority",
            "events_c64_gil",
            "gil_owners_c64",
            "events_c64_encoder_priority",
            "events_rate45",
            "identity_waits",
            "events_waits",
            "events_table_on_waits",
        ],
    )
    parser.add_argument("--arms", nargs="*", default=list(ARMS))
    arguments = parser.parse_args()
    RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if arguments.phase == "unit":
        run_unit_suites(arguments.arms)
    elif arguments.phase == "identity":
        run_identity(arguments.arms)
    elif arguments.phase == "pairs":
        run_pairs(PAIRS, PAIR_CELLS)
    elif arguments.phase == "on_m2":
        run_pairs([("m2", "stack")], PAIR_CELLS, probe_arm="stack")
    elif arguments.phase == "seedtts_c1":
        run_seedtts_c1(["main", "m2"], "seedtts_c1")
    elif arguments.phase == "seedtts_c1_swapped":
        run_seedtts_c1(["m2", "main"], "seedtts_c1_swapped")
    elif arguments.phase == "seedtts_pairs":
        run_pairs(
            [("main", "m2")],
            [
                ("stream_c16", 16, None, True),
                ("stream_c64", 64, None, True),
                ("nonstream_c16", 16, None, False),
            ],
            root="seedtts_pairs",
            warmup=1,
            probe_arm="m2",
        )
    elif arguments.phase == "score":
        score_generated(["seedtts_c1", "seedtts_pairs"])
    elif arguments.phase == "quality_draws":
        for round_index in range(3):
            run_pairs(
                [("main", "m2")],
                [
                    ("stream_c16", 16, None, True),
                    ("nonstream_c16", 16, None, False),
                ],
                root=f"quality_draws/round{round_index}",
                warmup=1,
            )
    elif arguments.phase == "score_draws":
        score_generated(["quality_draws"])
    elif arguments.phase == "rate_probe":
        for round_index in range(2):
            run_pairs(
                [("main", "m2")],
                [("stream_rate55", 128, None, True)],
                root=f"rate_probe/round{round_index}",
                warmup=1,
            )
    elif arguments.phase == "quality_draws_c64":
        for round_index in range(3):
            run_pairs(
                [("main", "m2")],
                [("stream_c64", 64, None, True)],
                root=f"quality_draws_c64/round{round_index}",
                warmup=1,
            )
        score_generated(["quality_draws_c64"])
    elif arguments.phase == "events_c64":
        run_event_pairs("main13", "pr", "events_c64", [("stream_c64", 64, None, True)])
    elif arguments.phase == "retract":
        run_pairs(
            [("main13", "pr")],
            [("stream_c16", 16, None, True), ("nonstream_c16", 16, None, False)],
            root="retract",
            warmup=1,
            server_env={
                "SGLANG_TEST_RETRACT": "1",
                "QWEN3_TTS_PROBE_DIR": str(RUN_DIRECTORY / "retract" / "probe"),
            },
        )
        score_generated(["retract"])
    elif arguments.phase == "abort_rows":
        run_abort_rows(["main13", "pr"], "abort_rows")
    elif arguments.phase == "cache_history":
        run_cache_history("main13", "cache_history")
    elif arguments.phase == "other_model":
        run_other_model_identity(["main13", "pr"], "other_model")
    elif arguments.phase == "other_model_eager":
        run_other_model_identity(
            ["main13", "pr"],
            "other_model_eager",
            ("--tts_engine.engine.enable_torch_compile", "false"),
        )
    elif arguments.phase == "events_c64_timed":
        run_event_pairs(
            "main13",
            "pr",
            "events_c64_timed",
            [("stream_c64", 64, None, True)],
            server_env={"QWEN3_TTS_TIME_PREPROCESSING": "1"},
        )
    elif arguments.phase == "gil_owners_c64":
        run_gil_owner_pairs("main13", "pr", "gil_owners_c64")
    elif arguments.phase == "identity_waits":
        run_seedtts_c1(["main13", "waits"], "identity_waits")
    elif arguments.phase == "events_waits":
        run_event_pairs(
            "main13",
            "waits",
            "events_waits",
            WAITS_CELLS,
            server_env={"QWEN3_TTS_TIME_PREPROCESSING": "1"},
        )
    elif arguments.phase == "events_table_on_waits":
        run_event_pairs(
            "waits",
            "table_on_waits",
            "events_table_on_waits",
            WAITS_CELLS,
            server_env={"QWEN3_TTS_TIME_PREPROCESSING": "1"},
        )
    elif arguments.phase == "events_rate45":
        run_event_pairs(
            "main13",
            "pr",
            "events_rate45",
            [("stream_rate45", 128, None, True), ("stream_c64", 64, None, True)],
            server_env={"QWEN3_TTS_TIME_PREPROCESSING": "1"},
        )
    elif arguments.phase == "events_c64_encoder_priority":
        run_event_pairs(
            "main13",
            "pr",
            "events_c64_encoder_priority",
            [("stream_c64", 64, None, True)],
            server_env={
                "QWEN3_TTS_TIME_PREPROCESSING": "1",
                "QWEN3_TTS_PREPROCESSING_PRIORITY": "greatest_with_encoder",
            },
        )
    elif arguments.phase == "events_c64_gil":
        run_event_pairs(
            "main13",
            "pr",
            "events_c64_gil",
            [("stream_c64", 64, None, True)],
            server_env={"QWEN3_TTS_TIME_PREPROCESSING": "1"},
        )
    elif arguments.phase == "events_c64_priority":
        run_event_pairs(
            "main13",
            "pr",
            "events_c64_priority",
            [("stream_c64", 64, None, True)],
            server_env={
                "QWEN3_TTS_TIME_PREPROCESSING": "1",
                "QWEN3_TTS_PREPROCESSING_PRIORITY": "greatest",
            },
        )
    else:
        run_pairs([("main", "stack"), ("m1", "m2")], [NONSTREAM_CELL])


if __name__ == "__main__":
    main()
