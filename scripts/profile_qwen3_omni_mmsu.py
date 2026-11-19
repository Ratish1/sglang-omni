# SPDX-License-Identifier: Apache-2.0
"""Measure the native MMSU Talker workload with optional stage and CUDA traces."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import requests

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.mmsu import MmsuSample, load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmsu import run
from sglang_omni.utils import find_available_port
from tests.test_model.conftest import (
    QWEN3_OMNI_FP8_TP2_THINKER_MEM_FRACTION,
    QWEN3_OMNI_TP2_TALKER_MEM_FRACTION,
    QWEN3_OMNI_TP2_THINKER_MAX_SEQ_LEN,
)
from tests.test_model.omni_ci_config import OMNI_CI_PRESETS
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import (
    CONCURRENCY,
    MAX_SAMPLES,
    build_args,
)

logger = logging.getLogger(__name__)
WARMUP_REQUEST_COUNT = 4
CONTROL_TIMEOUT_SECONDS = 30
STARTUP_TIMEOUT_SECONDS = 600
SEED = 20261119


def measure(
    samples: list[MmsuSample],
    port: int,
    output_directory: Path,
    seed: int | None,
    save_audio: bool,
) -> None:
    benchmark = build_args(OMNI_CI_PRESETS["qwen3-omni"], port, str(output_directory))
    benchmark.host = "127.0.0.1"
    benchmark.max_samples = len(samples)
    benchmark.max_concurrency = CONCURRENCY
    benchmark.disable_tqdm = True
    benchmark.save_audio = save_audio
    benchmark.seed = seed
    results = asyncio.run(run(benchmark, samples=samples, compute_wer=False))
    assert results["accuracy"]["failed_samples"] == 0, results["accuracy"]
    assert results["speed"]["audio_returned"] == len(samples), results["speed"]
    logger.info(f"{output_directory.name}: {results['speed']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--topology", required=True, choices=("bf16", "fp8-tp2"))
    parser.add_argument("--mode", required=True, choices=("measure", "nsys", "torch"))
    parser.add_argument("--seeded", action="store_true")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tree = arguments.tree.resolve(strict=True)
    checkpoint = arguments.checkpoint.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir()
    samples = load_mmsu_samples(
        max_samples=MAX_SAMPLES, repo_id=DATASETS["mmsu-ci-2000"]
    )
    assert len(samples) == MAX_SAMPLES
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": checkpoint.name,
                "serving_revision": subprocess.check_output(
                    ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
                ).strip(),
                "topology": arguments.topology,
                "mode": arguments.mode,
                "seed": SEED if arguments.seeded else None,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "input_sha256": hashlib.sha256(
                            Path(sample.audio_path).read_bytes()
                        ).hexdigest(),
                    }
                    for sample in samples
                ],
            },
            indent=2,
        )
    )
    port = find_available_port()
    if arguments.topology == "bf16":
        launch = [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--config",
            str(tree / "examples/configs/qwen3_omni_colocated_h200.yaml"),
        ]
    else:
        launch = [
            sys.executable,
            "-P",
            str(tree / "examples/run_qwen3_omni_speech_server.py"),
            "--thinker-max-seq-len",
            str(QWEN3_OMNI_TP2_THINKER_MAX_SEQ_LEN),
            "--thinker-tp-size",
            "2",
            "--gpu-thinker-tp",
            "0,1",
            "--gpu-talker",
            "1",
            "--gpu-code2wav",
            "1",
            "--thinker-mem-fraction-static",
            QWEN3_OMNI_FP8_TP2_THINKER_MEM_FRACTION,
            "--talker-mem-fraction-static",
            QWEN3_OMNI_TP2_TALKER_MEM_FRACTION,
        ]
    launch.extend(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            "qwen3-omni",
        ]
    )
    session_name = f"omni-{output.name}"
    if arguments.mode == "nsys":
        launch = [
            "nsys",
            "launch",
            f"--session-new={session_name}",
            "--trace=cuda,nvtx,osrt",
            "--cuda-graph-trace=node",
            *launch,
        ]
    else:
        pass
    server = start_server_from_cmd(
        launch,
        output / "server.log",
        port,
        timeout=STARTUP_TIMEOUT_SECONDS,
        env={"PYTHONPATH": str(tree)},
        strip_proxy=True,
    )
    try:
        measure(samples[:WARMUP_REQUEST_COUNT], port, output / "warmup", SEED, False)
        if arguments.mode == "measure":
            measure(
                samples,
                port,
                output / "result",
                SEED if arguments.seeded else None,
                True,
            )
        else:
            capture_torch = arguments.mode == "torch"
            start_route = "start_profile" if capture_torch else "start_request_profile"
            stop_route = "stop_profile" if capture_torch else "stop_request_profile"
            with requests.Session() as http:
                http.trust_env = False
                server_url = f"http://127.0.0.1:{port}"
                response = http.post(
                    f"{server_url}/{start_route}",
                    json={
                        "run_id": session_name,
                        "event_dir": str(output / "events"),
                        **(
                            {"trace_path_template": str(output / "{stage}")}
                            if capture_torch
                            else {}
                        ),
                    },
                    timeout=CONTROL_TIMEOUT_SECONDS,
                )
                response.raise_for_status()
                try:
                    if arguments.mode == "nsys":
                        subprocess.run(
                            [
                                "nsys",
                                "start",
                                f"--session={session_name}",
                                "--sample=none",
                                "--cpuctxsw=none",
                                f"--output={output / 'cuda'}",
                            ],
                            check=True,
                        )
                    else:
                        pass
                    try:
                        measure(
                            samples,
                            port,
                            output / "result",
                            SEED if arguments.seeded else None,
                            False,
                        )
                    finally:
                        if arguments.mode == "nsys":
                            subprocess.run(
                                ["nsys", "stop", f"--session={session_name}"],
                                check=True,
                            )
                        else:
                            pass
                finally:
                    http.post(
                        f"{server_url}/{stop_route}",
                        json={"run_id": session_name},
                        timeout=CONTROL_TIMEOUT_SECONDS,
                    ).raise_for_status()
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
