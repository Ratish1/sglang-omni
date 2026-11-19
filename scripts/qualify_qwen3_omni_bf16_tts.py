# SPDX-License-Identifier: Apache-2.0
"""Run full SeedTTS against one uninstrumented BF16 colocated server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import load_seedtts_samples
from benchmarks.eval.benchmark_omni_seedtts import (
    OmniSeedttsBenchmarkConfig,
    run_omni_seedtts_benchmark,
)
from sglang_omni.utils import find_available_port
from tests.test_model.test_qwen3_omni_tts_ci import CONCURRENCY

logger = logging.getLogger(__name__)
FULL_SEEDTTS_SAMPLE_COUNT = 1088
SERVER_TIMEOUT_SECONDS = 600


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--voice-probe", action="store_true")
    parser.add_argument("--seed", type=int)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    tree = arguments.tree.resolve(strict=True)
    checkpoint = arguments.checkpoint.resolve(strict=True)
    output = arguments.output.resolve()
    output.mkdir()

    samples = load_seedtts_samples(DATASETS["seedtts"])
    assert len(samples) == FULL_SEEDTTS_SAMPLE_COUNT
    (output / "inputs.json").write_text(
        json.dumps(
            {
                "serving_revision": subprocess.check_output(
                    ["git", "-C", str(tree), "rev-parse", "HEAD"], text=True
                ).strip(),
                "checkpoint": checkpoint.name,
                "seed": arguments.seed,
                "samples": [
                    {
                        "sample_id": sample.sample_id,
                        "text": sample.target_text,
                        "reference_audio_sha256": hashlib.sha256(
                            Path(sample.ref_audio).read_bytes()
                        ).hexdigest(),
                    }
                    for sample in samples
                ],
            },
            indent=2,
        )
    )
    port = find_available_port()
    server = start_server_from_cmd(
        [
            sys.executable,
            "-P",
            "-m",
            "sglang_omni.cli",
            "serve",
            "--config",
            str(tree / "examples/configs/qwen3_omni_colocated_h200.yaml"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            "qwen3-omni",
        ],
        output / "server.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        env={"PYTHONPATH": str(tree)},
        strip_proxy=True,
    )
    try:
        config = OmniSeedttsBenchmarkConfig(
            model="qwen3-omni",
            host="127.0.0.1",
            port=port,
            meta=DATASETS["seedtts"],
            voice_clone=True,
            max_concurrency=CONCURRENCY,
            output_dir=str(output / "generation"),
            seed=arguments.seed,
            disable_tqdm=True,
        )
        report = asyncio.run(run_omni_seedtts_benchmark(config))
        assert report["summary"]["completed_requests"] == FULL_SEEDTTS_SAMPLE_COUNT
        assert report["summary"]["failed_requests"] == 0
        logger.info(f"{output.name}: {report['summary']}")
        if arguments.voice_probe:
            subprocess.run(
                [
                    sys.executable,
                    "-P",
                    "-m",
                    "scripts.audit_qwen3_omni_voice_stream",
                    "--server-url",
                    f"http://127.0.0.1:{port}",
                    "--output",
                    str(output / "voice-stream.json"),
                ],
                check=True,
            )
        else:
            pass
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
