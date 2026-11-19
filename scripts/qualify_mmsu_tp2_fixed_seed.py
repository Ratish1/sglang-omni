# SPDX-License-Identifier: Apache-2.0
"""Repeat real MMSU Talker witnesses at the same sampling coordinates."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.mmsu import load_mmsu_samples
from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_mmsu import run
from sglang_omni.utils import find_available_port
from tests.test_model.conftest import (
    QWEN3_OMNI_FP8_TP2_THINKER_MEM_FRACTION,
    QWEN3_OMNI_TP2_TALKER_MEM_FRACTION,
    QWEN3_OMNI_TP2_THINKER_MAX_SEQ_LEN,
)
from tests.test_model.omni_ci_config import OMNI_CI_PRESETS
from tests.test_model.test_qwen3_omni_mmsu_talker_ci import MAX_SAMPLES, build_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--sample-ids", nargs="+", required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir()

    ci_samples = load_mmsu_samples(
        max_samples=MAX_SAMPLES, repo_id=DATASETS["mmsu-ci-2000"]
    )
    samples = [row for row in ci_samples if row.sample_id in arguments.sample_ids]
    assert {row.sample_id for row in samples} == set(arguments.sample_ids)
    input_hashes = {
        row.sample_id: hashlib.sha256(Path(row.audio_path).read_bytes()).hexdigest()
        for row in samples
    }

    port = find_available_port()
    server = start_server_from_cmd(
        [
            sys.executable,
            "examples/run_qwen3_omni_speech_server.py",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model-path",
            str(arguments.checkpoint),
            "--model-name",
            "qwen3-omni",
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
        ],
        arguments.output / "server.log",
        port,
        timeout=600,
        strip_proxy=True,
    )
    try:
        for repeat in ("a", "b"):
            output = arguments.output / repeat
            benchmark = build_args(OMNI_CI_PRESETS["qwen3-omni"], port, str(output))
            benchmark.seed = arguments.seed
            benchmark.max_samples = len(samples)
            benchmark.max_concurrency = 1
            benchmark.disable_tqdm = True
            results = asyncio.run(run(benchmark, samples=samples, compute_wer=False))
            assert results["accuracy"]["failed_samples"] == 0
            assert results["speed"]["audio_returned"] == len(samples)
            output_hashes = {
                row.sample_id: hashlib.sha256(
                    (output / "audio" / f"{row.sample_id}.wav").read_bytes()
                ).hexdigest()
                for row in samples
            }
            (output / "waveform-hashes.json").write_text(
                json.dumps({"inputs": input_hashes, "outputs": output_hashes}, indent=2)
            )
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
