# SPDX-License-Identifier: Apache-2.0
"""Capture selected speech trajectories under a full c32 SeedTTS workload."""

import argparse
import asyncio
import json
import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import run_qwen3_omni_rounded_add_screen as benchmark
from qwen3_omni_corpus_support import SERVER_TIMEOUT_SECONDS, server_command

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.utils import find_available_port

SELECTED_SAMPLES = (
    "common_voice_en_21555942-common_voice_en_21555950",
    "common_voice_en_24523505-common_voice_en_24523504",
    "common_voice_en_23639587-common_voice_en_23639589",
    "common_voice_en_19691556-common_voice_en_19691557",
)
STAGES = ("image_encoder", "audio_encoder", "thinker", "talker_ar", "code2wav")


def main(output: Path, checkpoint: Path, tactic_cache: Path) -> None:
    output.mkdir()
    samples = load_seedtts_samples(SEEDTTS_DATASET_ID)
    sample_ids = {sample.sample_id for sample in samples}
    assert len(samples) == len(sample_ids) == 1088
    assert set(SELECTED_SAMPLES) <= sample_ids
    shutil.copytree(tactic_cache, output / "cache")
    captures = output / "captures"
    captures.mkdir()
    port = find_available_port()
    server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(port),
            "--model-path",
            str(checkpoint),
            "--model-name",
            "qwen3-omni",
            "--config",
            "examples/configs/qwen3_omni_colocated_h200.yaml",
            *(argument for stage in STAGES for argument in (f"--{stage}.gpu", "0")),
        ],
        output / "server.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        env={
            "SGLANG_CACHE_DIR": str(output / "cache"),
            "OMNI_NATIVE_TRAJECTORY_DIRECTORY": str(captures),
            "OMNI_NATIVE_TRAJECTORY_SAMPLES": ",".join(SELECTED_SAMPLES),
        },
        strip_proxy=True,
    )
    try:
        with patch.object(benchmark, "CLIENT_CONCURRENCY", 32):
            asyncio.run(
                benchmark.run_seedtts(
                    output / "seedtts", port, sample_ids, trace_phase="native"
                )
            )
    finally:
        stop_server(server)
    for sample_id in SELECTED_SAMPLES:
        assert (captures / sample_id / "trajectory.pt").is_file()
        report = json.loads((captures / sample_id / "trajectory.json").read_text())
        assert report["finish_reason"] == "stop"
        assert len(report["frames"]) == len(report["output_ids"])
    print(json.dumps({"samples": len(samples), "captures": list(SELECTED_SAMPLES)}))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tactic_cache", type=Path)
    arguments = parser.parse_args()
    main(arguments.output, arguments.checkpoint, arguments.tactic_cache)
else:
    pass
