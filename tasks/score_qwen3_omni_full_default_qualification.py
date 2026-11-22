# SPDX-License-Identifier: Apache-2.0
"""Score complete SeedTTS and MMSU Talker qualification arms."""

import argparse
import json
import subprocess
from functools import partial
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import score_qwen3_omni_rounded_add_screen as prior_scorer
from qwen3_omni_corpus_support import (
    SERVER_TIMEOUT_SECONDS,
    seedtts_command,
    server_command,
)

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.tasks.asr import QWEN3_ASR_MODEL_PATH
from sglang_omni.utils import find_available_port

MMSU_ASR_CONCURRENCY = 4


def main(
    run_directory: Path,
    asr_checkpoint: Path,
    seedtts_only: bool,
    gpu_index: int,
    second_arm: str,
    stock_only: bool,
) -> None:
    arms = (
        (run_directory / "stock",)
        if stock_only
        else (run_directory / "stock", run_directory / second_arm)
    )
    identities = [(arm / "workload-identity.json").read_bytes() for arm in arms]
    assert all(identity == identities[0] for identity in identities)
    workload = json.loads(identities[0])
    stages = ("seedtts",) if seedtts_only else ("seedtts", "mmsu")
    expected_per_arm = sum(workload[f"{stage}_count"] for stage in stages)
    waveform_hashes = {}
    for arm in arms:
        waveforms = sorted(
            waveform
            for stage in stages
            for waveform in (arm / stage / "audio").glob("*.wav")
        )
        assert len(waveforms) == expected_per_arm, (arm, len(waveforms))
        waveform_hashes.update(
            {
                str(waveform.relative_to(run_directory)): sha256(
                    waveform.read_bytes()
                ).hexdigest()
                for waveform in waveforms
            }
        )
    hashes_path = run_directory / "waveform-hashes.json"
    if hashes_path.exists():
        assert json.loads(hashes_path.read_text()) == waveform_hashes
    else:
        with hashes_path.open("x") as hashes_file:
            json.dump(waveform_hashes, hashes_file, indent=2)

    port = find_available_port()
    server = start_server_from_cmd(
        [
            *server_command(),
            "--port",
            str(port),
            "--model-path",
            str(asr_checkpoint),
            "--model-name",
            QWEN3_ASR_MODEL_PATH,
            "--asr.gpu",
            str(gpu_index),
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        run_directory / "asr.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        with patch.multiple(
            prior_scorer,
            MMSU_SAMPLE_COUNT=workload["mmsu_count"],
            compute_text_audio_consistency_from_records=partial(
                prior_scorer.compute_text_audio_consistency_from_records,
                asr_concurrency=MMSU_ASR_CONCURRENCY,
            ),
        ):
            for arm in arms:
                if seedtts_only:
                    prior_scorer.score_seedtts(arm, port)
                else:
                    prior_scorer.score_arm(arm, port)
    finally:
        stop_server(server)
    for option, device in (
        ("--similarity-only", f"cuda:{gpu_index}"),
        ("--utmos-only", "cuda:0"),
    ):
        for arm in arms:
            subprocess.run(
                [
                    *seedtts_command(),
                    "--output-dir",
                    str(arm / "seedtts"),
                    "--device",
                    device,
                    option,
                ],
                check=True,
            )
    summaries = {}
    for arm in arms:
        summaries[arm.name] = {
            "seedtts": {
                metric_file: json.loads(
                    (arm / "seedtts" / f"{metric_file}.json").read_text()
                )["summary"]
                for metric_file in (
                    "speed_results",
                    "wer_results",
                    "returned-text-wer",
                    "similarity_results",
                    "utmos_results",
                )
            }
        }
        if not seedtts_only:
            mmsu = json.loads((arm / "mmsu/mmsu_results.json").read_text())
            summaries[arm.name]["mmsu"] = {
                "accuracy": mmsu["summary"],
                "speed": mmsu["speed_metrics"],
                "spoken_text_wer": mmsu["wer"]["summary"],
            }
        else:
            pass
    with (run_directory / "qualification-screen-results.json").open("x") as result_file:
        json.dump(
            {
                "workload": workload,
                "scored_stages": stages,
                "mmsu_asr_concurrency": None if seedtts_only else MMSU_ASR_CONCURRENCY,
                **summaries,
            },
            result_file,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("asr_checkpoint", type=Path)
    parser.add_argument("--seedtts-only", action="store_true")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument(
        "--second-arm", choices=("fixed", "stock_repeat"), default="fixed"
    )
    parser.add_argument("--stock-only", action="store_true")
    arguments = parser.parse_args()
    main(
        arguments.run_directory,
        arguments.asr_checkpoint,
        arguments.seedtts_only,
        arguments.gpu_index,
        arguments.second_arm,
        arguments.stock_only,
    )
else:
    pass
