# SPDX-License-Identifier: Apache-2.0
"""Validate native causal continuation and score its retained speech witness."""

import argparse
import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Literal

import torch
from compare_qwen3_omni_native_trajectories import validate
from qwen3_omni_corpus_support import SERVER_TIMEOUT_SECONDS, server_command

from benchmarks.benchmarker.utils import start_server_from_cmd, stop_server
from benchmarks.metrics.wer import SampleOutput
from benchmarks.tasks.asr import (
    QWEN3_ASR_MODEL_PATH,
    load_qwen3_asr,
    transcribe_and_compute_wer,
)
from sglang_omni.utils import find_available_port


def main(
    baseline: Path,
    intervention: Path,
    asr_checkpoint: Path,
    trial_kind: Literal["code", "cache", "continuation"],
) -> None:
    trial = json.loads((intervention / f"captures/{trial_kind}-trial.json").read_text())
    sample_id = trial["request_id"].removeprefix("feedback-trace-native-")
    roots = (baseline, intervention)
    frames = [validate(root / "captures" / sample_id) for root in roots]
    first_differences: dict[str, int] = {}
    for frame_index, (reference, forced) in enumerate(zip(*frames)):
        for name in (
            "predictor_query",
            "codec_ids",
            "feedback",
            "queued_feedback",
            "text_embedding",
            "stored_keys",
            "stored_values",
        ):
            if name in reference and name not in first_differences:
                if not torch.equal(reference[name], forced[name]):
                    first_differences[name] = frame_index
                else:
                    pass
            else:
                pass
    if trial_kind == "code":
        assert first_differences["codec_ids"] == trial["frame_index"]
        assert first_differences["feedback"] == trial["frame_index"]
        assert first_differences["queued_feedback"] == trial["frame_index"] + 1
    else:
        assert first_differences["stored_keys"] == trial["frame_index"]
        assert first_differences["stored_values"] == trial["frame_index"]
        for name in ("codec_ids", "feedback", "queued_feedback"):
            assert first_differences.get(name, len(frames[1])) > trial["frame_index"]
    if trial_kind == "continuation":
        assert first_differences["predictor_query"] == trial["frame_index"]
    else:
        assert (
            first_differences.get("predictor_query", len(frames[1]))
            > trial["frame_index"]
        )
    records = []
    for root in roots:
        matches = [
            row
            for row in json.loads((root / "mmsu/mmsu_results.json").read_text())[
                "per_sample"
            ]
            if row["sample_id"] == sample_id
        ]
        assert len(matches) == 1 and matches[0]["is_success"]
        records.append(matches[0])
    assert records[0]["raw_response"] == records[1]["raw_response"]
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
            "0",
            "--asr.engine.disable_speculative_decoding",
            "true",
        ],
        intervention / "witness-asr.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    scored = []
    waveform_hashes = []
    try:
        asr = load_qwen3_asr(port)
        for root, record in zip(roots, records, strict=True):
            waveform = root / "mmsu/audio" / f"{sample_id}.wav"
            waveform_hashes.append(sha256(waveform.read_bytes()).hexdigest())
            output = transcribe_and_compute_wer(
                SampleOutput(
                    sample_id=sample_id,
                    target_text=record["raw_response"],
                    audio_duration_s=record["audio_duration_s"],
                ),
                str(waveform),
                asr,
                "en",
                "cuda:0",
            )
            assert output.is_success and output.is_transcribed, output.error
            scored.append(asdict(output))
    finally:
        stop_server(server)
    report = {
        "roots": [str(root) for root in roots],
        "trial_kind": trial_kind,
        "sample_id": sample_id,
        "frames": [len(trajectory) for trajectory in frames],
        "first_differences": first_differences,
        "waveform_sha256": waveform_hashes,
        "scores": scored,
    }
    with (intervention / "witness-comparison.json").open("x") as receipt:
        json.dump(report, receipt, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("intervention", type=Path)
    parser.add_argument("asr_checkpoint", type=Path)
    parser.add_argument(
        "--trial-kind", choices=("code", "cache", "continuation"), default="code"
    )
    arguments = parser.parse_args()
    main(
        arguments.baseline,
        arguments.intervention,
        arguments.asr_checkpoint,
        arguments.trial_kind,
    )
else:
    pass
