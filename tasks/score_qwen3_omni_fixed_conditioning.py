# SPDX-License-Identifier: Apache-2.0
"""Compare held-text native trajectories and score both clips against that text."""

import argparse
import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path

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


@dataclass(kw_only=True)
class ConditioningComparison:
    sample_id: str
    original_text: str
    live_returned_text: str
    frame_counts: list[int]
    paired_frames: int
    changed_frames: dict[str, list[int]]
    first_differences: dict[str, int]
    divergence_metadata: list[dict[str, str | int | bool | list[str]]]
    waveform_hashes: dict[str, str] = field(default_factory=dict)
    scores: dict[str, SampleOutput] = field(default_factory=dict)


def main(baseline: Path, intervention: Path, asr_checkpoint: Path) -> None:
    roots = (baseline, intervention)
    references = json.loads((intervention / "conditioning-references.json").read_text())
    responses = [
        {
            row["sample_id"]: row
            for row in json.loads((root / "mmsu/mmsu_results.json").read_text())[
                "per_sample"
            ]
        }
        for root in roots
    ]
    comparisons: list[ConditioningComparison] = []
    for sample_id in sorted(references):
        receipt = json.loads(
            (intervention / "captures" / f"{sample_id}-conditioning.json").read_text()
        )
        assert receipt["request_id"] == f"feedback-trace-native-{sample_id}"
        assert all(receipt["checks"].values())
        assert (
            receipt["original_capture_sha256"]
            == sha256(
                (baseline / "captures" / sample_id / "trajectory.pt").read_bytes()
            ).hexdigest()
        )
        frames = [validate(root / "captures" / sample_id) for root in roots]
        metadata = [
            json.loads((root / "captures" / sample_id / "trajectory.json").read_text())
            for root in roots
        ]
        assert receipt["outcomes"]["emitted_frames"] == len(frames[1])
        changed_frames: dict[str, list[int]] = {}
        for frame_index, (original, forced) in enumerate(zip(*frames)):
            for name in (
                "predictor_query",
                "logits",
                "codec_ids",
                "feedback",
                "queued_feedback",
                "consumed_input",
                "text_embedding",
                "stored_keys",
                "stored_values",
                *(name for name in original if name.startswith("layer0_")),
            ):
                if name in original:
                    if not torch.equal(original[name], forced[name]):
                        changed_frames.setdefault(name, []).append(frame_index)
                    else:
                        pass
                else:
                    pass
            for name in ("stored_keys", "stored_values"):
                for layer_index, (original_rows, forced_rows) in enumerate(
                    zip(original[name], forced[name], strict=True)
                ):
                    if not torch.equal(original_rows, forced_rows):
                        changed_frames.setdefault(
                            f"layer{layer_index}_{name}", []
                        ).append(frame_index)
                    else:
                        pass
        first_differences = {
            name: indices[0] for name, indices in changed_frames.items()
        }
        assert "text_embedding" not in first_differences
        first_difference = min(first_differences.values(), default=None)
        assert all(records[sample_id]["is_success"] for records in responses)
        comparisons.append(
            ConditioningComparison(
                sample_id=sample_id,
                original_text=responses[0][sample_id]["raw_response"],
                live_returned_text=responses[1][sample_id]["raw_response"],
                frame_counts=[len(trajectory) for trajectory in frames],
                paired_frames=min(len(trajectory) for trajectory in frames),
                changed_frames=changed_frames,
                first_differences=first_differences,
                divergence_metadata=(
                    [report["frames"][first_difference] for report in metadata]
                    if first_difference is not None
                    else []
                ),
            )
        )
    output = intervention / "fixed-conditioning-comparison.json"
    output.write_text(json.dumps([asdict(report) for report in comparisons], indent=2))
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
        intervention / "fixed-conditioning-asr.log",
        port,
        timeout=SERVER_TIMEOUT_SECONDS,
        strip_proxy=True,
    )
    try:
        asr = load_qwen3_asr(port)
        for comparison in comparisons:
            for root, records in zip(roots, responses, strict=True):
                waveform = root / "mmsu/audio" / f"{comparison.sample_id}.wav"
                comparison.waveform_hashes[root.name] = sha256(
                    waveform.read_bytes()
                ).hexdigest()
                score = transcribe_and_compute_wer(
                    SampleOutput(
                        sample_id=comparison.sample_id,
                        target_text=comparison.original_text,
                        audio_duration_s=records[comparison.sample_id][
                            "audio_duration_s"
                        ],
                    ),
                    str(waveform),
                    asr,
                    "en",
                    "cuda:0",
                )
                assert score.is_success and score.is_transcribed, score.error
                comparison.scores[root.name] = score
                output.write_text(
                    json.dumps([asdict(report) for report in comparisons], indent=2)
                )
                print(
                    json.dumps(
                        {
                            "sample_id": comparison.sample_id,
                            "arm": root.name,
                            "wer": score.wer,
                        }
                    ),
                    flush=True,
                )
    finally:
        stop_server(server)
    assert all(len(report.scores) == len(roots) for report in comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("intervention", type=Path)
    parser.add_argument("asr_checkpoint", type=Path)
    arguments = parser.parse_args()
    main(arguments.baseline, arguments.intervention, arguments.asr_checkpoint)
else:
    pass
