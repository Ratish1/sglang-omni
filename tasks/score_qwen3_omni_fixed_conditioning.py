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
    original_results_sha256: str
    original_capture_sha256: str
    live_returned_text: str
    audio_duration_seconds: list[float]
    frame_counts: list[int]
    paired_frames: int
    changed_frames: dict[str, list[int]]
    peer_changed_frames: dict[str, list[int]]
    first_differences: dict[str, int]
    divergence_metadata: list[dict[str, str | int | bool | list[str]]]
    waveform_hashes: dict[str, str] = field(default_factory=dict)
    scores: dict[str, SampleOutput] = field(default_factory=dict)


def compare_conditioning(
    baseline: Path,
    intervention: Path,
    original_results: Path | None,
) -> list[ConditioningComparison]:
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
    if original_results is None:
        assert not (baseline / "conditioning-references.json").exists()
        original_results = baseline / "mmsu/mmsu_results.json"
        original_responses = responses[0]
    else:
        baseline_references = json.loads(
            (baseline / "conditioning-references.json").read_text()
        )
        assert baseline_references.keys() == references.keys()
        original_responses = {
            row["sample_id"]: row
            for row in json.loads(original_results.read_text())["per_sample"]
        }
    original_results_sha256 = sha256(original_results.read_bytes()).hexdigest()
    comparisons: list[ConditioningComparison] = []
    for sample_id in sorted(references):
        receipt = json.loads(
            (intervention / "captures" / f"{sample_id}-conditioning.json").read_text()
        )
        assert receipt["request_id"] == f"feedback-trace-native-{sample_id}"
        assert all(receipt["checks"].values())
        baseline_receipt_path = baseline / "captures" / f"{sample_id}-conditioning.json"
        if baseline_receipt_path.exists():
            baseline_receipt = json.loads(baseline_receipt_path.read_text())
            assert all(baseline_receipt["checks"].values())
            for name in (
                "request_id",
                "conditioning_sha256",
                "conditioning_receipt_sha256",
                "original_capture_sha256",
            ):
                assert baseline_receipt[name] == receipt[name], (sample_id, name)
        else:
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
        if baseline_receipt_path.exists():
            assert baseline_receipt["outcomes"]["emitted_frames"] == len(frames[0])
        else:
            pass
        changed_frames: dict[str, list[int]] = {}
        peer_changed_frames: dict[str, list[int]] = {}
        for frame_index, (original, forced) in enumerate(zip(*frames)):
            for name in (
                "prefill_embeddings",
                "input_ids",
                "positions",
                "rotary_positions",
                "predictor_query",
                "logits",
                "codec_ids",
                "feedback",
                "queued_feedback",
                "consumed_input",
                "text_embedding",
                "stored_keys",
                "stored_values",
                "attention_read_keys",
                "attention_read_values",
                *(
                    name
                    for name in original
                    if name.startswith("layer") and "_peer_" not in name
                ),
            ):
                if name in original:
                    if original[name].dtype != forced[name].dtype or not torch.equal(
                        original[name], forced[name]
                    ):
                        changed_frames.setdefault(name, []).append(frame_index)
                    else:
                        pass
                else:
                    pass
            for name in original:
                if (
                    name.startswith("layer")
                    and "_peer_" in name
                    and not torch.equal(original[name], forced[name])
                ):
                    peer_changed_frames.setdefault(name, []).append(frame_index)
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
        assert original_responses[sample_id]["is_success"]
        comparisons.append(
            ConditioningComparison(
                sample_id=sample_id,
                original_text=original_responses[sample_id]["raw_response"],
                original_results_sha256=original_results_sha256,
                original_capture_sha256=receipt["original_capture_sha256"],
                live_returned_text=responses[1][sample_id]["raw_response"],
                audio_duration_seconds=[
                    records[sample_id]["audio_duration_s"] for records in responses
                ],
                frame_counts=[len(trajectory) for trajectory in frames],
                paired_frames=min(len(trajectory) for trajectory in frames),
                changed_frames=changed_frames,
                peer_changed_frames=peer_changed_frames,
                first_differences=first_differences,
                divergence_metadata=(
                    [report["frames"][first_difference] for report in metadata]
                    if first_difference is not None
                    else []
                ),
            )
        )
    return comparisons


def main(
    baseline: Path,
    intervention: Path,
    asr_checkpoint: Path,
    original_results: Path | None,
) -> None:
    roots = (baseline, intervention)
    comparisons = compare_conditioning(baseline, intervention, original_results)
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
            for root, audio_duration_seconds in zip(
                roots, comparison.audio_duration_seconds, strict=True
            ):
                waveform = root / "mmsu/audio" / f"{comparison.sample_id}.wav"
                comparison.waveform_hashes[root.name] = sha256(
                    waveform.read_bytes()
                ).hexdigest()
                score = transcribe_and_compute_wer(
                    SampleOutput(
                        sample_id=comparison.sample_id,
                        target_text=comparison.original_text,
                        audio_duration_s=audio_duration_seconds,
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
    parser.add_argument(
        "--original-results",
        type=Path,
        help="Original MMSU results when both compared runs use held conditioning.",
    )
    arguments = parser.parse_args()
    main(
        arguments.baseline,
        arguments.intervention,
        arguments.asr_checkpoint,
        arguments.original_results,
    )
else:
    pass
