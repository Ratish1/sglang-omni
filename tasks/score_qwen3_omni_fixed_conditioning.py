# SPDX-License-Identifier: Apache-2.0
"""Compare held-text native trajectories and score both clips against that text."""

import argparse
import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Literal

import torch
from compare_qwen3_omni_native_trajectories import validate
from qwen3_omni_corpus_support import SERVER_TIMEOUT_SECONDS, server_command
from score_qwen3_omni_reference_continuation import RenderedClip, render

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
    unpaired_deep_frames: dict[str, list[int]]
    waveform_source: Literal["native", "common_decoder"] = "native"
    rendered_clips: list[RenderedClip] = field(default_factory=list)
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
        unpaired_deep_frames: dict[str, list[int]] = {
            "baseline": [],
            "intervention": [],
        }
        for frame_index, (original, forced) in enumerate(zip(*frames)):
            captured_layers = [
                record["frames"][frame_index].get("captured_layers")
                for record in metadata
            ]
            paired_deep_frame = all(count is not None for count in captured_layers)
            if paired_deep_frame:
                assert captured_layers[0] == captured_layers[1]
            else:
                for arm, count in zip(
                    unpaired_deep_frames, captured_layers, strict=True
                ):
                    if count is not None:
                        unpaired_deep_frames[arm].append(frame_index)
                    else:
                        pass
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
                    if name.startswith("layer")
                    and "_peer_" not in name
                    and (paired_deep_frame or name.startswith("layer0_"))
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
                    and (
                        paired_deep_frame
                        or name
                        in (
                            "layer0_peer_mlp_normalized",
                            "layer0_peer_expert_ids",
                            "layer0_peer_expert_weights",
                        )
                    )
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
                unpaired_deep_frames=unpaired_deep_frames,
            )
        )
    return comparisons


def main(
    baseline: Path,
    intervention: Path,
    asr_checkpoint: Path,
    original_results: Path | None,
    decoder_checkpoint: Path | None,
    sample_rate: int | None,
) -> None:
    assert (decoder_checkpoint is None) == (sample_rate is None)
    roots = (baseline, intervention)
    comparisons = compare_conditioning(baseline, intervention, original_results)
    if decoder_checkpoint is not None:
        assert sample_rate is not None and sample_rate > 0
        histories = {
            comparison.sample_id: {
                root.name: torch.stack(
                    [
                        frame["codec_ids"]
                        for frame in torch.load(
                            root / "captures" / comparison.sample_id / "trajectory.pt",
                            weights_only=True,
                            map_location="cpu",
                        )
                    ]
                )
                for root in roots
            }
            for comparison in comparisons
        }
        render_directory = intervention / "common-decoder"
        render_directory.mkdir()
        torch.cuda.set_device(0)
        clips = render(decoder_checkpoint, histories, render_directory, 0, sample_rate)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        for comparison in comparisons:
            comparison.waveform_source = "common_decoder"
            comparison.rendered_clips = [
                clip for clip in clips if clip.sample_id == comparison.sample_id
            ]
        output = intervention / "common-decoder-comparison.json"
    else:
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
            if decoder_checkpoint is None:
                clips = []
                for root, audio_duration_seconds in zip(
                    roots, comparison.audio_duration_seconds, strict=True
                ):
                    waveform = root / "mmsu/audio" / f"{comparison.sample_id}.wav"
                    clips.append(
                        RenderedClip(
                            sample_id=comparison.sample_id,
                            arm=root.name,
                            waveform_path=str(waveform),
                            waveform_sha256=sha256(waveform.read_bytes()).hexdigest(),
                            audio_duration_seconds=audio_duration_seconds,
                        )
                    )
            else:
                clips = comparison.rendered_clips
            for clip in clips:
                assert (
                    sha256(Path(clip.waveform_path).read_bytes()).hexdigest()
                    == clip.waveform_sha256
                )
                comparison.waveform_hashes[clip.arm] = clip.waveform_sha256
                score = transcribe_and_compute_wer(
                    SampleOutput(
                        sample_id=comparison.sample_id,
                        target_text=comparison.original_text,
                        audio_duration_s=clip.audio_duration_seconds,
                    ),
                    clip.waveform_path,
                    asr,
                    "en",
                    "cuda:0",
                )
                assert score.is_success and score.is_transcribed, score.error
                comparison.scores[clip.arm] = score
                output.write_text(
                    json.dumps([asdict(report) for report in comparisons], indent=2)
                )
                print(
                    json.dumps(
                        {
                            "sample_id": comparison.sample_id,
                            "arm": clip.arm,
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
    parser.add_argument("--decoder-checkpoint", type=Path)
    parser.add_argument("--sample-rate", type=int)
    arguments = parser.parse_args()
    main(
        arguments.baseline,
        arguments.intervention,
        arguments.asr_checkpoint,
        arguments.original_results,
        arguments.decoder_checkpoint,
        arguments.sample_rate,
    )
else:
    pass
