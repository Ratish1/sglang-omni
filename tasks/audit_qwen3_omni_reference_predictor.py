# SPDX-License-Identifier: Apache-2.0
"""Compare residual codes at every captured native query without changing Talker state."""

import argparse
import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import validate
from qwen3_omni_checkpoint_weights import load_talker
from torch import nn


@dataclass(kw_only=True)
class FrameComparison:
    frame_index: int
    changed_codebooks: list[int]
    head_logits_sha256: str
    native_code_at_first_difference: int | None
    reference_code_at_first_difference: int | None
    native_code_logit: float | None
    reference_code_logit: float | None
    reference_top_two_margin: float | None


@torch.inference_mode()
def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    torch.cuda.set_device(arguments.gpu)
    model = load_talker(arguments.checkpoint, arguments.gpu, arguments.output)
    predictor = model.code_predictor
    (arguments.output / "generation-config.json").write_text(
        predictor.generation_config.to_json_string()
    )
    comparisons: dict[str, dict[int, FrameComparison]] = {}
    predictions: dict[str, torch.Tensor] = {}
    source_hashes: dict[str, str] = {}
    head_logits: list[torch.Tensor] = []

    def capture_head(
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        outputs: torch.Tensor,
    ) -> None:
        assert module is predictor.lm_head[len(head_logits)]
        assert inputs[0].dtype == outputs.dtype == torch.bfloat16
        head_logits.append(outputs[:, -1].float().clone())

    with ExitStack() as hooks:
        for head in predictor.lm_head:
            hooks.callback(head.register_forward_hook(capture_head).remove)
        for repeat in range(2):
            directories = arguments.capture if repeat == 0 else arguments.capture[::-1]
            for directory in directories:
                frames = validate(directory)
                assert (
                    int(frames[-1]["codec_ids"][0]) == model.config.codec_eos_token_id
                )
                native_codes = torch.stack(
                    [frame["codec_ids"] for frame in frames[:-1]]
                )
                frame_count = len(native_codes)
                if repeat == 0:
                    comparisons[directory.name] = {}
                    predictions[directory.name] = torch.empty_like(native_codes)
                    source_hashes[directory.name] = sha256(
                        (directory / "trajectory.pt").read_bytes()
                    ).hexdigest()
                    frame_indices = range(frame_count)
                else:
                    frame_indices = range(frame_count - 1, -1, -1)
                for frame_index in frame_indices:
                    query = (
                        frames[frame_index]["predictor_query"]
                        .to(model.device)
                        .reshape(1, 1, -1)
                    )
                    main_code = (
                        native_codes[frame_index, :1].to(model.device).reshape(1, 1)
                    )
                    main_embedding = model.get_input_embeddings()(main_code)
                    assert query.dtype == main_embedding.dtype == torch.bfloat16
                    head_logits.clear()
                    result = predictor.generate(
                        inputs_embeds=torch.cat((query, main_embedding), dim=1),
                        max_new_tokens=model.config.num_code_groups - 1,
                        do_sample=False,
                        output_logits=True,
                        return_dict_in_generate=True,
                    )
                    assert result.sequences.shape == (
                        1,
                        model.config.num_code_groups - 1,
                    )
                    actual_logits = torch.stack(head_logits)[:, 0]
                    assert torch.isfinite(actual_logits).all()
                    assert torch.equal(actual_logits, torch.stack(result.logits)[:, 0])
                    assert torch.equal(actual_logits.argmax(-1), result.sequences[0])
                    codes = torch.cat((main_code, result.sequences), dim=-1)[0].cpu()
                    changed = (
                        (codes != native_codes[frame_index])
                        .nonzero()
                        .flatten()
                        .tolist()
                    )
                    if changed:
                        first_codebook = changed[0]
                        assert first_codebook > 0
                        native_code = int(native_codes[frame_index, first_codebook])
                        reference_code = int(codes[first_codebook])
                        shared_history_logits = actual_logits[first_codebook - 1]
                        native_logit = float(shared_history_logits[native_code])
                        reference_logit = float(shared_history_logits[reference_code])
                        margin = float(
                            shared_history_logits.topk(2).values.diff().abs()[0]
                        )
                    else:
                        native_code = reference_code = None
                        native_logit = reference_logit = margin = None
                    comparison = FrameComparison(
                        frame_index=frame_index,
                        changed_codebooks=changed,
                        head_logits_sha256=sha256(
                            actual_logits.cpu().numpy().tobytes()
                        ).hexdigest(),
                        native_code_at_first_difference=native_code,
                        reference_code_at_first_difference=reference_code,
                        native_code_logit=native_logit,
                        reference_code_logit=reference_logit,
                        reference_top_two_margin=margin,
                    )
                    if repeat == 0:
                        comparisons[directory.name][frame_index] = comparison
                        predictions[directory.name][frame_index].copy_(codes)
                    else:
                        assert comparison == comparisons[directory.name][frame_index], (
                            directory.name,
                            frame_index,
                        )
                        assert torch.equal(
                            predictions[directory.name][frame_index], codes
                        )
                ordered = [
                    comparisons[directory.name][index] for index in range(frame_count)
                ]
                report = {
                    "sample_id": directory.name,
                    "repeat": repeat,
                    "capture_sha256": source_hashes[directory.name],
                    "compared_frames": frame_count,
                    "identical_codec_frames": sum(
                        not row.changed_codebooks for row in ordered
                    ),
                    "changed_codec_frames": sum(
                        bool(row.changed_codebooks) for row in ordered
                    ),
                    "reversed_order_reproduces_all_head_logits": (
                        True if repeat == 1 else None
                    ),
                    "frames": [asdict(row) for row in ordered],
                }
                (arguments.output / f"{directory.name}.json").write_text(
                    json.dumps(report, indent=2)
                )
                torch.save(
                    predictions[directory.name],
                    arguments.output / f"{directory.name}.pt",
                )
                print(
                    json.dumps(
                        {
                            name: value
                            for name, value in report.items()
                            if name != "frames"
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    main(parser.parse_args())
else:
    pass
