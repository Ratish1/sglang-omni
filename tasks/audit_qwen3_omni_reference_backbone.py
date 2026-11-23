# SPDX-License-Identifier: Apache-2.0
"""Compare each native decode step with checkpoint computation on its held state."""

import argparse
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import validate
from qwen3_omni_checkpoint_weights import load_talker
from replay_qwen3_omni_reference_continuation import SamplingPolicy
from transformers.cache_utils import DynamicCache, DynamicLayer


@dataclass(kw_only=True)
class TensorDifference:
    changed_values: int
    maximum_absolute: float
    relative_l2: float | None


def difference(native: torch.Tensor, reference: torch.Tensor) -> TensorDifference:
    assert native.shape == reference.shape and native.dtype == reference.dtype
    assert torch.isfinite(native).all() and torch.isfinite(reference).all()
    delta = reference.double() - native.double()
    magnitude = float(torch.linalg.vector_norm(native.double()))
    return TensorDifference(
        changed_values=int((native != reference).count_nonzero()),
        maximum_absolute=float(delta.abs().max()),
        relative_l2=(
            float(torch.linalg.vector_norm(delta)) / magnitude
            if magnitude != 0
            else None
        ),
    )


@dataclass(kw_only=True)
class FrameComparison:
    frame_index: int
    sequence_tokens: int
    native_code: int
    reference_code: int
    native_head: TensorDifference
    final_query: TensorDifference
    logits: TensorDifference
    keys_by_layer: list[TensorDifference]
    values_by_layer: list[TensorDifference]
    reference_tensors_sha256: str


@torch.inference_mode()
def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    torch.cuda.set_device(arguments.gpu)
    model = load_talker(arguments.checkpoint, arguments.gpu, arguments.output)
    policy = SamplingPolicy(
        seed=arguments.seed,
        temperature=arguments.temperature,
        repetition_penalty=arguments.repetition_penalty,
        top_k=arguments.top_k,
        top_p=arguments.top_p,
        vocabulary_tokens=model.config.text_config.vocab_size,
        eos_token=model.config.codec_eos_token_id,
        device=model.device,
    )
    for directory in arguments.capture:
        frames = validate(directory)
        native_codes = torch.stack([frame["codec_ids"][0] for frame in frames]).to(
            model.device
        )
        assert int(native_codes[-1]) == policy.eos_token
        prefill_tokens = frames[0]["stored_keys"].shape[1]
        native_keys = torch.cat([frame["stored_keys"] for frame in frames], dim=1)
        native_values = torch.cat([frame["stored_values"] for frame in frames], dim=1)
        assert native_keys.shape == native_values.shape
        assert native_keys.shape[0] == len(model.model.layers)
        assert native_keys.shape[1] == prefill_tokens + len(frames) - 1
        native_keys = native_keys.permute(0, 2, 1, 3).to(model.device)
        native_values = native_values.permute(0, 2, 1, 3).to(model.device)
        comparisons: dict[int, FrameComparison] = {}
        for repeat in range(2):
            frame_indices = (
                range(1, len(frames)) if repeat == 0 else range(len(frames) - 1, 0, -1)
            )
            for frame_index in frame_indices:
                frame = frames[frame_index]
                position = prefill_tokens + frame_index - 1
                assert int(frame["positions"].item()) == position
                assert torch.equal(
                    frame["rotary_positions"], frame["positions"].expand(3, -1)
                )
                cache = DynamicCache(
                    ddp_cache_data=[
                        (keys[None, :, :position], values[None, :, :position])
                        for keys, values in zip(native_keys, native_values, strict=True)
                    ],
                    config=model.config.text_config,
                )
                assert cache.get_seq_length() == position
                embedding = frame["consumed_input"].to(model.device).reshape(1, 1, -1)
                output = model.model(
                    inputs_embeds=embedding,
                    past_key_values=cache,
                    position_ids=torch.full(
                        (4, 1, 1), position, dtype=torch.long, device=model.device
                    ),
                    use_cache=True,
                )
                assert output.past_key_values is cache
                assert cache.get_seq_length() == position + 1
                query = output.last_hidden_state.reshape(1, -1)
                logits = model.codec_head(query).float()
                native_query = frame["predictor_query"].to(model.device).reshape(1, -1)
                projected_native = model.codec_head(native_query).float().cpu()[0]
                repeated = torch.zeros(
                    (1, policy.vocabulary_tokens), dtype=torch.bool, device=model.device
                )
                repeated[:, native_codes[:frame_index]] = True
                native_logits = frame["logits"].reshape(-1)
                native_choice = policy.sample(
                    policy.probabilities(
                        native_logits[None].to(model.device), repeated
                    ),
                    position,
                )
                assert int(native_choice.item()) == int(native_codes[frame_index])
                reference_choice = policy.sample(
                    policy.probabilities(logits, repeated), position
                )
                current_keys: list[torch.Tensor] = []
                current_values: list[torch.Tensor] = []
                for layer_index, layer in enumerate(cache.layers):
                    assert type(layer) is DynamicLayer
                    assert torch.equal(
                        layer.keys[0, :, :-1], native_keys[layer_index, :, :position]
                    )
                    assert torch.equal(
                        layer.values[0, :, :-1],
                        native_values[layer_index, :, :position],
                    )
                    current_keys.append(layer.keys[0, :, -1].cpu())
                    current_values.append(layer.values[0, :, -1].cpu())
                query = query[0].cpu()
                logits = logits[0].cpu()
                tensor_hash = sha256()
                for tensor in (query, logits, *current_keys, *current_values):
                    tensor_hash.update(
                        tensor.contiguous().view(torch.uint8).numpy().tobytes()
                    )
                comparison = FrameComparison(
                    frame_index=frame_index,
                    sequence_tokens=position + 1,
                    native_code=int(native_codes[frame_index]),
                    reference_code=int(reference_choice.item()),
                    native_head=difference(native_logits, projected_native),
                    final_query=difference(frame["predictor_query"].reshape(-1), query),
                    logits=difference(native_logits, logits),
                    keys_by_layer=[
                        difference(native[:, 0], reference)
                        for native, reference in zip(
                            frame["stored_keys"].transpose(1, 2),
                            current_keys,
                            strict=True,
                        )
                    ],
                    values_by_layer=[
                        difference(native[:, 0], reference)
                        for native, reference in zip(
                            frame["stored_values"].transpose(1, 2),
                            current_values,
                            strict=True,
                        )
                    ],
                    reference_tensors_sha256=tensor_hash.hexdigest(),
                )
                if repeat == 0:
                    comparisons[frame_index] = comparison
                else:
                    assert comparison == comparisons[frame_index], (
                        directory.name,
                        frame_index,
                    )
            ordered = [comparisons[index] for index in range(1, len(frames))]
            report = {
                "sample_id": directory.name,
                "repeat": repeat,
                "capture_sha256": sha256(
                    (directory / "trajectory.pt").read_bytes()
                ).hexdigest(),
                "compared_decode_frames": len(ordered),
                "changed_main_codes": sum(
                    row.native_code != row.reference_code for row in ordered
                ),
                "changed_native_query_head_projections": sum(
                    row.native_head.changed_values != 0 for row in ordered
                ),
                "reverse_order_reproduces_all_output_tensors": repeat == 1,
                "native_prior_cache_unchanged": True,
                "sampling_policy": {**asdict(policy), "device": str(policy.device)},
                "frames": [asdict(row) for row in ordered],
            }
            (arguments.output / f"{directory.name}.json").write_text(
                json.dumps(report, indent=2)
            )
            print(
                json.dumps(
                    {name: value for name, value in report.items() if name != "frames"}
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--capture", type=Path, action="append", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    main(parser.parse_args())
else:
    pass
