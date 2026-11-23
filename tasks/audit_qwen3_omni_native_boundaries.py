# SPDX-License-Identifier: Apache-2.0
"""Replay captured normalization and projection boundaries with native modules."""

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from hashlib import sha256
from os import environ
from pathlib import Path

import torch
from audit_qwen3_omni_native_predictor import load_native_talker
from compare_qwen3_omni_native_trajectories import validate
from safetensors import safe_open

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.vendor.sglang.layers import MRotaryEmbedding
from sglang_omni.vendor.sglang.models import apply_qk_norm


@dataclass(kw_only=True)
class BoundaryComparison:
    frame_index: int
    layer_index: int
    boundary: str
    compared_values: int
    changed_values: int
    maximum_absolute_difference: float


def verify_weights(model: Qwen3OmniTalker, checkpoint: Path) -> list[str]:
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    parameters: dict[str, torch.Tensor] = {
        "talker.model.norm.weight": model.model.norm.weight
    }
    for layer_index, layer in enumerate(model.model.layers):
        assert layer.self_attn.qkv_proj.bias is None
        assert layer.self_attn.o_proj.bias is None
        prefix = f"talker.model.layers.{layer_index}."
        for name, parameter in (
            ("input_layernorm.weight", layer.input_layernorm.weight),
            ("post_attention_layernorm.weight", layer.post_attention_layernorm.weight),
            ("self_attn.q_norm.weight", layer.self_attn.q_norm.weight),
            ("self_attn.k_norm.weight", layer.self_attn.k_norm.weight),
            ("self_attn.o_proj.weight", layer.self_attn.o_proj.weight),
        ):
            parameters[prefix + name] = parameter
        widths = (
            layer.self_attn.q_size,
            layer.self_attn.kv_size,
            layer.self_attn.kv_size,
        )
        for name, parameter in zip(
            ("q", "k", "v"), layer.self_attn.qkv_proj.weight.split(widths), strict=True
        ):
            parameters[f"{prefix}self_attn.{name}_proj.weight"] = parameter
    with ExitStack() as stack:
        shards = {
            shard: stack.enter_context(
                safe_open(checkpoint / shard, framework="pt", device="cpu")
            )
            for shard in sorted({weight_map[name] for name in parameters})
        }
        for name, parameter in parameters.items():
            expected = shards[weight_map[name]].get_tensor(name)
            assert expected.dtype == parameter.dtype == torch.bfloat16, name
            assert torch.equal(parameter.cpu(), expected), name
    return sorted(parameters)


@torch.inference_mode()
def main(arguments: argparse.Namespace) -> None:
    arguments.output.mkdir()
    captures = sorted(arguments.root.glob("captures/*/trajectory.pt"))
    assert captures
    environ["OMNI_NATIVE_TRAJECTORY_DIRECTORY"] = str(arguments.output)
    environ["OMNI_NATIVE_TRAJECTORY_SAMPLES"] = ",".join(
        capture.parent.name for capture in captures
    )
    torch.set_num_threads(1)
    torch.cuda.set_device(arguments.gpu)
    scheduler = load_native_talker(
        arguments.checkpoint, arguments.config, arguments.gpu, arguments.output
    )
    model = scheduler.model_runner.model
    assert isinstance(model, Qwen3OmniTalker)
    verified_weights = verify_weights(model, arguments.checkpoint)
    reports = {}
    for capture in captures:
        frames = validate(capture.parent)
        steps = json.loads(capture.with_suffix(".json").read_text())["frames"]
        comparisons: list[BoundaryComparison] = []
        for frame_index, (frame, step) in enumerate(zip(frames, steps, strict=True)):
            if "captured_layers" not in step:
                continue
            else:
                assert step["captured_layers"] == len(model.model.layers)
            live_rows = len(step["request_ids"])
            for layer_index, layer in enumerate(model.model.layers):
                prefix = f"layer{layer_index}_peer_"
                operands = {
                    name.removeprefix(prefix): values.cuda()
                    for name, values in frame.items()
                    if name.startswith(prefix)
                }
                contribution = operands["contribution"].clone()
                if layer_index == 0:
                    normalized = layer.input_layernorm(contribution)
                    residual = contribution
                else:
                    normalized, residual = layer.input_layernorm(
                        contribution, operands["previous_residual"].clone()
                    )
                actual = {
                    "attention_normalized": normalized,
                    "attention_residual": residual,
                }
                attention = layer.self_attn
                assert not attention.use_fused_qk_norm_rope
                actual["qkv_projection"], _ = attention.qkv_proj(
                    operands["attention_normalized"]
                )
                query, key, value = (
                    operands["qkv_projection"]
                    .clone()
                    .split(
                        (attention.q_size, attention.kv_size, attention.kv_size), dim=-1
                    )
                )
                query, key = apply_qk_norm(
                    query, key, attention.q_norm, attention.k_norm, attention.head_dim
                )
                actual["query_before_rope"] = query.clone()
                actual["key_before_rope"] = key.clone()
                actual["value_before_cache"] = value.clone()
                positions = operands["positions"].movedim(0, -1)
                if positions.dim() == 2 and not isinstance(
                    attention.rotary_emb, MRotaryEmbedding
                ):
                    positions = positions[0]
                else:
                    pass
                query, key = attention.rotary_emb(
                    positions,
                    operands["query_before_rope"].clone(),
                    operands["key_before_rope"].clone(),
                )
                actual["query_after_rope"] = query
                actual["key_after_rope"] = key
                actual["attention_projection"], _ = attention.o_proj(
                    operands["attention_context"]
                )
                normalized, residual = layer.post_attention_layernorm(
                    operands["attention_projection"].clone(),
                    operands["attention_residual"].clone(),
                )
                actual["mlp_normalized"] = normalized
                actual["mlp_residual"] = residual
                expected = operands
                if layer_index == len(model.model.layers) - 1:
                    final_query, _ = model.model.norm(
                        operands["layer_output"].clone(),
                        operands["mlp_residual"].clone(),
                    )
                    actual["predictor_query"] = final_query[
                        step["request_row"] : step["request_row"] + 1
                    ]
                    expected["predictor_query"] = (
                        frame["predictor_query"].cuda().reshape(1, -1)
                    )
                else:
                    pass
                for boundary, tensor in actual.items():
                    reference = expected[boundary][:live_rows]
                    tensor = tensor[:live_rows]
                    assert (
                        tensor.shape == reference.shape
                        and tensor.dtype == reference.dtype
                    )
                    assert (
                        torch.isfinite(tensor).all() and torch.isfinite(reference).all()
                    )
                    comparisons.append(
                        BoundaryComparison(
                            frame_index=frame_index,
                            layer_index=layer_index,
                            boundary=boundary,
                            compared_values=tensor.numel(),
                            changed_values=int((tensor != reference).count_nonzero()),
                            maximum_absolute_difference=float(
                                (tensor.float() - reference.float()).abs().max()
                            ),
                        )
                    )
        assert comparisons
        mismatches = [
            asdict(comparison)
            for comparison in comparisons
            if comparison.changed_values
        ]
        reports[capture.parent.name] = {
            "capture_sha256": sha256(capture.read_bytes()).hexdigest(),
            "checked_boundaries": len(comparisons),
            "checks_by_boundary": dict(
                Counter(comparison.boundary for comparison in comparisons)
            ),
            "mismatches": mismatches,
        }
        print(json.dumps(reports[capture.parent.name]), flush=True)
    with (arguments.output / "boundaries.json").open("x") as receipt:
        json.dump(
            {
                "scope": "Native modules on recorded whole padded peer inputs, compared only on live rows at captured deep boundaries. Each operator receives its captured input independently; this is not a full free-running forward or a speech-quality comparison.",
                "verified_checkpoint_weights": verified_weights,
                "captures": reports,
            },
            receipt,
            indent=2,
        )
    torch.distributed.destroy_process_group()
    assert all(not report["mismatches"] for report in reports.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", type=int, required=True)
    main(parser.parse_args())
else:
    pass
