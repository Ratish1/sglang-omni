"""Measure expert rounding stages on actual native Thinker inputs and routing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from sgl_kernel import moe_sum_reduce
from sglang.kernels.ops.activation.activation import (
    silu_and_mul,
    silu_and_mul_with_activation_rounding,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextExperts,
)

from scripts.debug_qwen3_omni_feedback_audit import compare_tensors
from scripts.qwen3_omni_checkpoint_weights import CheckpointIndex


def source_rounded_experts(
    experts: Qwen3OmniMoeThinkerTextExperts,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
    *,
    rounding: Literal["source", "activation", "down", "reduction"] = "source",
) -> torch.Tensor:
    hidden, indices, weights = inputs
    assert hidden.dtype == experts.gate_up_proj.dtype == torch.bfloat16
    assert output.shape == hidden.shape and output.dtype == hidden.dtype
    routes = torch.zeros(
        (*indices.shape, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device
    )
    accumulated = torch.zeros_like(hidden) if rounding == "reduction" else None
    for expert_index in indices.unique(sorted=True).tolist():
        token_indices, ranks = torch.where(indices == expert_index)
        gate, up = torch.nn.functional.linear(
            hidden[token_indices], experts.gate_up_proj[expert_index]
        ).chunk(2, dim=-1)
        if rounding == "activation":
            activation = torch.nn.functional.silu(gate) * up
        else:
            activation = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                hidden.dtype
            )
        projected = torch.mm(
            activation, experts.down_proj[expert_index].T, out_dtype=torch.float32
        )
        if rounding == "down":
            projected = projected.to(hidden.dtype).float()
        else:
            pass
        weighted = (projected * weights[token_indices, ranks, None].float()).to(
            hidden.dtype
        )
        if accumulated is not None:
            accumulated.index_add_(0, token_indices, weighted)
        else:
            routes[token_indices, ranks] = weighted
    return (
        accumulated
        if accumulated is not None
        else routes.float().sum(1).to(hidden.dtype)
    )


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delivery-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output_directory.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(arguments.gpu)
    device = torch.device("cuda", arguments.gpu)
    checkpoint_directory = Path(
        snapshot_download(arguments.model_path, revision=arguments.model_revision)
    )
    index = CheckpointIndex.model_validate_json(
        (checkpoint_directory / "model.safetensors.index.json").read_bytes()
    )
    selected = []
    for path in sorted((arguments.delivery_directory / "writes").glob("*.pt")):
        captured = torch.load(path, map_location="cpu", weights_only=True)
        if captured["request_id"] == arguments.request_id:
            selected.append(captured)
        else:
            pass
    assert len(selected) == 1
    captured = selected[0]
    manifest = json.loads(
        (arguments.delivery_directory / "native-thinker-state.json").read_text()
    )
    live = slice(
        captured["batch_offset"], captured["batch_offset"] + captured["extend_length"]
    )
    operators = captured["operator_batches"]
    if arguments.layer_index is not None and arguments.layer_index not in range(
        len(captured["layer_batches"]) - 1
    ):
        raise ValueError("Selected layer is outside the captured expert layers")
    else:
        pass
    reports = []
    for layer_index in range(len(captured["layer_batches"]) - 1):
        if arguments.layer_index is not None and layer_index != arguments.layer_index:
            continue
        else:
            pass
        runtime = manifest["layers"][layer_index]
        assert runtime["activation"] in ("silu", "silu_rounded")
        assert runtime["is_gated"]
        assert not runtime["apply_router_weight_on_input"]
        assert not runtime["no_combine"] and not runtime["w13_swiglu_interleaved"]
        prefix = f"operator_{layer_index:02d}"
        hidden = operators[f"{prefix}_router_input"].to(device)
        expert_indices = operators[f"{prefix}_expert_indices"].long().to(device)
        expert_weights = operators[f"{prefix}_expert_weights"].to(device)
        assert hidden.dtype == torch.bfloat16
        assert expert_weights.dtype == torch.float32
        token_count, top_k = expert_indices.shape
        route_shape = (token_count, top_k, hidden.shape[-1])
        source_routes = torch.zeros(route_shape, dtype=hidden.dtype, device=device)
        rounded_down_only_routes = torch.zeros_like(source_routes)
        rounded_activation_routes = torch.zeros_like(source_routes)
        rounded_down_routes = torch.zeros_like(source_routes)
        checkpoint_output = torch.zeros_like(hidden)
        activation_differences = 0
        activation_values = 0
        down_store_differences = 0
        down_store_values = 0
        native_activation_differences = 0
        native_rounded_activation_differences = 0
        native_activation_maximum_error = 0.0
        native_rounded_activation_maximum_error = 0.0
        for expert_index in expert_indices.unique(sorted=True).tolist():
            token_indices, ranks = torch.where(expert_indices == expert_index)
            weights: dict[str, torch.Tensor] = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                name = f"thinker.model.layers.{layer_index}.mlp.experts.{expert_index}.{projection}.weight"
                with safe_open(
                    checkpoint_directory / index.weight_map[name],
                    framework="pt",
                    device="cpu",
                ) as checkpoint:
                    weights[projection] = checkpoint.get_tensor(name).to(device)
                    assert weights[projection].dtype == hidden.dtype
            gate_up = torch.nn.functional.linear(
                hidden[token_indices],
                torch.cat((weights["gate_proj"], weights["up_proj"])),
            )
            gate, up = gate_up.chunk(2, dim=-1)
            fused_activation = (torch.nn.functional.silu(gate.float()) * up.float()).to(
                hidden.dtype
            )
            rounded_activation = torch.nn.functional.silu(gate) * up
            native_activation = silu_and_mul(gate_up)
            native_rounded_activation = silu_and_mul_with_activation_rounding(gate_up)
            native_activation_differences += int(
                (native_activation != fused_activation).sum()
            )
            native_rounded_activation_differences += int(
                (native_rounded_activation != rounded_activation).sum()
            )
            native_activation_maximum_error = max(
                native_activation_maximum_error,
                float(
                    (native_activation.float() - fused_activation.float()).abs().max()
                ),
            )
            native_rounded_activation_maximum_error = max(
                native_rounded_activation_maximum_error,
                float(
                    (native_rounded_activation.float() - rounded_activation.float())
                    .abs()
                    .max()
                ),
            )
            activation_differences += int(
                (fused_activation != rounded_activation).sum()
            )
            activation_values += fused_activation.numel()
            source_down = torch.mm(
                fused_activation, weights["down_proj"].T, out_dtype=torch.float32
            )
            rounded_activation_down = torch.mm(
                rounded_activation, weights["down_proj"].T, out_dtype=torch.float32
            )
            checkpoint_down = torch.nn.functional.linear(
                rounded_activation, weights["down_proj"]
            )
            down_store_differences += int(
                (rounded_activation_down.to(hidden.dtype) != checkpoint_down).sum()
            )
            down_store_values += checkpoint_down.numel()
            scales = expert_weights[token_indices, ranks, None]
            source_routes[token_indices, ranks] = (source_down * scales).to(
                hidden.dtype
            )
            rounded_down_only_routes[token_indices, ranks] = (
                source_down.to(hidden.dtype).float() * scales
            ).to(hidden.dtype)
            rounded_activation_routes[token_indices, ranks] = (
                rounded_activation_down * scales
            ).to(hidden.dtype)
            rounded_down_routes[token_indices, ranks] = (
                rounded_activation_down.to(hidden.dtype).float() * scales
            ).to(hidden.dtype)
            checkpoint_output.index_add_(
                0,
                token_indices,
                (checkpoint_down * scales).to(hidden.dtype),
            )
        source_output = source_routes.float().sum(1).to(hidden.dtype)
        rounded_down_only_output = (
            rounded_down_only_routes.float().sum(1).to(hidden.dtype)
        )
        native_down_sum = torch.empty_like(hidden)
        moe_sum_reduce(rounded_down_only_routes, native_down_sum, 1.0)
        ordered_down_sum = torch.zeros_like(hidden, dtype=torch.float32)
        for rank in range(top_k):
            ordered_down_sum.add_(rounded_down_only_routes[:, rank].float())
        rounded_activation_output = (
            rounded_activation_routes.float().sum(1).to(hidden.dtype)
        )
        rounded_down_output = rounded_down_routes.float().sum(1).to(hidden.dtype)
        actual = operators[f"{prefix}_moe_output"].to(device)
        differences = {
            "native_to_source_rounding_formula": compare_tensors(
                actual[live], source_output[live]
            ),
            "native_to_rounded_down_only": compare_tensors(
                actual[live], rounded_down_only_output[live]
            ),
            "native_to_rounded_down_native_sum": compare_tensors(
                actual[live], native_down_sum[live]
            ),
            "native_to_rounded_down_rank_ordered_fp32_sum": compare_tensors(
                actual[live], ordered_down_sum[live].to(hidden.dtype)
            ),
            "native_to_rounded_activation": compare_tensors(
                actual[live], rounded_activation_output[live]
            ),
            "native_to_rounded_activation_and_down": compare_tensors(
                actual[live], rounded_down_output[live]
            ),
            "native_to_checkpoint_bf16_accumulation": compare_tensors(
                actual[live], checkpoint_output[live]
            ),
            "source_to_rounded_activation": compare_tensors(
                source_output[live], rounded_activation_output[live]
            ),
            "rounded_activation_to_down_store": compare_tensors(
                rounded_activation_output[live], rounded_down_output[live]
            ),
            "rounded_down_to_checkpoint_accumulation": compare_tensors(
                rounded_down_output[live], checkpoint_output[live]
            ),
        }
        reports.append(
            {
                "layer_index": layer_index,
                "native_activation_contract": runtime["activation"],
                "activation_different_values": activation_differences,
                "activation_values": activation_values,
                "native_activation_different_values": native_activation_differences,
                "native_rounded_activation_different_values": native_rounded_activation_differences,
                "native_activation_maximum_absolute_error": native_activation_maximum_error,
                "native_rounded_activation_maximum_absolute_error": native_rounded_activation_maximum_error,
                "fp32_down_cast_vs_bf16_gemm_different_values": down_store_differences,
                "down_values": down_store_values,
                "down_sum_disagreements": (
                    [
                        {
                            "live_token_index": row,
                            "hidden_column": column,
                            "weighted_bf16_routes": rounded_down_only_routes[live][
                                row, :, column
                            ]
                            .double()
                            .tolist(),
                            "native_output": float(actual[live][row, column]),
                            "torch_sum_output": float(
                                rounded_down_only_output[live][row, column]
                            ),
                            "native_reduction_output": float(
                                native_down_sum[live][row, column]
                            ),
                            "rank_ordered_fp32_sum": float(
                                ordered_down_sum[live][row, column]
                            ),
                            "fp64_sum": float(
                                rounded_down_only_routes[live][row, :, column]
                                .double()
                                .sum()
                            ),
                        }
                        for row, column in (
                            actual[live] != rounded_down_only_output[live]
                        )
                        .nonzero()
                        .tolist()
                    ]
                    if differences["native_to_rounded_down_native_sum"].exact
                    else []
                ),
                "comparisons": {
                    name: asdict(difference) for name, difference in differences.items()
                },
            }
        )
        print(json.dumps({"completed_layer": layer_index}), flush=True)
    report = {
        "request_id": arguments.request_id,
        "down_sum_scope": "Native C++ and explicit rank-ordered FP32 reductions consume the same independently calculated BF16 routes. Disagreement coordinates are retained only when native reduction matches the entire live output. The parallel torch.sum comparison is preserved, without a widened tolerance.",
        "scope": "Held native inputs and routing with raw checkpoint weights. These source-semantic counterfactuals use per-expert cuBLAS GEMMs, not the native padded Triton GEMM geometry or its fast SiLU intrinsics. The native activation measurements separately execute the actual ordinary and activation-rounded SiLU kernels on the same projected operands, reporting differences rather than assuming exactness. Changing a rounding stage is not a production repair or a speech-quality result. The FP32-down-cast versus BF16-GEMM comparison explicitly reports any additional matmul difference.",
        "layers": reports,
    }
    (arguments.output_directory / "comparisons.json").write_text(
        json.dumps(report, indent=2)
    )


if __name__ == "__main__":
    main()
