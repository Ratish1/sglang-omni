"""Replay recorded Talker operations on their actual native inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import Qwen3OmniMoeConfig
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerForConditionalGeneration,
)


@dataclass(kw_only=True)
class LayerOperationReport:
    request_id: str
    frame: int
    layer: int
    selected_experts: list[int]
    selected_experts_are_topk: bool
    comparisons: dict[str, dict[str, float | bool]]


@dataclass(kw_only=True)
class BoundaryReport:
    reference_file: str
    layer: int
    expert_set_equal: bool
    comparisons: dict[str, dict[str, float | bool]]


def compare_tensors(
    actual: torch.Tensor, expected: torch.Tensor
) -> dict[str, float | bool]:
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    difference = actual.float() - expected.float()
    expected_norm = expected.float().norm()
    assert expected_norm > 0
    return {
        "exact": torch.equal(actual, expected),
        "equal_fraction": float((actual == expected).float().mean()),
        "maximum_absolute_error": float(difference.abs().max()),
        "relative_l2_error": float(difference.norm() / expected_norm),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--capture-directory", type=Path, required=True)
    parser.add_argument("--reference-directory", action="append", type=Path, default=[])
    parser.add_argument("--request-id", action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    assert not arguments.output_file.exists()
    torch.set_float32_matmul_precision("highest")
    config = Qwen3OmniMoeConfig.from_pretrained(arguments.model_path).talker_config
    register_checkpoint_conversion_mapping(
        config.model_type, get_checkpoint_conversion_mapping("qwen3_omni_moe")
    )
    reference, loading = Qwen3OmniMoeTalkerForConditionalGeneration.from_pretrained(
        arguments.model_path,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        experts_implementation="eager",
        use_kernels=False,
        output_loading_info=True,
    )
    assert not loading["missing_keys"], loading["missing_keys"]
    assert not loading["mismatched_keys"], loading["mismatched_keys"]
    reference.eval()
    operations: list[LayerOperationReport] = []
    boundaries: list[BoundaryReport] = []
    with torch.inference_mode():
        for capture_file in sorted(arguments.capture_directory.glob("request-*.pt")):
            capture = torch.load(capture_file, map_location="cpu", weights_only=True)
            request_id = capture["request_id"]
            if request_id not in arguments.request_id:
                continue
            else:
                pass
            for frame, snapshot in enumerate(capture["snapshots"]):
                reference_snapshots = {
                    str(filename): torch.load(
                        filename, map_location=reference.device, weights_only=True
                    )["reference_boundaries"]
                    for directory in arguments.reference_directory
                    for filename in directory.glob(f"{request_id}.*.{frame}.pt")
                }
                for layer_index, layer in enumerate(reference.model.layers):
                    native_attention = {
                        name.rsplit(".", 1)[-1].removeprefix("diagnostic_"): tensor.to(
                            reference.device
                        )
                        for name, tensor in snapshot.items()
                        if f".layers.{layer_index}.self_attn.attn.diagnostic_" in name
                    }
                    native_moe = {
                        name.rsplit(".", 1)[-1].removeprefix(
                            "diagnostic_moe_"
                        ): tensor.to(reference.device)
                        for name, tensor in snapshot.items()
                        if f".layers.{layer_index}.mlp.diagnostic_moe_" in name
                    }
                    attention_input = native_attention["attention_input"][None]
                    hidden_states = native_moe["input"][None]
                    expert_indices = native_moe["experts"][None].long()
                    assert expert_indices.unique().numel() == expert_indices.numel()
                    routing_weights = native_moe["weights"][None]
                    router_logits = native_moe["router_logits"][None]
                    assert torch.equal(
                        native_moe["routed_output"] + native_moe["shared_output"],
                        native_moe["output"],
                    )
                    selected_logits = router_logits.gather(1, expert_indices)
                    cutoff = router_logits.topk(expert_indices.shape[1]).values[:, -1:]
                    assert (selected_logits >= cutoff).all()
                    expected_weights = router_logits.softmax(dim=-1).gather(
                        1, expert_indices
                    )
                    if config.text_config.norm_topk_prob:
                        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
                    else:
                        pass
                    query_key_value_weights = torch.cat(
                        [
                            projection.weight
                            for projection in (
                                layer.self_attn.q_proj,
                                layer.self_attn.k_proj,
                                layer.self_attn.v_proj,
                            )
                        ]
                    )
                    assert not config.text_config.attention_bias
                    query = native_attention["query"].float()
                    keys = snapshot[f"layer_{layer_index}_keys"].to(reference.device)
                    values = snapshot[f"layer_{layer_index}_values"].to(
                        reference.device
                    )
                    repeats = query.shape[0] // keys.shape[1]
                    assert query.shape[0] % keys.shape[1] == 0
                    scaling, logit_cap, window = native_attention[
                        "attention_properties"
                    ].tolist()
                    assert logit_cap == 0 and window == -1
                    assert int(native_attention["position"]) + 1 == len(keys)
                    scores = (
                        torch.einsum(
                            "hd,shd->hs",
                            query,
                            keys.float().repeat_interleave(repeats, 1),
                        )
                        * scaling
                    )
                    expected_attention = torch.einsum(
                        "hs,shd->hd",
                        scores.softmax(dim=-1),
                        values.float().repeat_interleave(repeats, 1),
                    )
                    reference_routed = layer.mlp.experts(
                        hidden_states, expert_indices, routing_weights
                    )
                    float_routed = torch.zeros_like(hidden_states, dtype=torch.float32)
                    for expert_index in expert_indices[0].tolist():
                        selected_column = (expert_indices[0] == expert_index).nonzero()[
                            0, 0
                        ]
                        gate, up = F.linear(
                            hidden_states.float(),
                            layer.mlp.experts.gate_up_proj[expert_index].float(),
                        ).chunk(2, dim=-1)
                        expert_output = F.linear(
                            F.silu(gate) * up,
                            layer.mlp.experts.down_proj[expert_index].float(),
                        )
                        float_routed.add_(
                            expert_output * routing_weights[0, selected_column]
                        )
                    float_shared = (
                        F.linear(
                            F.silu(
                                F.linear(
                                    hidden_states.float(),
                                    layer.mlp.shared_expert.gate_proj.weight.float(),
                                )
                            )
                            * F.linear(
                                hidden_states.float(),
                                layer.mlp.shared_expert.up_proj.weight.float(),
                            ),
                            layer.mlp.shared_expert.down_proj.weight.float(),
                        )
                        * F.linear(
                            hidden_states.float(),
                            layer.mlp.shared_expert_gate.weight.float(),
                        ).sigmoid()
                    )
                    reference_shared = (
                        layer.mlp.shared_expert(hidden_states)
                        * layer.mlp.shared_expert_gate(hidden_states).sigmoid()
                    )
                    comparisons = {
                        "qkv_vs_bfloat16": compare_tensors(
                            native_attention["qkv_projection"][None],
                            F.linear(attention_input, query_key_value_weights),
                        ),
                        "qkv_vs_float32": compare_tensors(
                            native_attention["qkv_projection"][None],
                            F.linear(
                                attention_input.float(), query_key_value_weights.float()
                            ),
                        ),
                        "attention_vs_float32": compare_tensors(
                            native_attention["attention_output"], expected_attention
                        ),
                        "attention_projection_vs_bfloat16": compare_tensors(
                            native_attention["attention_projection"][None],
                            layer.self_attn.o_proj(
                                native_attention["attention_output"].reshape(1, -1)
                            ),
                        ),
                        "router_vs_float32": compare_tensors(
                            router_logits,
                            F.linear(
                                hidden_states.float(), layer.mlp.gate.weight.float()
                            ),
                        ),
                        "routing_weights_vs_softmax": compare_tensors(
                            routing_weights, expected_weights
                        ),
                        "routed_vs_bfloat16": compare_tensors(
                            native_moe["routed_output"][None], reference_routed
                        ),
                        "routed_vs_float32": compare_tensors(
                            native_moe["routed_output"][None], float_routed
                        ),
                        "reference_routed_vs_float32": compare_tensors(
                            reference_routed, float_routed
                        ),
                        "shared_vs_bfloat16": compare_tensors(
                            native_moe["shared_output"][None], reference_shared
                        ),
                        "shared_vs_float32": compare_tensors(
                            native_moe["shared_output"][None], float_shared
                        ),
                        "reference_shared_vs_float32": compare_tensors(
                            reference_shared, float_shared
                        ),
                    }
                    report = LayerOperationReport(
                        request_id=request_id,
                        frame=frame,
                        layer=layer_index,
                        selected_experts=expert_indices[0].tolist(),
                        selected_experts_are_topk=True,
                        comparisons=comparisons,
                    )
                    operations.append(report)
                    for (
                        reference_file,
                        boundary_snapshot,
                    ) in reference_snapshots.items():
                        prefix = f"layer_{layer_index}_"
                        boundary_pairs = {
                            "attention_input": native_attention["attention_input"],
                            "attention_projection": native_attention[
                                "attention_projection"
                            ],
                            "moe_input": native_moe["input"],
                            "moe_output": native_moe["output"],
                            "router_logits": native_moe["router_logits"],
                        }
                        boundary_comparisons = {
                            name: compare_tensors(
                                tensor, boundary_snapshot[f"{prefix}{name}"]
                            )
                            for name, tensor in boundary_pairs.items()
                        }
                        native_weights_by_expert = torch.zeros_like(router_logits[0])
                        reference_weights_by_expert = torch.zeros_like(router_logits[0])
                        native_weights_by_expert.scatter_(
                            0, expert_indices[0], native_moe["weights"]
                        )
                        reference_weights_by_expert.scatter_(
                            0,
                            boundary_snapshot[f"{prefix}router_experts"].long(),
                            boundary_snapshot[f"{prefix}router_weights"].float(),
                        )
                        boundary_comparisons["router_weights_by_expert"] = (
                            compare_tensors(
                                native_weights_by_expert, reference_weights_by_expert
                            )
                        )
                        boundaries.append(
                            BoundaryReport(
                                reference_file=reference_file,
                                layer=layer_index,
                                expert_set_equal=torch.equal(
                                    native_moe["experts"].sort().values,
                                    boundary_snapshot[f"{prefix}router_experts"]
                                    .sort()
                                    .values,
                                ),
                                comparisons=boundary_comparisons,
                            )
                        )
                arguments.output_file.write_text(
                    json.dumps(
                        {
                            "operations": [asdict(report) for report in operations],
                            "boundaries": [asdict(report) for report in boundaries],
                        },
                        indent=2,
                    )
                )
                print(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "frame": frame,
                            "layers": len(reference.model.layers),
                        }
                    ),
                    flush=True,
                )
    assert {report.request_id for report in operations} == set(arguments.request_id)


if __name__ == "__main__":
    main()
