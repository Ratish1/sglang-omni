"""Separate arithmetic families in checkpoint conditioning."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import torch
from transformers import DynamicCache
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextExperts,
    Qwen3OmniMoeThinkerTextTopKRouter,
)

from scripts.debug_qwen3_omni_thinker_rotary import contracted_bf16_rotation
from scripts.qwen3_omni_native_residual_control import NativeResidualControl
from scripts.qwen3_omni_native_rotary_control import NativeRotaryControl


@dataclass(kw_only=True)
class ThinkerPrecisionInputs:
    embedded_input: torch.Tensor
    positions: torch.Tensor
    prefix_token_count: int
    accepted_layer: int
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    operators: dict[str, torch.Tensor]
    native_accepted_hidden: torch.Tensor
    batch_offset: int
    extend_token_count: int


def keep_fp32_route_weights(
    router: Qwen3OmniMoeThinkerTextTopKRouter,
    inputs: tuple[torch.Tensor],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, weights, indices = output
    assert inputs[0].shape[-1] == router.hidden_dim
    assert router.norm_topk_prob
    assert indices.shape[-1] == router.top_k
    probabilities = logits.float().softmax(-1)
    selected = probabilities.gather(1, indices)
    selected /= selected.sum(-1, keepdim=True)
    assert torch.equal(selected.to(weights.dtype), weights)
    return logits, selected, indices


def source_rounded_experts(
    experts: Qwen3OmniMoeThinkerTextExperts,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    output: torch.Tensor,
) -> torch.Tensor:
    hidden, indices, weights = inputs
    assert hidden.dtype == experts.gate_up_proj.dtype == torch.bfloat16
    assert output.shape == hidden.shape and output.dtype == hidden.dtype
    routes = torch.zeros(
        (*indices.shape, hidden.shape[-1]), dtype=hidden.dtype, device=hidden.device
    )
    for expert_index in indices.unique(sorted=True).tolist():
        token_indices, ranks = torch.where(indices == expert_index)
        gate, up = torch.nn.functional.linear(
            hidden[token_indices], experts.gate_up_proj[expert_index]
        ).chunk(2, dim=-1)
        activation = (torch.nn.functional.silu(gate.float()) * up.float()).to(
            hidden.dtype
        )
        projected = torch.mm(
            activation, experts.down_proj[expert_index].T, out_dtype=torch.float32
        )
        routes[token_indices, ranks] = (
            projected * weights[token_indices, ranks, None].float()
        ).to(hidden.dtype)
    return routes.float().sum(1).to(hidden.dtype)


@torch.inference_mode()
def run_precision_controls(
    reference: Qwen3OmniMoeThinkerForConditionalGeneration,
    captured: ThinkerPrecisionInputs,
    output_directory: Path,
    variants: tuple[
        Literal[
            "checkpoint",
            "router",
            "experts",
            "router-and-experts",
            "rotary",
            "residual-norm",
            "residual-norm-and-rotary",
            "source-arithmetic",
            "residual-rotary-qknorm",
            "residual-rotary-moe",
            "residual-rotary-qknorm-router",
            "residual-rotary-qknorm-experts",
        ],
        ...,
    ] = ("checkpoint", "router", "experts", "router-and-experts"),
) -> None:
    prefix_token_count = captured.prefix_token_count
    device = reference.device
    embedded = captured.embedded_input.to(device)
    positions = captured.positions.to(device)
    suffix = embedded[None, prefix_token_count:]
    query_positions = prefix_token_count + torch.arange(suffix.shape[1], device=device)
    allowed = (
        torch.arange(len(embedded), device=device)[None] <= query_positions[:, None]
    )
    attention_mask = torch.zeros(
        (1, 1, suffix.shape[1], len(embedded)), dtype=embedded.dtype, device=device
    ).masked_fill_(~allowed[None, None], float("-inf"))
    position_embeddings = reference.model.rotary_emb(
        suffix, positions[:, None, prefix_token_count:]
    )
    live = slice(
        captured.batch_offset, captured.batch_offset + captured.extend_token_count
    )
    rotary_table: torch.Tensor | None = None
    rotary_variants = (
        "rotary",
        "residual-norm-and-rotary",
        "source-arithmetic",
        "residual-rotary-qknorm",
        "residual-rotary-moe",
        "residual-rotary-qknorm-router",
        "residual-rotary-qknorm-experts",
    )
    expert_variants = (
        "experts",
        "router-and-experts",
        "source-arithmetic",
        "residual-rotary-moe",
        "residual-rotary-qknorm-experts",
    )
    query_key_variants = (
        "source-arithmetic",
        "residual-rotary-qknorm",
        "residual-rotary-qknorm-router",
        "residual-rotary-qknorm-experts",
    )
    if any(name in rotary_variants for name in variants):
        table_rows: dict[int, torch.Tensor] = {}
        suffix_positions = positions[:, prefix_token_count:].T.cpu()
        for token_positions, cache_rows in zip(
            suffix_positions,
            captured.operators["operator_00_rotary_cache_rows"][live],
            strict=True,
        ):
            for position, row in zip(token_positions.tolist(), cache_rows, strict=True):
                if position in table_rows:
                    assert torch.equal(table_rows[position], row)
                else:
                    table_rows[position] = row
        rotary_table = torch.zeros(
            (max(table_rows) + 1, reference.model.config.head_dim), dtype=suffix.dtype
        )
        for position, row in table_rows.items():
            rotary_table[position].copy_(row)
        rotary_table = rotary_table.to(device)
    else:
        pass
    expert_checks = 0
    rotary_checks = 0
    for layer_index in range(captured.accepted_layer):
        prefix = f"operator_{layer_index:02d}"
        if any(name in expert_variants for name in variants):
            native_output = captured.operators[f"{prefix}_moe_output"].to(device)
            calculated = source_rounded_experts(
                reference.model.layers[layer_index].mlp.experts,
                (
                    captured.operators[f"{prefix}_router_input"].to(device),
                    captured.operators[f"{prefix}_expert_indices"].long().to(device),
                    captured.operators[f"{prefix}_expert_weights"].to(device),
                ),
                native_output,
            )
            assert torch.equal(calculated[live], native_output[live]), layer_index
            expert_checks += 1
        else:
            pass
        if any(name in rotary_variants for name in variants):
            for kind in ("query", "key"):
                normalized = captured.operators[f"{prefix}_{kind}_normalized"][live].to(
                    device
                )
                expected = captured.operators[f"{prefix}_{kind}_rotated"][live].to(
                    device
                )
                normalized = normalized.reshape(
                    1, suffix.shape[1], -1, reference.model.config.head_dim
                ).transpose(1, 2)
                actual = (
                    contracted_bf16_rotation(
                        normalized,
                        position_embeddings[0].unsqueeze(1),
                        position_embeddings[1].unsqueeze(1),
                    )
                    .transpose(1, 2)
                    .reshape_as(expected)
                )
                assert torch.equal(actual, expected), (layer_index, kind)
            rotary_checks += 1
        else:
            pass
    baseline = torch.load(
        output_directory / "native-prefix-chained-operands.pt", weights_only=True
    )
    residual_checks: dict[str, dict[str, int]] = {}
    normalization_comparisons: dict[str, dict[str, dict[str, bool | float | None]]] = {}
    for name in variants:
        cache = DynamicCache(config=reference.model.config)
        hidden = suffix
        saved = {"checkpoint_hidden_0": suffix[0].cpu()}
        routing: dict[str, torch.Tensor] = {}
        rotary_calls: list[int] = []
        residual_control = NativeResidualControl()
        use_native_residual = name in (
            "residual-norm",
            "residual-norm-and-rotary",
            "source-arithmetic",
            "residual-rotary-qknorm",
            "residual-rotary-moe",
            "residual-rotary-qknorm-router",
            "residual-rotary-qknorm-experts",
        )
        for layer_index in range(captured.accepted_layer):
            cache.update(
                captured.keys[layer_index][:prefix_token_count]
                .to(device)
                .transpose(0, 1)
                .unsqueeze(0),
                captured.values[layer_index][:prefix_token_count]
                .to(device)
                .transpose(0, 1)
                .unsqueeze(0),
                layer_index,
            )
            layer = reference.model.layers[layer_index]
            with ExitStack() as hooks:
                rotary_control: NativeRotaryControl | None = None
                if name in rotary_variants:
                    assert rotary_table is not None
                    rotary_control = NativeRotaryControl(
                        positions=positions[:, prefix_token_count:].contiguous(),
                        table=rotary_table,
                        sections=reference.model.rotary_emb.mrope_section,
                        head_features=reference.model.config.head_dim,
                        epsilon=reference.model.config.rms_norm_eps,
                        query_norm_weight=(
                            layer.self_attn.q_norm.weight
                            if name in query_key_variants
                            else None
                        ),
                        key_norm_weight=(
                            layer.self_attn.k_norm.weight
                            if name in query_key_variants
                            else None
                        ),
                    )
                    hooks.enter_context(
                        patch.object(
                            modeling_qwen3_omni_moe,
                            "apply_rotary_pos_emb",
                            rotary_control,
                        )
                    )
                    if name in query_key_variants:
                        hooks.enter_context(
                            layer.self_attn.q_proj.register_forward_hook(
                                rotary_control.record_query_projection
                            )
                        )
                        hooks.enter_context(
                            layer.self_attn.k_proj.register_forward_hook(
                                rotary_control.record_key_projection
                            )
                        )
                    else:
                        pass
                else:
                    pass
                if use_native_residual:
                    for module, callback in (
                        (layer.input_layernorm, residual_control.input_norm),
                        (layer.self_attn.o_proj, residual_control.record_projection),
                        (
                            layer.post_attention_layernorm,
                            residual_control.post_attention_norm,
                        ),
                        (layer.mlp, residual_control.record_contribution),
                    ):
                        hooks.enter_context(module.register_forward_hook(callback))
                else:
                    pass
                if name in (
                    "router",
                    "router-and-experts",
                    "source-arithmetic",
                    "residual-rotary-moe",
                    "residual-rotary-qknorm-router",
                ):
                    hooks.enter_context(
                        layer.mlp.gate.register_forward_hook(keep_fp32_route_weights)
                    )
                else:
                    pass
                if name in expert_variants:
                    hooks.enter_context(
                        layer.mlp.experts.register_forward_hook(source_rounded_experts)
                    )
                else:
                    pass

                def record_routing(
                    router: Qwen3OmniMoeThinkerTextTopKRouter,
                    inputs: tuple[torch.Tensor],
                    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                ) -> None:
                    assert inputs[0].shape[-1] == router.hidden_dim
                    logits, weights, indices = output
                    for field, tensor in (
                        ("logits", logits),
                        ("weights", weights),
                        ("indices", indices),
                    ):
                        routing[f"layer_{layer_index:02d}_{field}"] = (
                            tensor.detach().cpu().clone()
                        )

                hooks.enter_context(
                    layer.mlp.gate.register_forward_hook(record_routing)
                )
                hidden = layer(
                    hidden,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=position_embeddings,
                )
                if rotary_control is not None:
                    assert rotary_control.calls == 1
                    rotary_calls.append(layer_index)
                    normalization_comparisons[f"{name}_layer_{layer_index:02d}"] = {
                        kind: asdict(comparison)
                        for kind, comparison in rotary_control.normalization_comparisons.items()
                    }
                else:
                    pass
                if use_native_residual:
                    residual_control.check_layer(hidden)
                else:
                    pass
            keys = cache.layers[layer_index].keys
            values = cache.layers[layer_index].values
            assert keys is not None and values is not None
            assert torch.equal(
                keys[0, :, :prefix_token_count].transpose(0, 1).cpu(),
                captured.keys[layer_index][:prefix_token_count],
            )
            assert torch.equal(
                values[0, :, :prefix_token_count].transpose(0, 1).cpu(),
                captured.values[layer_index][:prefix_token_count],
            )
            assert bool(torch.isfinite(hidden).all())
            saved[f"checkpoint_hidden_{layer_index + 1}"] = hidden[0].cpu()
            if name == "checkpoint":
                assert torch.equal(
                    saved[f"checkpoint_hidden_{layer_index + 1}"],
                    baseline[f"checkpoint_hidden_{layer_index + 1}"],
                ), layer_index
            else:
                pass
        torch.save(
            {
                "embedded_input": embedded.cpu(),
                "mrope_positions": positions.cpu(),
                "native_accepted_hidden": captured.native_accepted_hidden,
                **saved,
            },
            output_directory / f"native-prefix-{name}-operands.pt",
        )
        torch.save(routing, output_directory / f"native-prefix-{name}-routing.pt")
        assert rotary_calls == (
            list(range(captured.accepted_layer)) if name in rotary_variants else []
        )
        assert residual_control.input_norm_calls == (
            captured.accepted_layer if use_native_residual else 0
        )
        assert residual_control.post_attention_norm_calls == (
            captured.accepted_layer if use_native_residual else 0
        )
        residual_checks[name] = {
            "input_norm_calls": residual_control.input_norm_calls,
            "post_attention_norm_calls": residual_control.post_attention_norm_calls,
        }
        print(json.dumps({"completed_precision_control": name}), flush=True)
    (output_directory / "precision-controls.json").write_text(
        json.dumps(
            {
                "source_expert_formula_native_operand_checks": expert_checks,
                "source_rotary_formula_native_operand_checks": rotary_checks,
                "checkpoint_repeated_boundaries_exact": captured.accepted_layer + 1,
                "variants": list(variants),
                "residual_control_checks": residual_checks,
                "query_key_normalization": normalization_comparisons,
                "scope": "Only the stated arithmetic families change in an independently loaded checkpoint producer. Prefix KV, embeddings, positions and other layer operations stay fixed. Source-style expert arithmetic uses cuBLAS rather than the native Triton kernel. The rotary control runs the actual native Triton rotary kernel with retained native tables and independently checks every resulting Q/K against the measured BF16 multiply/FMA formula before propagation. Residual controls run the original FlashInfer normalization kernels, retaining the unrounded two-operand sum for normalization and the BF16 residual for the next addition. Every layer checks composed residual identity. Source-arithmetic also applies the real paired Q/K-normalization kernel before rotation, retains FP32 router scales and uses the already-validated source-style expert rounding. Its norm distance to an FP32 formula is measured separately, not assumed bit-identical. Attention and linear projections remain reference operations. These are causal diagnostic controls, not production repairs or a full native-backend replay.",
            },
            indent=2,
        )
    )
