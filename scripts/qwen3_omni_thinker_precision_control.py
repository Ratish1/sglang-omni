"""Separate arithmetic families in checkpoint conditioning."""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import torch
from sglang.kernels.ops.attention.rotary_triton import triton_mrope_fused
from transformers import DynamicCache
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextExperts,
    Qwen3OmniMoeThinkerTextTopKRouter,
)

from scripts.debug_qwen3_omni_thinker_rotary import contracted_bf16_rotation


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
        Literal["checkpoint", "router", "experts", "router-and-experts", "rotary"], ...
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
    if "rotary" in variants:
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
        if "experts" in variants or "router-and-experts" in variants:
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
        if "rotary" in variants:
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
    for name in variants:
        cache = DynamicCache(config=reference.model.config)
        hidden = suffix
        saved = {"checkpoint_hidden_0": suffix[0].cpu()}
        routing: dict[str, torch.Tensor] = {}
        rotary_calls: list[int] = []
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
                if name == "rotary":

                    def apply_rotary(
                        query: torch.Tensor,
                        key: torch.Tensor,
                        cosine: torch.Tensor,
                        sine: torch.Tensor,
                        unsqueeze_dim: int = 1,
                    ) -> tuple[torch.Tensor, torch.Tensor]:
                        assert (
                            query.dtype
                            == key.dtype
                            == cosine.dtype
                            == sine.dtype
                            == torch.bfloat16
                        )
                        rotary_calls.append(layer_index)
                        cosine = cosine.unsqueeze(unsqueeze_dim)
                        sine = sine.unsqueeze(unsqueeze_dim)
                        expected_query, expected_key = (
                            contracted_bf16_rotation(query, cosine, sine),
                            contracted_bf16_rotation(key, cosine, sine),
                        )
                        assert query.shape[0] == key.shape[0] == 1
                        query_rows = (
                            query.transpose(1, 2).reshape(query.shape[2], -1).clone()
                        )
                        key_rows = key.transpose(1, 2).reshape(key.shape[2], -1).clone()
                        assert rotary_table is not None
                        triton_mrope_fused(
                            query_rows,
                            key_rows,
                            rotary_table,
                            positions[:, prefix_token_count:].contiguous(),
                            reference.model.rotary_emb.mrope_section,
                            reference.model.config.head_dim,
                            reference.model.config.head_dim,
                            True,
                            False,
                            True,
                            None,
                        )
                        query_output = query_rows.reshape(
                            1, query.shape[2], query.shape[1], query.shape[3]
                        ).transpose(1, 2)
                        key_output = key_rows.reshape(
                            1, key.shape[2], key.shape[1], key.shape[3]
                        ).transpose(1, 2)
                        assert torch.equal(query_output, expected_query)
                        assert torch.equal(key_output, expected_key)
                        return query_output, key_output

                    hooks.enter_context(
                        patch.object(
                            modeling_qwen3_omni_moe,
                            "apply_rotary_pos_emb",
                            apply_rotary,
                        )
                    )
                else:
                    pass
                if name in ("router", "router-and-experts"):
                    hooks.enter_context(
                        layer.mlp.gate.register_forward_hook(keep_fp32_route_weights)
                    )
                else:
                    pass
                if name in ("experts", "router-and-experts"):
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
            list(range(captured.accepted_layer)) if name == "rotary" else []
        )
        print(json.dumps({"completed_precision_control": name}), flush=True)
    (output_directory / "precision-controls.json").write_text(
        json.dumps(
            {
                "source_expert_formula_native_operand_checks": expert_checks,
                "source_rotary_formula_native_operand_checks": rotary_checks,
                "checkpoint_repeated_boundaries_exact": captured.accepted_layer + 1,
                "variants": list(variants),
                "scope": "Only the stated arithmetic family changes in an independently loaded checkpoint producer. Prefix KV, embeddings, positions and other layer operations stay fixed. Source-style expert arithmetic uses cuBLAS rather than the native Triton kernel. The rotary control runs the actual native Triton rotary kernel with retained native tables and independently checks every resulting Q/K against the measured BF16 multiply/FMA formula before propagation. Each requested formula must reproduce every captured native operator batch first. These are causal diagnostic controls, not production repairs.",
            },
            indent=2,
        )
    )
