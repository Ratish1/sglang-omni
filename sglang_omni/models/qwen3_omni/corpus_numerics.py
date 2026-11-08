"""Measure local numerical error on every recorded predictor operation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker


class PredictorNumerics:
    def __init__(self, native: Qwen3OmniTalker) -> None:
        self.layers = native.code_predictor.model.layers
        self.weights = [
            {
                "qkv": layer.self_attn.qkv_proj.weight.double(),
                "output": layer.self_attn.o_proj.weight.double(),
                "gate_up": layer.mlp.gate_up_proj.weight.double(),
                "down": layer.mlp.down_proj.weight.double(),
            }
            for layer in self.layers
        ]

    def measure(
        self, dense: dict[str, torch.Tensor], live_rows: int
    ) -> dict[str, torch.Tensor]:
        measurements: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.layers):
            actual = {
                name: torch.cat(
                    (
                        dense[f"opening_{name}"][layer_index][None],
                        tensor[:, layer_index],
                    ),
                    dim=0,
                )[:, :live_rows].cuda()
                for name, tensor in dense.items()
                if name
                in (
                    "input",
                    "query",
                    "attention",
                    "attention_residual",
                    "activation",
                    "mlp_residual",
                )
            }
            weights = self.weights[layer_index]
            attention = layer.self_attn
            token_count = actual["input"].shape[0]
            queries = (
                actual["query"]
                .double()
                .reshape(
                    token_count, live_rows, attention.num_heads, attention.head_dim
                )
                .permute(1, 2, 0, 3)
            )
            caches = {}
            for name in ("keys", "values"):
                written = dense[name][:, layer_index, :live_rows]
                caches[name] = (
                    torch.cat(
                        (
                            written[0, :, 0][None],
                            written[0, :, 1][None],
                            written[1:, :, 0],
                        ),
                        dim=0,
                    )
                    .cuda()
                    .double()
                    .reshape(
                        token_count,
                        live_rows,
                        attention.num_kv_heads,
                        attention.head_dim,
                    )
                    .permute(1, 2, 0, 3)
                    .repeat_interleave(
                        attention.num_heads // attention.num_kv_heads, dim=1
                    )
                )
            scores = (
                queries @ caches["keys"].transpose(-1, -2)
            ) * attention.head_dim**-0.5
            causal_mask = torch.ones(
                token_count, token_count, device="cuda", dtype=torch.bool
            ).triu(1)
            scores.masked_fill_(causal_mask, -torch.inf)
            expected_attention = (
                (scores.softmax(-1) @ caches["values"]).permute(2, 0, 1, 3).flatten(2)
            )
            comparisons = {
                "attention_fp64_stored_qkv": (actual["attention"], expected_attention)
            }
            inputs = actual["input"].double()
            scale = (
                inputs.square().mean(-1, keepdim=True)
                + layer.input_layernorm.variance_epsilon
            ).rsqrt()
            weighted = inputs * layer.input_layernorm.weight.double()
            canonical_qkv = (weighted * scale).bfloat16().double() @ weights["qkv"].T
            delayed_qkv = (weighted.bfloat16().double() @ weights["qkv"].T) * scale
            full_qkv = (weighted * scale) @ weights["qkv"].T
            value_width = attention.num_kv_heads * attention.head_dim
            actual_values = torch.cat(
                (
                    dense["values"][0, layer_index, :live_rows, 0][None],
                    dense["values"][0, layer_index, :live_rows, 1][None],
                    dense["values"][1:, layer_index, :live_rows, 0],
                ),
                dim=0,
            ).cuda()
            comparisons["value_canonical_bf16_input"] = (
                actual_values,
                canonical_qkv[..., -value_width:].bfloat16().double(),
            )
            comparisons["value_delayed_norm"] = (
                actual_values,
                delayed_qkv[..., -value_width:].bfloat16().double(),
            )
            comparisons["value_fp64_projection"] = (
                actual_values,
                full_qkv[..., -value_width:],
            )
            attention_projection = actual["attention"].double() @ weights["output"].T
            expected_attention_residual = (
                attention_projection.bfloat16() + actual["input"]
            ).bfloat16()
            comparisons["attention_residual_fp64_projection"] = (
                actual["attention_residual"],
                expected_attention_residual,
            )
            residual = actual["attention_residual"].double()
            scale = (
                residual.square().mean(-1, keepdim=True)
                + layer.post_attention_layernorm.variance_epsilon
            ).rsqrt()
            weighted = residual * layer.post_attention_layernorm.weight.double()
            for name, projected in (
                (
                    "canonical_bf16_input",
                    (weighted * scale).bfloat16().double() @ weights["gate_up"].T,
                ),
                (
                    "delayed_norm",
                    (weighted.bfloat16().double() @ weights["gate_up"].T) * scale,
                ),
            ):
                gate, up = projected.bfloat16().double().chunk(2, dim=-1)
                comparisons[f"activation_{name}"] = (
                    actual["activation"],
                    (F.silu(gate) * up).bfloat16(),
                )
            down_projection = actual["activation"].double() @ weights["down"].T
            expected_mlp_residual = (
                down_projection.bfloat16() + actual["attention_residual"]
            ).bfloat16()
            comparisons["mlp_residual_fp64_projection"] = (
                actual["mlp_residual"],
                expected_mlp_residual,
            )
            for name, (observed, reference) in comparisons.items():
                difference = observed.double() - reference.double()
                measurements[
                    f"numerics.{name}.{layer_index}.relative_l2"
                ] = difference.norm(dim=-1) / reference.double().norm(dim=-1).clamp_min(
                    torch.finfo(torch.float64).tiny
                )
                measurements[f"numerics.{name}.{layer_index}.max_absolute"] = (
                    difference.abs().amax(-1)
                )
            if layer_index > 0:
                previous = torch.cat(
                    (
                        dense["opening_mlp_residual"][layer_index - 1][None],
                        dense["mlp_residual"][:, layer_index - 1],
                    ),
                    dim=0,
                )[:, :live_rows].cuda()
                measurements[f"numerics.layer_input_chain.{layer_index}.exact"] = (
                    actual["input"] == previous
                ).all(-1)
            else:
                pass
        return {name: tensor.cpu() for name, tensor in measurements.items()}
