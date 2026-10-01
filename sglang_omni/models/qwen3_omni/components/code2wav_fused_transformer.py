# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni code2wav's pre-transformer on fused kernels, for CUDA in bf16 and fp16."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeCode2WavTransformerModel,
)


class FusedCode2WavTransformer(torch.nn.Module):
    """The pre-transformer on its own weights with one qkv and one gate-up GEMM per layer, and
    RMSNorm, rotary, SwiGLU and layer scale plus residual one kernel each."""

    def __init__(self, transformer: Qwen3OmniMoeCode2WavTransformerModel) -> None:
        super().__init__()
        self.transformer = transformer
        rotary = transformer.rotary_emb
        positions = torch.arange(
            transformer.config.max_position_embeddings,
            device=rotary.inv_freq.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, rotary.inv_freq.float())
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((angles.cos(), angles.sin()), dim=-1) * rotary.attention_scaling,
            persistent=False,
        )
        # The q, k, v and the gate, up weights become views of one matrix each, so the fused
        # GEMMs read the loaded values with no second copy.
        for layer in transformer.layers:
            attention, mlp = layer.self_attn, layer.mlp
            assert attention.q_proj.bias is None, "code2wav attention has no bias"
            projections = (attention.q_proj, attention.k_proj, attention.v_proj)
            attention.qkv_weight = torch.cat([p.weight.data for p in projections])
            start = 0
            for projection in projections:
                rows = projection.weight.shape[0]
                projection.weight.data = attention.qkv_weight[start : start + rows]
                start += rows
            mlp.gate_up_weight = torch.cat(
                (mlp.gate_proj.weight.data, mlp.up_proj.weight.data)
            )
            mlp.gate_proj.weight.data = mlp.gate_up_weight[: mlp.intermediate_size]
            mlp.up_proj.weight.data = mlp.gate_up_weight[mlp.intermediate_size :]

    def forward(self, inputs_embeds: torch.Tensor) -> BaseModelOutputWithPast:
        # note (ratish): these kernels exist only on CUDA builds, where this module is installed.
        from sgl_kernel import rmsnorm, silu_and_mul
        from sglang.kernels.ops.attention.rope import (
            apply_rope_with_cos_sin_cache_inplace,
        )

        batch_size, length, hidden_size = inputs_embeds.shape
        first = self.transformer.layers[0].self_attn
        head_dim = first.head_dim
        positions = torch.arange(length, device=inputs_embeds.device).repeat(batch_size)
        if length <= first.sliding_window:
            window_mask = None
        else:
            query = torch.arange(length, device=inputs_embeds.device)[:, None]
            key = torch.arange(length, device=inputs_embeds.device)[None, :]
            window_mask = (key <= query) & (query - key < first.sliding_window)
        hidden_states = inputs_embeds.reshape(batch_size * length, hidden_size)
        for layer in self.transformer.layers:
            attention = layer.self_attn
            residual = hidden_states
            normed = rmsnorm(
                hidden_states,
                layer.input_layernorm.weight,
                layer.input_layernorm.variance_epsilon,
            )
            query, key, value = F.linear(normed, attention.qkv_weight).split(
                [
                    attention.q_proj.out_features,
                    attention.k_proj.out_features,
                    attention.v_proj.out_features,
                ],
                dim=-1,
            )
            query = query.view(batch_size * length, -1, head_dim)
            key = key.view(batch_size * length, -1, head_dim)
            apply_rope_with_cos_sin_cache_inplace(
                query, key, self.cos_sin_cache, positions, is_neox=True
            )
            attended = F.scaled_dot_product_attention(
                query.view(batch_size, length, -1, head_dim).transpose(1, 2),
                key.view(batch_size, length, -1, head_dim).transpose(1, 2),
                value.view(batch_size, length, -1, head_dim).transpose(1, 2),
                attn_mask=window_mask,
                is_causal=window_mask is None,
                scale=attention.scaling,
                enable_gqa=attention.num_key_value_groups > 1,
            )
            attended = attended.transpose(1, 2).reshape(batch_size * length, -1)
            hidden_states = torch.addcmul(
                residual,
                layer.self_attn_layer_scale.scale,
                F.linear(attended, attention.o_proj.weight),
            )
            residual = hidden_states
            normed = rmsnorm(
                hidden_states,
                layer.post_attention_layernorm.weight,
                layer.post_attention_layernorm.variance_epsilon,
            )
            activated = silu_and_mul(F.linear(normed, layer.mlp.gate_up_weight))
            hidden_states = torch.addcmul(
                residual,
                layer.mlp_layer_scale.scale,
                F.linear(activated, layer.mlp.down_proj.weight),
            )
        hidden_states = rmsnorm(
            hidden_states,
            self.transformer.norm.weight,
            self.transformer.norm.variance_epsilon,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states.view(batch_size, length, hidden_size)
        )


def fuse_code2wav_transformer(model: Qwen3OmniMoeCode2Wav) -> bool:
    """Install the fused pre-transformer where its kernels run; returns whether it did."""
    weight = model.code_embedding.weight
    if weight.device.type != "cuda" or weight.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return False
    else:
        model.pre_transformer = FusedCode2WavTransformer(model.pre_transformer)
        return True
