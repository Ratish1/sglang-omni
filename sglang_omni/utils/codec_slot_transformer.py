# SPDX-License-Identifier: Apache-2.0
"""The codec decoders' sliding-window transformer layers on fused kernels for slot decodes: each
layer stores its new K/V in the rows' rings with the rope kernel and attends to its own frames on a
stream's first decode, to its window in the ring on a later one."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.layernorm import rmsnorm

from sglang_omni.platforms.interface import PagedWindowAttention, RopeKVStoreKernel
from sglang_omni.utils.codec_state_pool import CodecStatePool, RingAddresses


def merge_layer_weights(layers: torch.nn.ModuleList) -> None:
    """Give every layer one q, k, v matrix and one gate, up matrix whose slices the projections'
    weights become, so the fused GEMMs read the loaded weights without a second copy."""
    for layer in layers:
        attention, mlp = layer.self_attn, layer.mlp
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


def rope_cos_sin_table(rotary: torch.nn.Module, max_positions: int) -> torch.Tensor:
    """The rope kernel's (max_positions, head_dim) table: cos then sin of each position's angles,
    scaled as the rotary module scales them."""
    positions = torch.arange(
        max_positions, device=rotary.inv_freq.device, dtype=torch.float32
    )
    angles = torch.outer(positions, rotary.inv_freq.float())
    return torch.cat((angles.cos(), angles.sin()), dim=-1) * rotary.attention_scaling


def slot_transformer_layers(
    layers: torch.nn.ModuleList,
    inputs_embeds: torch.Tensor,
    pool: CodecStatePool,
    addresses: RingAddresses,
    cos_sin_cache: torch.Tensor,
    *,
    first_decode: bool,
    sliding_window: int,
    rope_store: RopeKVStoreKernel,
    paged_attention: PagedWindowAttention,
) -> torch.Tensor:
    """The (B x L, hidden) states after the layers, before the final norm, of the rows' (B, L,
    hidden) new frames. A first decode attends causally over its own frames, within the window.
    """
    batch_size, length, hidden_size = inputs_embeds.shape
    head_dim = layers[0].self_attn.head_dim
    if length <= sliding_window:
        window_mask = None
    else:
        query = torch.arange(length, device=inputs_embeds.device)[:, None]
        key = torch.arange(length, device=inputs_embeds.device)[None, :]
        window_mask = (key <= query) & (query - key < sliding_window)
    hidden_states = inputs_embeds.reshape(batch_size * length, hidden_size)
    for layer_index, layer in enumerate(layers):
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
        value = value.view(batch_size * length, -1, head_dim)
        rope_store(
            query,
            key,
            value,
            pool.keys[layer_index],
            pool.values[layer_index],
            cos_sin_cache,
            addresses.frame_positions,
            addresses.write_rows,
            is_neox=True,
        )
        if first_decode:
            attended = F.scaled_dot_product_attention(
                query.view(batch_size, length, -1, head_dim).transpose(1, 2),
                key.view(batch_size, length, -1, head_dim).transpose(1, 2),
                value.view(batch_size, length, -1, head_dim).transpose(1, 2),
                attn_mask=window_mask,
                is_causal=window_mask is None,
                scale=attention.scaling,
                enable_gqa=attention.num_key_value_groups > 1,
            ).transpose(1, 2)
        else:
            attended = paged_attention(
                query.view(batch_size, length, -1, head_dim),
                pool.keys[layer_index].view(-1, 1, key.shape[1], head_dim),
                pool.values[layer_index].view(-1, 1, key.shape[1], head_dim),
                page_table=addresses.window_rows,
                cache_seqlens=addresses.window_lengths,
                softmax_scale=attention.scaling,
                causal=True,
                window_size=(pool.retained_frames, 0),
            )
        attended = attended.reshape(batch_size * length, -1)
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
    return hidden_states
