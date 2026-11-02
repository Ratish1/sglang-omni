# SPDX-License-Identifier: Apache-2.0
"""Triton kernel of the thinker's decode attention inputs.

At decode, norm_rope_store replaces three launches (the q and k RMSNorm, the interleaved
MRoPE and the KV store) with one, bit for bit: the norm reproduces the fused q/k norm's
reduction (lane t holds elements 4t to 4t + 3, a sequential per-lane fold, a butterfly
pairing lane i with i ^ 16 first, an approximate rsqrt, (x * rstd) * w rounded once), the
rotation is the bf16 fmas the MRoPE kernel compiles to on the same bf16 table (x1 * cos -
x2 * sin with x2 * sin rounded apart, x1 * sin + x2 * cos with x2 * cos rounded apart; the
equality test guards that lowering), and the store copies the rotated k and the v rows
into each token's cache slot, skipping the reserved slot 0. q is written contiguous, so
the attention makes no copy of it.
"""

from __future__ import annotations

import torch
from sglang.kernels.ops.attention.rope import FusedSetKVBufferArg
from sglang.srt.environ import envs
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.models.qwen3_moe import Qwen3MoeAttention
from sglang.srt.runtime_context import get_exec

from sglang_omni.platforms import current_platform

try:  # keep the module importable where Triton is unavailable
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None
    HAS_TRITON = False

HEAD_DIM = 128
# note (ratish): q heads per program, one more program takes the kv heads and the store;
# measured faster than one program per token at every decode bucket from 1 to 64.
Q_GROUP = 4


if HAS_TRITON:

    @triton.jit
    def norm_rotate(values, weight, eps, cos_row, sin_row, HEADS: tl.constexpr):
        """RMSNorm then NeoX rotation of [HEADS, 32 lanes, 4] fp32 head values."""
        element = tl.arange(0, 4)
        squares = values * values
        lane_sums = tl.zeros([HEADS, 32], dtype=tl.float32)
        for index in tl.static_range(4):
            lane_sums += tl.sum(
                tl.where(element[None, None, :] == index, squares, 0.0), 2
            )
        lane_sums = tl.sum(tl.reshape(lane_sums, [HEADS, 2, 16]), 1)
        lane_sums = tl.sum(tl.reshape(lane_sums, [HEADS, 2, 8]), 1)
        lane_sums = tl.sum(tl.reshape(lane_sums, [HEADS, 2, 4]), 1)
        lane_sums = tl.sum(tl.reshape(lane_sums, [HEADS, 2, 2]), 1)
        total = tl.sum(lane_sums, 1)
        rstd = libdevice.rsqrt(total * (1.0 / 128) + eps)
        normed = (values * rstd[:, None, None] * weight[None, :, :]).to(tl.bfloat16)
        halves = tl.permute(tl.reshape(normed, [HEADS, 2, 64]), (0, 2, 1))
        first, second = tl.split(halves)
        # note (ratish): the bf16 fmas the MRoPE kernel's rotation compiles to, one
        # product rounded apart.
        return (
            tl.fma(first, cos_row[None, :], -(second * sin_row[None, :])),
            tl.fma(first, sin_row[None, :], second * cos_row[None, :]),
        )

    @triton.jit
    def norm_rope_store_kernel(
        QKV,
        Q_OUT,
        K_CACHE,
        V_CACHE,
        CACHE_SLOTS,
        POSITIONS,
        COS_SIN_CACHE,
        Q_WEIGHT,
        K_WEIGHT,
        eps,
        qkv_stride,
        cache_stride,
        positions_stride,
        Q_HEADS: tl.constexpr,
        KV_HEADS: tl.constexpr,
        Q_GROUP: tl.constexpr,
        SECTION_H: tl.constexpr,
        SECTION_W: tl.constexpr,
    ):
        token = tl.program_id(0)
        qkv_row = QKV + token * qkv_stride
        position_t = tl.load(POSITIONS + token)
        position_h = tl.load(POSITIONS + positions_stride + token)
        position_w = tl.load(POSITIONS + 2 * positions_stride + token)
        # note (ratish): the MRoPE kernel's interleaved section select and bf16 sums of
        # the masked rows.
        offsets = tl.arange(0, 64)
        h_mask = ((offsets % 3) == 1) & (offsets <= 3 * SECTION_H)
        w_mask = ((offsets % 3) == 2) & (offsets <= 3 * SECTION_W)
        t_mask = ~(h_mask | w_mask)
        cos_row = (
            tl.load(COS_SIN_CACHE + position_t * 128 + offsets, mask=t_mask, other=0)
            + tl.load(COS_SIN_CACHE + position_h * 128 + offsets, mask=h_mask, other=0)
            + tl.load(COS_SIN_CACHE + position_w * 128 + offsets, mask=w_mask, other=0)
        )
        sin_row = (
            tl.load(
                COS_SIN_CACHE + position_t * 128 + 64 + offsets, mask=t_mask, other=0
            )
            + tl.load(
                COS_SIN_CACHE + position_h * 128 + 64 + offsets, mask=h_mask, other=0
            )
            + tl.load(
                COS_SIN_CACHE + position_w * 128 + 64 + offsets, mask=w_mask, other=0
            )
        )

        lane_offsets = tl.arange(0, 32)[:, None] * 4 + tl.arange(0, 4)[None, :]
        group = tl.program_id(1)
        if group < Q_HEADS // Q_GROUP:
            heads = tl.arange(0, Q_GROUP)
            q_row = qkv_row + group * Q_GROUP * 128
            first, second = norm_rotate(
                tl.load(
                    q_row + heads[:, None, None] * 128 + lane_offsets[None, :, :]
                ).to(tl.float32),
                tl.load(Q_WEIGHT + lane_offsets).to(tl.float32),
                eps,
                cos_row,
                sin_row,
                Q_GROUP,
            )
            half = heads[:, None] * 128 + offsets[None, :]
            q_out_row = Q_OUT + token * Q_HEADS * 128 + group * Q_GROUP * 128
            tl.store(q_out_row + half, first)
            tl.store(q_out_row + half + 64, second)
        else:
            heads = tl.arange(0, KV_HEADS)
            k_row = qkv_row + Q_HEADS * 128
            first, second = norm_rotate(
                tl.load(
                    k_row + heads[:, None, None] * 128 + lane_offsets[None, :, :]
                ).to(tl.float32),
                tl.load(K_WEIGHT + lane_offsets).to(tl.float32),
                eps,
                cos_row,
                sin_row,
                KV_HEADS,
            )
            half = heads[:, None] * 128 + offsets[None, :]
            tl.store(k_row + half, first)
            tl.store(k_row + half + 64, second)
            # note (ratish): slot 0 is the sink padded graph rows point at; the store
            # kernel skips it.
            cache_slot = tl.load(CACHE_SLOTS + token)
            is_real_slot = cache_slot != 0
            k_cache_row = K_CACHE + cache_slot * cache_stride
            tl.store(k_cache_row + half, first, mask=is_real_slot)
            tl.store(k_cache_row + half + 64, second, mask=is_real_slot)
            v_offsets = tl.arange(0, KV_HEADS * 128)
            v_values = tl.load(qkv_row + (Q_HEADS + KV_HEADS) * 128 + v_offsets)
            tl.store(
                V_CACHE + cache_slot * cache_stride + v_offsets,
                v_values,
                mask=is_real_slot,
            )

else:
    pass


def supports_norm_rope_store(attention: Qwen3MoeAttention) -> bool:
    """Whether the kernel reproduces this attention's decode norm, rotation and store."""
    if not (HAS_TRITON and current_platform.is_cuda()):
        return False
    else:
        pass
    rotary = attention.rotary_emb
    q_norm, k_norm = attention.q_norm, attention.k_norm
    return (
        isinstance(rotary, MRotaryEmbedding)
        and rotary.mrope_interleaved
        and not rotary.mrope_interleaved_glm
        and rotary.is_neox_style
        and rotary.rotary_dim == HEAD_DIM
        and attention.head_dim == HEAD_DIM
        and attention.num_heads == attention.total_num_heads
        and attention.num_heads % Q_GROUP == 0
        and triton.next_power_of_2(attention.num_kv_heads) == attention.num_kv_heads
        and q_norm.variance_epsilon == k_norm.variance_epsilon
        and not q_norm.cast_x_before_out_mul
        # note (ratish): these keep sglang's decode path on the fused q/k norm kernel.
        and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
        and get_exec().graph.cuda_graph_config.prefill.tc_compiler != "inductor"
    )


def norm_rope_store(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    attention: Qwen3MoeAttention,
    kv_store: FusedSetKVBufferArg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns q contiguous, and k (normed and rotated in place) and v as views of qkv."""
    tokens = qkv.shape[0]
    q_size = attention.num_heads * HEAD_DIM
    kv_size = attention.num_kv_heads * HEAD_DIM
    q = torch.empty((tokens, q_size), dtype=qkv.dtype, device=qkv.device)
    assert attention.rotary_emb.cos_sin_cache.dtype == qkv.dtype
    assert kv_store.k_buffer.stride(0) == kv_store.v_buffer.stride(0)
    norm_rope_store_kernel[(tokens, attention.num_heads // Q_GROUP + 1)](
        qkv,
        q,
        kv_store.k_buffer,
        kv_store.v_buffer,
        kv_store.cache_loc,
        positions,
        attention.rotary_emb.cos_sin_cache,
        attention.q_norm.weight,
        attention.k_norm.weight,
        attention.q_norm.variance_epsilon,
        qkv.stride(0),
        kv_store.k_buffer.stride(0),
        positions.stride(0),
        Q_HEADS=attention.num_heads,
        KV_HEADS=attention.num_kv_heads,
        Q_GROUP=Q_GROUP,
        SECTION_H=attention.rotary_emb.mrope_section[1],
        SECTION_W=attention.rotary_emb.mrope_section[2],
        enable_fp_fusion=False,
    )
    return q, qkv[:, q_size : q_size + kv_size], qkv[:, q_size + kv_size :]
