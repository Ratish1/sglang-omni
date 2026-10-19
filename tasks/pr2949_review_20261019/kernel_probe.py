"""Check fused ring writes and paged attention against independent tensor operations."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.rope import (
    apply_rope_inplace,
    apply_rope_inplace_with_kvcache,
)

from sglang_omni.models.qwen3_omni.components.codec_state_pool import ring_addresses


def main() -> None:
    torch.manual_seed(2949)
    head_dim, query_heads, kv_heads = 64, 4, 2
    ring_frames, retained_frames = 81, 71
    slots = torch.tensor([3, 1, 5], device="cuda")
    inverse_frequencies = 1 / (
        10000 ** (torch.arange(0, head_dim, 2, device="cuda").float() / head_dim)
    )
    angles = torch.outer(torch.arange(512, device="cuda").float(), inverse_frequencies)
    rotary_table = torch.cat((angles.cos(), angles.sin()), dim=-1)
    rows = []
    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float16):
            for positions in ((0, 1, 70), (71, 80, 81), (82, 162, 200)):
                for new_frames in (1, 3, 10):
                    addresses = ring_addresses(
                        slots,
                        torch.tensor(positions, device="cuda"),
                        new_frames,
                        ring_frames,
                        retained_frames,
                    )
                    projected = torch.randn(
                        3 * new_frames,
                        (query_heads + 2 * kv_heads) * head_dim,
                        device="cuda",
                        dtype=dtype,
                    )
                    query, key, value = projected.split(
                        (
                            query_heads * head_dim,
                            kv_heads * head_dim,
                            kv_heads * head_dim,
                        ),
                        dim=-1,
                    )
                    query = query.view(-1, query_heads, head_dim)
                    key = key.view(-1, kv_heads, head_dim)
                    value = value.view(-1, kv_heads, head_dim)
                    wanted_query, wanted_key = query.clone(), key.clone()
                    apply_rope_inplace(
                        wanted_query,
                        wanted_key,
                        rotary_table,
                        addresses.frame_positions,
                        is_neox=True,
                    )
                    keys = torch.randn(
                        6 * ring_frames, kv_heads * head_dim, device="cuda", dtype=dtype
                    )
                    values = torch.randn_like(keys)
                    wanted_keys, wanted_values = keys.clone(), values.clone()
                    wanted_keys.index_copy_(
                        0, addresses.write_rows, wanted_key.flatten(1)
                    )
                    wanted_values.index_copy_(0, addresses.write_rows, value.flatten(1))
                    apply_rope_inplace_with_kvcache(
                        query,
                        key,
                        value,
                        keys,
                        values,
                        rotary_table,
                        addresses.frame_positions,
                        addresses.write_rows,
                        is_neox=True,
                    )
                    assert torch.equal(query, wanted_query)
                    assert torch.equal(key, wanted_key)
                    assert torch.equal(keys, wanted_keys)
                    assert torch.equal(values, wanted_values)
                    queries = query.view(3, new_frames, query_heads, head_dim)
                    paged = flash_attn_with_kvcache(
                        queries,
                        keys.view(-1, 1, kv_heads, head_dim),
                        values.view(-1, 1, kv_heads, head_dim),
                        page_table=addresses.window_rows.to(torch.int32),
                        cache_seqlens=addresses.window_lengths.to(torch.int32),
                        causal=True,
                        window_size=(retained_frames, 0),
                    )
                    query_positions = torch.tensor(positions, device="cuda")[
                        :, None
                    ] + torch.arange(new_frames, device="cuda")
                    window_starts = (
                        torch.tensor(positions, device="cuda") - retained_frames
                    ).clamp(min=0)
                    key_positions = window_starts[:, None] + torch.arange(
                        ring_frames, device="cuda"
                    )
                    differences = (
                        query_positions[:, :, None] - key_positions[:, None, :]
                    )
                    mask = ((differences >= 0) & (differences <= retained_frames))[
                        :, None
                    ]
                    reference = F.scaled_dot_product_attention(
                        queries.float().transpose(1, 2),
                        keys[addresses.window_rows]
                        .view(3, ring_frames, kv_heads, head_dim)
                        .float()
                        .transpose(1, 2),
                        values[addresses.window_rows]
                        .view(3, ring_frames, kv_heads, head_dim)
                        .float()
                        .transpose(1, 2),
                        attn_mask=mask,
                        enable_gqa=True,
                    ).transpose(1, 2)
                    torch.testing.assert_close(
                        paged.float(), reference, atol=0.02, rtol=0.02
                    )
                    rows.append(
                        {
                            "dtype": str(dtype),
                            "start_positions": positions,
                            "new_frames": new_frames,
                            "max_absolute_attention_error": float(
                                (paged.float() - reference).abs().max()
                            ),
                        }
                    )
    result = {"checks": rows, "fused_query_key_and_ring_bit_identical": True}
    Path("/data/review2949/kernel.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
