"""Research probe: real CFG FlashInfer planner/kernel versus dense attention."""

from types import SimpleNamespace as NS

import torch
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
)
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
)
from sglang.srt.layers.radix_attention import AttentionType

from sglang_omni.models.llada2_uni.cfg_attention_backend import (
    LLaDA2CFGFlashInferAttnBackend,
)

torch.manual_seed(20261008)
device = "cuda"
heads, dimension, block_size, capacity = 2, 128, 32, 160
workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
backend = object.__new__(LLaDA2CFGFlashInferAttnBackend)
backend.num_wrappers = 1
backend.workspace_buffer = workspace
backend.cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
    workspace, "NHD", backend="fa2"
)
backend.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
    workspace, "NHD", backend="fa2"
)
backend.prefill_wrappers_paged = [
    BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD", backend="fa2")
]
backend.prefill_split_tile_size = None
backend.dq_paged_kernel_lens = None
backend.prefill_uses_dequant_workspace = False
backend.is_dllm_model = True
backend._kv_write_scales = lambda layer: (None, None)


class Translator:
    reads_are_translated = False

    def fill_packed_read_stream(
        self,
        *,
        req_pool_indices,
        seq_lens,
        indptr,
        total_tokens,
        out,
        kv_start_idx,
        sliding_window,
    ):
        for index, (request, length) in enumerate(
            zip(req_pool_indices.tolist(), seq_lens.tolist())
        ):
            start = 0 if kv_start_idx is None else int(kv_start_idx[index])
            out[indptr[index] : indptr[index + 1]] = torch.arange(
                request * capacity + start,
                request * capacity + start + length,
                device=device,
            )


backend.kv_index_translator = Translator()
updater = object.__new__(FlashInferIndicesUpdaterPrefill)
updater.attn_backend = backend
updater.num_qo_heads = updater.num_kv_heads = heads
updater.head_dim = dimension
updater.q_data_type = updater.data_type = torch.bfloat16
updater.kv_indptr = [torch.zeros(3, dtype=torch.int32, device=device)]
updater.qo_indptr = [torch.zeros(3, dtype=torch.int32, device=device)]
updater.kv_last_page_len = torch.ones(2, dtype=torch.int32, device=device)
updater.prefill_wrapper_ragged = backend.prefill_wrapper_ragged
backend.indices_updater_prefill = updater
layer = NS(
    tp_q_head_num=heads,
    tp_k_head_num=heads,
    tp_v_head_num=heads,
    head_dim=dimension,
    scaling=dimension**-0.5,
    logit_cap=0.0,
    layer_id=0,
    k_scale_float=None,
    v_scale_float=None,
    is_cross_attention=False,
    attn_type=AttentionType.ENCODER_ONLY,
    sliding_window_size=-1,
)


def probe(prefix, pad):
    cached_key = torch.randn(
        2 * capacity, 1, heads, dimension, dtype=torch.bfloat16, device=device
    )
    cached_value = torch.randn_like(cached_key)
    cache_before_key, cache_before_value = cached_key.clone(), cached_value.clone()
    writes = []

    def write(layer, location, key, value, *scales):
        writes.append(location.loc.clone())
        cached_key[location.loc, 0] = key.view(-1, heads, dimension)
        cached_value[location.loc, 0] = value.view(-1, heads, dimension)

    backend.token_to_kv_pool = NS(
        get_kv_buffer=lambda index: (cached_key, cached_value), set_kv_buffer=write
    )
    query = torch.randn(
        2 * block_size, heads, dimension, dtype=torch.bfloat16, device=device
    )
    key, value = torch.randn_like(query), torch.randn_like(query)
    cache_locations = torch.cat(
        [
            torch.arange(i * capacity + p, i * capacity + p + block_size, device=device)
            for i, p in enumerate(prefix)
        ]
    )
    batch = NS(
        dllm_left_pad_lens_cpu=pad,
        extend_prefix_lens_cpu=prefix,
        extend_seq_lens_cpu=[block_size, block_size],
        forward_mode=NS(is_dllm_extend=lambda: True),
        seq_lens=torch.tensor(
            [p + block_size for p in prefix], dtype=torch.int32, device=device
        ),
        extend_prefix_lens=torch.tensor(prefix, dtype=torch.int32, device=device),
        req_pool_indices=torch.arange(2, dtype=torch.int32, device=device),
        out_cache_loc=cache_locations,
    )
    backend.init_forward_metadata(batch)
    output = backend.forward_extend(
        query.flatten(1), key.flatten(1), value.flatten(1), layer, batch
    ).view_as(query)
    references = []
    for row, (prefix_length, pad_length) in enumerate(zip(prefix, pad)):
        local_pad = min(max(pad_length - prefix_length, 0), block_size)
        cached_start = row * capacity + min(pad_length, prefix_length)
        cached_end = row * capacity + prefix_length
        keys = torch.cat(
            [
                cache_before_key[cached_start:cached_end, 0],
                key[row * block_size : (row + 1) * block_size],
            ]
        )
        values = torch.cat(
            [
                cache_before_value[cached_start:cached_end, 0],
                value[row * block_size : (row + 1) * block_size],
            ]
        )
        allowed = torch.ones(block_size, keys.shape[0], dtype=torch.bool, device=device)
        cached_count = cached_end - cached_start
        allowed[:, cached_count : cached_count + local_pad] = False
        for index in range(local_pad):
            allowed[index, cached_count + index] = True
        scores = (
            torch.einsum(
                "qhd,khd->hqk",
                query[row * block_size : (row + 1) * block_size].float(),
                keys.float(),
            )
            * layer.scaling
        )
        scores.masked_fill_(~allowed.unsqueeze(0), -torch.inf)
        references.append(
            torch.einsum("hqk,khd->qhd", scores.softmax(-1), values.float())
        )
    reference = torch.cat(references)
    assert torch.isfinite(output).all(), (prefix, pad, "nonfinite output")
    torch.testing.assert_close(output.float(), reference, atol=0.025, rtol=0.025)
    assert len(writes) == 1
    torch.testing.assert_close(cached_key[cache_locations, 0], key)
    torch.testing.assert_close(cached_value[cache_locations, 0], value)
    print(
        {
            "prefix": prefix,
            "pads": pad,
            "local_mask": backend.cfg_local_left_pad_active,
            "max_abs_error": (output.float() - reference).abs().max().item(),
            "kv_writes": len(writes),
        }
    )


for prefix, pads in [
    ([32, 32], [0, 40]),
    ([64, 64], [0, 40]),
    ([32, 32], [0, 80]),
    ([0, 0], [0, 8]),
    ([96, 96], [0, 80]),
]:
    probe(prefix, pads)
