"""Isolate native attention operations on captured Talker conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Unpack

import torch
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorAttention,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextAttention,
    eager_attention_forward,
)

from sglang_omni.utils.predictor_layers import decode_attention

ATTENTION_IMPLEMENTATION = "diagnostic_qwen3_omni_fa3"
PREDICTOR_ATTENTION_IMPLEMENTATION = "diagnostic_qwen3_omni_predictor_attention"


def validate_causal_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[int, int]:
    assert query.shape[0] == key.shape[0] == value.shape[0] == 1
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    query_tokens = query.shape[-2]
    key_tokens = key.shape[-2]
    assert key_tokens >= query_tokens and key.shape == value.shape
    if attention_mask is not None:
        expected_allowed = torch.arange(key_tokens, device=query.device)[None] <= (
            torch.arange(query_tokens, device=query.device)[:, None]
            + key_tokens
            - query_tokens
        )
        assert attention_mask.shape == (1, 1, query_tokens, key_tokens)
        assert torch.equal(attention_mask[0, 0] == 0, expected_allowed)
        assert (
            attention_mask[0, 0][~expected_allowed] <= torch.finfo(query.dtype).min
        ).all()
    else:
        pass
    return query_tokens, key_tokens


@dataclass(kw_only=True)
class NativePredictorAttention:
    reference: Qwen3OmniMoeTalkerForConditionalGeneration
    calls: int = 0
    maximum_absolute_error: float = 0.0

    def install(self) -> None:
        assert (
            self.reference.model.config
            is not self.reference.code_predictor.model.config
        )
        ALL_ATTENTION_FUNCTIONS.register(
            PREDICTOR_ATTENTION_IMPLEMENTATION, self.forward
        )
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            PREDICTOR_ATTENTION_IMPLEMENTATION, ALL_MASK_ATTENTION_FUNCTIONS["eager"]
        )

    def enable(self) -> None:
        assert self.reference.model.config._attn_implementation == "eager"
        configuration = self.reference.code_predictor.model.config
        assert all(
            layer.self_attn.config is configuration
            for layer in self.reference.code_predictor.model.layers
        )
        configuration._attn_implementation = PREDICTOR_ATTENTION_IMPLEMENTATION
        self.calls = 0
        self.maximum_absolute_error = 0.0

    def forward(
        self,
        module: Qwen3OmniMoeTalkerCodePredictorAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        sliding_window: int | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, None]:
        assert dropout == 0.0 and sliding_window is None and not module.training
        assert scaling == module.scaling and module.is_causal
        query_tokens, key_tokens = validate_causal_inputs(
            query, key, value, attention_mask
        )
        if attention_mask is None:
            assert query_tokens == 1
        else:
            pass
        slot_count = self.reference.config.num_code_groups + 1
        assert query_tokens in (1, 2) and key_tokens <= slot_count
        expected, _ = eager_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
            sliding_window=sliding_window,
            **kwargs,
        )
        flattened_query = query.transpose(1, 2).contiguous().reshape(query_tokens, -1)
        keys = torch.zeros(
            (1, slot_count, key.shape[1], key.shape[-1]),
            dtype=key.dtype,
            device=key.device,
        )
        values = torch.zeros_like(keys)
        keys[:, :key_tokens].copy_(key.transpose(1, 2))
        values[:, :key_tokens].copy_(value.transpose(1, 2))
        output = torch.empty_like(flattened_query)
        decode_attention(
            q=flattened_query,
            tokens_per_row=query_tokens,
            k_cache=keys,
            v_cache=values,
            end=key_tokens,
            attention_output=output,
        )
        output = output.view(1, query_tokens, query.shape[1], query.shape[-1])
        assert output.shape == expected.shape and output.isfinite().all()
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float((output.float() - expected.float()).abs().max()),
        )
        self.calls += 1
        return output, None


@dataclass(kw_only=True)
class NativeBackboneAttention:
    reference: Qwen3OmniMoeTalkerForConditionalGeneration
    decode_page_table_tokens: int | None = None
    calls: int = 0

    def install(self) -> None:
        ALL_ATTENTION_FUNCTIONS.register(ATTENTION_IMPLEMENTATION, self.forward)
        ALL_MASK_ATTENTION_FUNCTIONS.register(
            ATTENTION_IMPLEMENTATION, ALL_MASK_ATTENTION_FUNCTIONS["eager"]
        )
        assert all(
            layer.self_attn.config is not self.reference.model.config
            for layer in self.reference.code_predictor.model.layers
        )

    def set_enabled(self, enabled: bool) -> None:
        implementation = ATTENTION_IMPLEMENTATION if enabled else "eager"
        self.reference.model.config._attn_implementation = implementation
        for layer in self.reference.model.layers:
            layer.self_attn.config._attn_implementation = implementation
        assert all(
            layer.self_attn.config._attn_implementation == "eager"
            for layer in self.reference.code_predictor.model.layers
        )
        self.calls = 0

    def forward(
        self,
        module: Qwen3OmniMoeThinkerTextAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        sliding_window: int | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, None]:
        assert dropout == 0.0 and sliding_window is None and not module.training
        assert scaling == module.scaling and module.is_causal
        query_tokens, key_tokens = validate_causal_inputs(
            query, key, value, attention_mask
        )
        cache_lengths = torch.tensor(
            [key_tokens], dtype=torch.int32, device=query.device
        )
        if self.decode_page_table_tokens is not None and query_tokens == 1:
            assert key_tokens <= self.decode_page_table_tokens
            page_table = torch.zeros(
                (1, self.decode_page_table_tokens),
                dtype=torch.int32,
                device=query.device,
            )
            page_table[0, :key_tokens] = torch.arange(
                key_tokens, dtype=torch.int32, device=query.device
            )
            output = flash_attn_with_kvcache(
                query.transpose(1, 2).contiguous().squeeze(0),
                key.transpose(1, 2).contiguous().squeeze(0).unsqueeze(1),
                value.transpose(1, 2).contiguous().squeeze(0).unsqueeze(1),
                page_table=page_table,
                cache_seqlens=cache_lengths,
                cu_seqlens_q=torch.arange(2, dtype=torch.int32, device=query.device),
                max_seqlen_q=1,
                softmax_scale=scaling,
                causal=True,
                ver=3,
            ).unsqueeze(0)
        else:
            output = flash_attn_with_kvcache(
                query.transpose(1, 2).contiguous(),
                key.transpose(1, 2).contiguous(),
                value.transpose(1, 2).contiguous(),
                cache_seqlens=cache_lengths,
                softmax_scale=scaling,
                causal=True,
                ver=3,
            )
        assert output.shape == (1, query_tokens, query.shape[1], query.shape[-1])
        assert torch.isfinite(output).all()
        self.calls += 1
        return output, None
