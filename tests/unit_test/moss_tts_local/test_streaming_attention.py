# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.moss_tts_local import streaming_attention
from sglang_omni.models.moss_tts_local.streaming_attention import (
    _ORIGINAL_STREAMING_SDPA_ATTR,
    _derive_cache_leftpad,
    moss_decoder_attention_modules,
    moss_local_window_size,
    patch_codec_streaming_attention,
)


class _FakeAttention(nn.Module):
    attention_implementation = "flash_attention_2"

    def __init__(self) -> None:
        super().__init__()
        self.context = 4
        self.rope = None
        self.embed_dim = 8

    def _forward_streaming_sdpa(self, x, state):
        return ("sdpa", x, state)

    def _update_streaming_cache(self, *args):
        return None

    def _get_backend_check_dtype(self, x: torch.Tensor) -> torch.dtype:
        return x.dtype


class _FakeStage(nn.Module):
    def __init__(self, attention: nn.Module) -> None:
        super().__init__()
        self.attention = attention


class _FakeCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder_attention = _FakeAttention()
        self.encoder_attention = _FakeAttention()
        self.decoder = nn.ModuleList([_FakeStage(self.decoder_attention)])
        self.encoder = nn.ModuleList([_FakeStage(self.encoder_attention)])


def test_moss_local_window_size_matches_context_exclusive_mask() -> None:
    assert moss_local_window_size(None) == (-1, -1)
    assert moss_local_window_size(1) == (0, 0)
    assert moss_local_window_size(4) == (3, 0)
    assert moss_local_window_size(4, causal=False) == (-1, -1)


def test_derive_cache_leftpad_from_right_aligned_cached_positions() -> None:
    pos_k = torch.tensor(
        [
            [-1, -1, -1, -1, 0, 1],
            [-1, -1, 3, 4, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.long,
    )

    leftpad = _derive_cache_leftpad(pos_k, context=4)

    assert torch.equal(leftpad, torch.tensor([4, 2, 0], dtype=torch.int32))


def test_moss_decoder_attention_modules_only_returns_decoder_attention() -> None:
    codec = _FakeCodec()

    modules = moss_decoder_attention_modules(codec)

    assert modules == [codec.decoder_attention]


def test_patch_codec_streaming_attention_is_decoder_only_and_idempotent(
    monkeypatch,
) -> None:
    codec = _FakeCodec()
    monkeypatch.setattr(streaming_attention, "flash_attn_with_kvcache", object())

    first = patch_codec_streaming_attention(codec)
    second = patch_codec_streaming_attention(codec)

    assert first == 1
    assert second == 0
    assert hasattr(codec.decoder_attention, _ORIGINAL_STREAMING_SDPA_ATTR)
    assert not hasattr(codec.encoder_attention, _ORIGINAL_STREAMING_SDPA_ATTR)
    assert codec.decoder_attention._forward_streaming_sdpa.__name__ == (
        streaming_attention._forward_streaming_sglang.__name__
    )


def test_patched_streaming_attention_keeps_cpu_inputs_on_original_path(
    monkeypatch,
) -> None:
    codec = _FakeCodec()
    monkeypatch.setattr(streaming_attention, "flash_attn_with_kvcache", object())
    patch_codec_streaming_attention(codec)
    x = torch.randn(1, 2, 8)
    state = SimpleNamespace(offset=torch.zeros(1, dtype=torch.long))

    out = codec.decoder_attention._forward_streaming_sdpa(x, state)

    assert out[0] == "sdpa"
    assert out[1] is x
    assert out[2] is state
