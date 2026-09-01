# SPDX-License-Identifier: Apache-2.0
"""The talker output cap follows the thinker text length."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
pytest.importorskip("xxhash")

from sglang_omni.models.qwen3_omni.request_builders import (
    TALKER_MAX_NEW_TOKENS_BASE,
    TALKER_MAX_NEW_TOKENS_DEFAULT,
    TALKER_MAX_NEW_TOKENS_PER_TEXT_TOKEN,
    _build_talker_request_data,
    talker_max_new_tokens_bound,
)

HIDDEN = 8
PROMPT_LEN = 5


def test_bound_grows_with_the_text_and_stops_at_the_default() -> None:
    assert talker_max_new_tokens_bound(0) == TALKER_MAX_NEW_TOKENS_BASE
    assert (
        talker_max_new_tokens_bound(14)
        == TALKER_MAX_NEW_TOKENS_BASE + 14 * TALKER_MAX_NEW_TOKENS_PER_TEXT_TOKEN
    )
    assert talker_max_new_tokens_bound(1000) == TALKER_MAX_NEW_TOKENS_DEFAULT
    assert talker_max_new_tokens_bound(-3) == TALKER_MAX_NEW_TOKENS_BASE


class _FakePrefillBuilder:
    def build_prompt_prefill(self, payload, thinker_chunks, *, thinker_done):
        return {
            "input_embeds": torch.zeros(PROMPT_LEN, HIDDEN),
            "input_ids": torch.arange(PROMPT_LEN, dtype=torch.long),
            "pending_text_queue": None,
            "tts_pad_embed": torch.zeros(HIDDEN),
            "tts_eos_embed": torch.zeros(HIDDEN),
            "prompt_model_inputs": {},
        }


def _resolve_sampling_config(params: dict) -> dict:
    return {
        "max_new_tokens": int(
            params.get("talker_max_new_tokens", TALKER_MAX_NEW_TOKENS_DEFAULT)
        ),
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "codec_eos_id": 2150,
        "suppress_tokens": [],
        "seed": None,
    }


def _payload(params: dict, *, chunks: int, thinker_done: bool):
    chunk = SimpleNamespace(metadata={"token_id": 7}, data=None)
    return SimpleNamespace(
        request_id="talker-cap-test",
        request=SimpleNamespace(params=params),
        data={},
        prefetched_chunks=[chunk] * chunks,
        prefetched_stream_done=thinker_done,
    )


def _build(
    monkeypatch: pytest.MonkeyPatch, params: dict, *, chunks: int, thinker_done: bool
):
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    return _build_talker_request_data(
        _payload(params, chunks=chunks, thinker_done=thinker_done),
        prefill_builder=_FakePrefillBuilder(),
        tokenizer=object(),
        codec_vocab_size=3072,
        codec_bos_id=2149,
        audio_token_id=None,
        image_token_id=None,
        video_token_id=None,
        thinker_config=None,
        resolve_sampling_config=_resolve_sampling_config,
    )


def test_cap_follows_the_prefetched_text_when_the_thinker_is_done(monkeypatch) -> None:
    data = _build(monkeypatch, {}, chunks=14, thinker_done=True)
    expected = talker_max_new_tokens_bound(14)
    assert data.max_new_tokens == expected
    assert data.req.sampling_params.max_new_tokens == expected


def test_explicit_talker_max_new_tokens_is_kept(monkeypatch) -> None:
    data = _build(
        monkeypatch, {"talker_max_new_tokens": 4096}, chunks=14, thinker_done=True
    )
    assert data.max_new_tokens == 4096
    assert data.req.sampling_params.max_new_tokens == 4096


def test_partial_start_keeps_the_default_cap(monkeypatch) -> None:
    data = _build(monkeypatch, {}, chunks=3, thinker_done=False)
    assert data.max_new_tokens == TALKER_MAX_NEW_TOKENS_DEFAULT
