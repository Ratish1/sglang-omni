# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("torchdiffeq")

from sglang_omni.models.dots_tts.native.models.dots_tts.model import DotsTtsModel


def test_allocate_generate_state_uses_request_owned_fm_buffers() -> None:
    class FakeModel:
        _optimize_enabled = True
        _allocate_fm_state_buffers = DotsTtsModel._allocate_fm_state_buffers

        def __init__(self) -> None:
            self.core = type(
                "FakeCore",
                (),
                {
                    "hidden_patch_size": 1,
                    "latent_patch_size": 4,
                    "fm_hidden_size": 8,
                    "patch_encoder": None,
                },
            )()

        def _resolve_state_audio_patch_count(self, max_audio_patch_count: int) -> int:
            return 32

    model = FakeModel()

    first = DotsTtsModel._allocate_generate_state(
        model,
        max_audio_patch_count=12,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    second = DotsTtsModel._allocate_generate_state(
        model,
        max_audio_patch_count=12,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert first.fm_sequence is not second.fm_sequence
    assert first.fm_cfg_sequence is not second.fm_cfg_sequence
    assert first.fm_null_g_cond is not second.fm_null_g_cond
    assert first.fm_sequence.data_ptr() != second.fm_sequence.data_ptr()
    assert first.fm_cfg_sequence.data_ptr() != second.fm_cfg_sequence.data_ptr()

    first.fm_sequence.fill_(7.0)
    second.fm_sequence.zero_()

    assert torch.all(first.fm_sequence == 7.0)
    assert torch.all(second.fm_sequence == 0.0)


def test_encode_audio_patch_feedback_does_not_call_llm() -> None:
    model = DotsTtsModel.__new__(DotsTtsModel)
    state = SimpleNamespace(
        patch_encoder_state=SimpleNamespace(
            seq_len=3,
            conv_tail=torch.zeros(1, 2),
            layer_caches=[(torch.zeros(1, 1, 8, 2), torch.zeros(1, 1, 8, 2))],
        ),
        fm_sequence=torch.zeros(1, 16, 4),
    )
    calls = []

    class FakeIOHelper:
        def denormalize(self, audio_patch):
            calls.append(("denormalize", audio_patch.shape))
            return audio_patch + 1

    class FakePatchEncoder:
        out_ds_rate = 2

        def decode_patch(self, audio_patch, conv_tail, layer_caches, patch_positions):
            calls.append(
                (
                    "decode_patch",
                    audio_patch.clone(),
                    conv_tail.clone(),
                    layer_caches,
                    patch_positions.clone(),
                )
            )
            return torch.full((1, 2, 6), 3.0), torch.full_like(conv_tail, 4.0)

    class FakeCore:
        io_helper = FakeIOHelper()
        patch_encoder = FakePatchEncoder()

        def step_llm(self, **kwargs):
            calls.append(("step_llm", kwargs))
            raise AssertionError("SGLang feedback primitive must not call dots LLM")

    model.core = FakeCore()
    model._append_history_chunk = lambda state_arg, patch: calls.append(
        ("append_history", patch.shape)
    )
    model._ensure_patch_encoder_state_capacity = lambda *args, **kwargs: calls.append(
        ("ensure_capacity", kwargs["required_seq_len"])
    )
    model._get_compiled_model = lambda *args, **kwargs: model.core.patch_encoder.decode_patch
    model._patch_encoder_compile_signature = lambda patch_encoder_state: (8, torch.float32)

    feedback = model._encode_audio_patch_feedback(
        state,
        audio_patch=torch.ones(1, 4, 128),
    )

    assert torch.equal(feedback, torch.full((1, 2, 6), 3.0))
    assert torch.equal(state.patch_encoder_state.conv_tail, torch.full((1, 2), 4.0))
    assert state.patch_encoder_state.seq_len == 5
    assert [call[0] for call in calls] == [
        "denormalize",
        "append_history",
        "ensure_capacity",
        "decode_patch",
    ]
    assert torch.equal(calls[-1][4], torch.tensor([3, 4]))


def test_prepare_prompt_conditioning_casts_speaker_embedding_to_projection_dtype() -> None:
    class FakeCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
            self.xvec_proj = nn.Linear(4, 4).to(dtype=torch.bfloat16)

    class FakeSpeakerEncoder(nn.Module):
        sample_rate = 48000
        max_audio_seconds = 30

        def forward(self, prompt_audio):
            assert prompt_audio.dtype == torch.float32
            return torch.ones(1, 4, dtype=torch.float32)

    model = DotsTtsModel.__new__(DotsTtsModel)
    nn.Module.__init__(model)
    model.core = FakeCore()
    model.vocoder = nn.Identity()
    model.xvector_extractor = FakeSpeakerEncoder()
    model._prompt_feature_cache = {}
    model._prepare_prompt_audio_for_conditioning = lambda prompt_audio: (
        prompt_audio,
        "cache-key",
    )
    model._get_prompt_feature_cache_entry = lambda cache_key: None
    model._store_prompt_feature_cache_entry = lambda cache_key, entry: None
    model._get_compiled_model = lambda name, module: module

    conditioning = DotsTtsModel._prepare_prompt_conditioning(
        model,
        torch.zeros(16, dtype=torch.float32),
        use_prompt_prefill=False,
        speaker_scale=1.5,
    )

    assert conditioning.g_cond.dtype == torch.bfloat16
