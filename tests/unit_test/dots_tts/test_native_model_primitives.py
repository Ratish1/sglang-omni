# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

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
