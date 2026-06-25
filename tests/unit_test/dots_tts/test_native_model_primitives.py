# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("torchdiffeq")

from sglang_omni.models.dots_tts.native.models.dots_tts.model import DotsTtsModel
from sglang_omni.models.dots_tts.native.side_runtime import DotsTtsSideModel
from sglang_omni.models.dots_tts.payload_types import DotsTTSState


def test_side_model_prepare_request_builds_serving_inputs() -> None:
    class FakeCore(nn.Module):
        audio_span_token_ids = [99]

        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

    model = DotsTtsSideModel.__new__(DotsTtsSideModel)
    nn.Module.__init__(model)
    model.core = FakeCore()
    model._prepare_prompt_conditioning = lambda *args, **kwargs: SimpleNamespace(
        prompt_patches=None,
        prompt_latents=None,
        g_cond=None,
    )
    model._find_audio_span_positions = lambda schedule, *, audio_placeholder_ids: (
        schedule.reshape(-1).eq(next(iter(audio_placeholder_ids))).nonzero().reshape(-1)
    )
    model._allocate_generate_state = lambda *, max_audio_patch_count, device, dtype: (
        SimpleNamespace(fm_sequence=torch.zeros(1, device=device, dtype=dtype))
    )
    model._prefill_prompt_latents = lambda prompt_latents, *, state: None
    model._locate_prefill_boundary = (
        lambda *, span_positions, prompt_patch_count: (
            int(span_positions[prompt_patch_count].item()),
            span_positions[:prompt_patch_count],
        )
    )
    model._build_prefill_inputs_embeds = lambda *args, **kwargs: None

    raw_inputs = {"generation_schedule": torch.tensor([[1, 2, 99, 99]])}
    prepared = model.prepare_request(
        raw_inputs,
        DotsTTSState(text="Hello."),
        generation_schedule=raw_inputs["generation_schedule"],
        precision="bfloat16",
    )

    assert prepared.input_ids.tolist() == [[1, 2]]
    assert prepared.generation_schedule.tolist() == [[1, 2, 99, 99]]
    assert prepared.audio_span_positions.tolist() == [2, 3]
    assert prepared.prefill_end == 2
    assert prepared.audio_placeholder_ids == {99}
    assert prepared.fm_state is raw_inputs["fm_state"]
    assert prepared.generation_kwargs["device"] == torch.device("cpu")
    assert prepared.generation_kwargs["ode_method"] == "euler"


def test_side_model_decode_audio_step_returns_payload_and_feedback() -> None:
    class FakeIOHelper:
        def denormalize(self, latent_patch):
            return latent_patch + 2

    class FakeCore(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))
            self.io_helper = FakeIOHelper()

    model = DotsTtsSideModel.__new__(DotsTtsSideModel)
    nn.Module.__init__(model)
    model.core = FakeCore()
    fm_state = {"history": []}
    calls = []
    model._append_hidden_chunk = lambda state, hidden: calls.append(
        ("append_hidden", state, hidden)
    )
    model._decode_next_audio = lambda **kwargs: torch.ones(1, 4, 8)
    model._encode_audio_patch_feedback = lambda state, *, audio_patch: (
        audio_patch.mean(dim=1, keepdim=True)
    )
    model._should_stop_after_current_audio = lambda state, *, eos_threshold: True

    hidden_state = torch.zeros(1, 1, 4)
    result = model.decode_audio_step(
        fm_state=fm_state,
        generation_kwargs={
            "device": torch.device("cpu"),
            "g_cond": None,
            "ode_method": "euler",
            "num_steps": 2,
            "guidance_scale": 1.2,
            "eos_threshold": 0.8,
        },
        hidden_state=hidden_state,
        precision="float32",
    )

    assert len(calls) == 1
    assert calls[0][0] == "append_hidden"
    assert calls[0][1] is fm_state
    assert calls[0][2] is hidden_state
    assert torch.equal(result.latent_patch, torch.full((1, 4, 8), 3.0))
    assert torch.equal(result.feedback_embedding, torch.ones(1, 1, 8))
    assert torch.equal(result.eos_score, torch.tensor([1.0]))


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


def test_side_model_warmup_only_runs_side_compile_buckets() -> None:
    class FakeCore(nn.Module):
        audio_gen_span_id = 42

        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

    model = DotsTtsSideModel.__new__(DotsTtsSideModel)
    nn.Module.__init__(model)
    model.core = FakeCore()
    model.audio_gen_start_id = 41
    calls = []

    model._warmup_fm_bucket = lambda **kwargs: calls.append(
        ("fm", kwargs["max_audio_patch_count"], kwargs["precision"])
    )
    model._warmup_patch_encoder_bucket = lambda **kwargs: calls.append(
        ("patch_encoder", kwargs["max_audio_patch_count"], kwargs["precision"])
    )

    model.run_warmup(max_generate_length=32, precision="float32")

    assert calls == [
        ("fm", 32, "float32"),
        ("patch_encoder", 32, "float32"),
    ]


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
    model._get_compiled_model = (
        lambda *args, **kwargs: model.core.patch_encoder.decode_patch
    )
    model._patch_encoder_compile_signature = lambda patch_encoder_state: (
        8,
        torch.float32,
    )

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


def test_prepare_prompt_conditioning_casts_speaker_embedding_to_projection_dtype() -> (
    None
):
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
