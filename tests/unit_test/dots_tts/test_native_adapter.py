# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.dots_tts.native_adapter import DotsTTSNativeAdapter
from sglang_omni.models.dots_tts.payload_types import DotsTTSState


class FakeRuntime:
    precision = "bfloat16"

    def __init__(self) -> None:
        self.model = SimpleNamespace()
        self.prepared_kwargs = None

    def _prepare_inputs(self, **kwargs):
        self.prepared_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "generation_schedule": torch.tensor([0, 1, 2]),
            "audio_span_positions": torch.tensor([2]),
        }


def test_adapter_prepares_inputs_from_state() -> None:
    runtime = FakeRuntime()
    adapter = DotsTTSNativeAdapter(runtime)
    state = DotsTTSState(
        text="Hello.",
        prompt_audio_path="ref.wav",
        prompt_text="Hi.",
        template_name="tts",
        language="en",
        normalize_text=False,
    )

    prepared = adapter.prepare_inputs(state)

    assert runtime.prepared_kwargs == {
        "text": "Hello.",
        "prompt_audio_path": "ref.wav",
        "prompt_text": "Hi.",
        "template_name": "tts",
        "language": "en",
        "normalize_text": False,
    }
    assert prepared.input_ids.tolist() == [[1, 2, 3]]
    assert prepared.generation_schedule.tolist() == [0, 1, 2]
    assert prepared.audio_span_positions.tolist() == [2]


def test_adapter_uses_generation_schedule_as_input_ids_when_missing() -> None:
    class ScheduleOnlyRuntime(FakeRuntime):
        def _prepare_inputs(self, **kwargs):
            del kwargs
            return {
                "generation_schedule": torch.tensor([[5, 6, 7]]),
                "audio_span_positions": torch.tensor([2]),
            }

    prepared = DotsTTSNativeAdapter(ScheduleOnlyRuntime()).prepare_inputs(
        DotsTTSState(text="Hello.")
    )

    assert prepared.input_ids.tolist() == [[5, 6, 7]]
    assert prepared.generation_schedule.tolist() == [[5, 6, 7]]


class FakeDotsModel:
    def __init__(self) -> None:
        self.calls = []

    def _decode_next_audio(self, **kwargs):
        self.calls.append(kwargs)
        return torch.ones(1, 4, 128)

    def _encode_audio_patch_feedback(self, fm_state, *, audio_patch):
        assert fm_state == {"history": []}
        return audio_patch.mean(dim=1, keepdim=True)

    def _encode_audio_patch(self, latent_patch):
        return latent_patch.mean(dim=1, keepdim=True)

    def _predict_eos(self, hidden_state, latent_patch):
        del hidden_state, latent_patch
        return torch.tensor([0.8])

    def _should_stop_after_current_audio(self, fm_state, *, eos_threshold):
        assert fm_state == {"history": []}
        assert eos_threshold == 0.8
        return True


def test_adapter_generates_patch_feedback_and_eos() -> None:
    runtime = SimpleNamespace(model=FakeDotsModel(), precision="bfloat16")
    adapter = DotsTTSNativeAdapter(runtime)
    hidden_state = torch.ones(1, 1, 2048)

    result = adapter.generate_audio_step(
        hidden_state=hidden_state,
        fm_state={"history": []},
        generation_kwargs={
            "ode_method": "euler",
            "num_steps": 2,
            "guidance_scale": 1.2,
            "speaker_scale": 1.5,
            "device": torch.device("cpu"),
            "g_cond": None,
            "eos_threshold": 0.8,
        },
    )

    assert result.latent_patch.shape == (1, 4, 128)
    assert result.feedback_embedding.shape == (1, 1, 128)
    assert torch.allclose(result.eos_score, torch.tensor([1.0]))
