# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.dots_tts._vendored.side_runtime import (
    DotsTtsSideModel,
    DotsTtsSideRuntime,
)
from sglang_omni.models.dots_tts.native_adapter import DotsTTSNativeAdapter
from sglang_omni.models.dots_tts.payload_types import DotsTTSState


def test_side_runtime_loads_speech_data_uri() -> None:
    samples = (
        np.sin(np.linspace(0, 20 * np.pi, 4000)) * np.iinfo(np.int16).max
    ).astype("<i2")
    wav = io.BytesIO()
    with wave.open(wav, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8000)
        writer.writeframes(samples.tobytes())
    uri = "data:audio/wav;base64," + base64.b64encode(wav.getvalue()).decode()
    runtime = DotsTtsSideRuntime.__new__(DotsTtsSideRuntime)
    runtime.sample_rate = 8000

    audio = runtime._load_prompt_audio(uri)

    assert audio.ndim == 2
    assert audio.shape[0] == 1
    assert audio.dtype == torch.float32
    assert audio.numel() > 0


class FakeNativeModel:
    """Minimal model implementing the native preparation API the adapter requires."""

    AUDIO_SPAN_ID = 99
    prepare_request = DotsTtsSideModel.prepare_request
    _generation_kwargs = staticmethod(DotsTtsSideModel._generation_kwargs)

    def __init__(self) -> None:
        self.llm_config = SimpleNamespace(vocab_size=32000)
        self.core = SimpleNamespace(
            parameters=lambda: iter([torch.zeros(1)]),
            audio_span_token_ids=[self.AUDIO_SPAN_ID],
        )

    def _prepare_prompt_conditioning(
        self, prompt_audio, *, use_prompt_prefill, speaker_scale
    ):
        del prompt_audio, use_prompt_prefill, speaker_scale
        return SimpleNamespace(prompt_patches=None, prompt_latents=None, g_cond=None)

    def _find_audio_span_positions(self, schedule, *, audio_placeholder_ids):
        flat = schedule.reshape(-1)
        mask = torch.zeros_like(flat, dtype=torch.bool)
        for placeholder in audio_placeholder_ids:
            mask |= flat == placeholder
        return mask.nonzero(as_tuple=False).reshape(-1)

    def _allocate_generate_state(self, *, max_audio_patch_count, device, dtype):
        del max_audio_patch_count, dtype
        return SimpleNamespace(fm_sequence=torch.zeros(1, device=device))

    def _prefill_prompt_latents(self, prompt_latents, *, state):
        del prompt_latents, state
        return None

    def _build_prefill_inputs_embeds(
        self,
        generation_schedule,
        *,
        prompt_patch_embeddings,
        prompt_span_positions,
    ):
        del prompt_patch_embeddings
        return torch.zeros(
            generation_schedule.size(0),
            generation_schedule.size(1),
            4,
            dtype=torch.bfloat16,
        )

    def _locate_prefill_boundary(self, *, span_positions, prompt_patch_count):
        # First audio span beyond the prompt patches starts generation.
        prefill_end = int(span_positions[prompt_patch_count].item())
        prompt_span_positions = span_positions[:prompt_patch_count]
        return prefill_end, prompt_span_positions


class FakeRuntime:
    precision = "bfloat16"
    max_generate_length = 0

    def __init__(self) -> None:
        self.model = FakeNativeModel()
        self.prepared_kwargs = None

    def _prepare_inputs(self, **kwargs):
        self.prepared_kwargs = kwargs
        return {
            "generation_schedule": torch.tensor([[1, 2, 99, 99]]),
        }


def test_adapter_prepares_native_state_from_state() -> None:
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
    # Native prep derives input_ids from the schedule up to the first decode span.
    assert prepared.input_ids.tolist() == [[1, 2]]
    assert prepared.generation_schedule.tolist() == [[1, 2, 99, 99]]
    assert prepared.audio_span_positions.tolist() == [2, 3]
    assert prepared.prefill_end == 2
    assert prepared.fm_state is not None
    assert prepared.audio_placeholder_ids == {FakeNativeModel.AUDIO_SPAN_ID}
    assert adapter.llm_vocab_size == 32000


def test_adapter_requires_generation_schedule() -> None:
    class ScheduleLessRuntime(FakeRuntime):
        def _prepare_inputs(self, **kwargs):
            del kwargs
            return {}

    with pytest.raises(RuntimeError, match="generation_schedule"):
        DotsTTSNativeAdapter(ScheduleLessRuntime()).prepare_inputs(
            DotsTTSState(text="Hello.")
        )


def test_adapter_casts_prompt_latents_to_generate_state_dtype() -> None:
    class PromptPrefillModel(FakeNativeModel):
        def __init__(self) -> None:
            super().__init__()
            self.prefill_dtype = None

        def _prepare_prompt_conditioning(
            self, prompt_audio, *, use_prompt_prefill, speaker_scale
        ):
            del prompt_audio, use_prompt_prefill, speaker_scale
            return SimpleNamespace(
                prompt_patches=torch.zeros(1, 1, 4, dtype=torch.float32),
                prompt_latents=torch.ones(1, 4, 128, dtype=torch.float32),
                g_cond=None,
            )

        def _allocate_generate_state(self, *, max_audio_patch_count, device, dtype):
            del max_audio_patch_count
            return SimpleNamespace(
                fm_sequence=torch.zeros(1, device=device, dtype=dtype)
            )

        def _prefill_prompt_latents(self, prompt_latents, *, state):
            del state
            self.prefill_dtype = prompt_latents.dtype
            return torch.zeros(1, 1, 4, dtype=prompt_latents.dtype)

    class PromptPrefillRuntime(FakeRuntime):
        def __init__(self) -> None:
            self.model = PromptPrefillModel()
            self.prepared_kwargs = None

    runtime = PromptPrefillRuntime()

    DotsTTSNativeAdapter(runtime).prepare_inputs(
        DotsTTSState(text="Hello.", prompt_audio_path="ref.wav", prompt_text="Hi.")
    )

    assert runtime.model.prefill_dtype == torch.bfloat16


def test_adapter_rejects_request_longer_than_runtime_schedule() -> None:
    class ShortScheduleRuntime(FakeRuntime):
        max_generate_length = 4

        def _prepare_inputs(self, **kwargs):
            del kwargs
            return {
                "generation_schedule": torch.tensor([[1, 2, 99, 99, 99, 99]]),
                "audio_span_positions": torch.tensor([2, 3, 4, 5]),
            }

    adapter = DotsTTSNativeAdapter(ShortScheduleRuntime())

    with pytest.raises(ValueError, match="max_generate_length"):
        adapter.prepare_inputs(DotsTTSState(text="Hello.", max_generate_length=8))
