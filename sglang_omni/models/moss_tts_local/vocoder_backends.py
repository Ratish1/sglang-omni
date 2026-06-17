# SPDX-License-Identifier: Apache-2.0
"""Non-streaming MOSS-TTS Local vocoder decode backends."""

from __future__ import annotations

import os
from typing import Any, Callable, Protocol

import torch

from sglang_omni.profiler.ranges import torch_profile_range

NONSTREAM_VOCODER_BACKEND_ENV = "MOSS_TTS_LOCAL_NONSTREAM_VOCODER_BACKEND"
_SUPPORTED_BACKENDS = frozenset({"processor", "session", "sglang"})


class NonStreamingVocoderBackend(Protocol):
    """Decode generated code rows into CPU fp32 waveforms."""

    name: str

    def decode_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Decode ``[T, >=n_vq]`` rows to ``[channels, samples]`` CPU fp32."""
        ...


def resolve_nonstream_vocoder_backend(value: str | None = None) -> str:
    raw = value
    if raw is None:
        raw = os.getenv(NONSTREAM_VOCODER_BACKEND_ENV, "processor")
    name = str(raw).strip().lower()
    if not name:
        name = "processor"
    if name not in _SUPPORTED_BACKENDS:
        allowed = ", ".join(sorted(_SUPPORTED_BACKENDS))
        raise ValueError(
            f"Unsupported MOSS-TTS Local non-streaming vocoder backend {raw!r}; "
            f"expected one of: {allowed}"
        )
    return name


class ProcessorDecodeBackend:
    """Current ``processor.decode_audio_codes`` non-streaming decode path."""

    name = "processor"

    def __init__(self, processor: Any) -> None:
        self._processor = processor

    def decode_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        with torch_profile_range("moss.vocoder.backend.processor_decode"):
            return [
                torch.as_tensor(wav).detach().to("cpu", torch.float32).contiguous()
                for wav in self._processor.decode_audio_codes(codes_list)
            ]


class SessionDecodeBackend:
    """Offline decode through the persistent codec streaming session lanes."""

    name = "session"

    def __init__(
        self,
        *,
        session_provider: Callable[[], Any],
        state_lock: Any,
        n_vq: int,
        max_step_frames: int,
    ) -> None:
        self._session_provider = session_provider
        self._state_lock = state_lock
        self._n_vq = int(n_vq)
        self._max_step_frames = int(max_step_frames)

    def decode_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        with torch_profile_range("moss.vocoder.backend.session_decode"):
            channels_first = [
                codes[:, : self._n_vq].transpose(0, 1).contiguous()
                for codes in codes_list
            ]
            with self._state_lock:
                wavs = self._session_provider().decode_offline(
                    channels_first, max_step_frames=self._max_step_frames
                )
            return [wav.detach().to("cpu", torch.float32).contiguous() for wav in wavs]


class SGLangCodecDecodeBackend:
    """Future SGLang-native codec decoder backend."""

    name = "sglang"

    def decode_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        del codes_list
        raise NotImplementedError(
            "MOSS-TTS Local native SGLang codec decoder execution is not "
            "implemented yet. The 'sglang' non-streaming vocoder backend is "
            "accepted only as an explicit fail-closed selection while the "
            "codec decoder contract is validated."
        )


__all__ = [
    "NONSTREAM_VOCODER_BACKEND_ENV",
    "NonStreamingVocoderBackend",
    "ProcessorDecodeBackend",
    "SGLangCodecDecodeBackend",
    "SessionDecodeBackend",
    "resolve_nonstream_vocoder_backend",
]
