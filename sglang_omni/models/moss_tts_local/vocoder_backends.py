# SPDX-License-Identifier: Apache-2.0
"""Non-streaming MOSS-TTS Local vocoder decode backends."""

from __future__ import annotations

import inspect
import os
import threading
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Callable, Iterator, Protocol

import torch

from sglang_omni.profiler.ranges import torch_profile_range

NONSTREAM_VOCODER_BACKEND_ENV = "MOSS_TTS_LOCAL_NONSTREAM_VOCODER_BACKEND"
_SUPPORTED_BACKENDS = frozenset({"processor", "session", "sglang"})
_MOSS_FLASH_ATTN_SYMBOL = "flash_attn_varlen_func"
_MISSING = object()
_FLASH_ATTENTION_PATCH_LOCK = threading.RLock()


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


def _load_sglang_flash_attn_varlen_func() -> Callable[..., Any]:
    try:
        from sglang.jit_kernel.flash_attention import flash_attn_varlen_func
    except ImportError as exc:
        raise RuntimeError(
            "MOSS-TTS Local sglang non-streaming vocoder backend requires "
            "sglang.jit_kernel.flash_attention.flash_attn_varlen_func"
        ) from exc
    return flash_attn_varlen_func


def _audio_tokenizer_module(processor: Any) -> ModuleType:
    codec = getattr(processor, "audio_tokenizer", None)
    if codec is None:
        raise RuntimeError(
            "MOSS-TTS Local sglang non-streaming vocoder backend requires "
            "processor.audio_tokenizer"
        )
    module = inspect.getmodule(type(codec))
    if module is None:
        raise RuntimeError(
            "MOSS-TTS Local sglang non-streaming vocoder backend could not "
            f"resolve remote module for {type(codec).__module__}.{type(codec).__name__}"
        )
    return module


def _check_patchable_remote_flash_attention(module: ModuleType) -> None:
    if not hasattr(module, _MOSS_FLASH_ATTN_SYMBOL):
        raise RuntimeError(
            "MOSS-TTS Local sglang non-streaming vocoder backend could not patch "
            f"remote module {module.__name__}: missing {_MOSS_FLASH_ATTN_SYMBOL}"
        )


@contextmanager
def _patch_remote_flash_attention(
    module: ModuleType,
    sglang_flash_attn_varlen_func: Callable[..., Any],
) -> Iterator[ModuleType]:
    _check_patchable_remote_flash_attention(module)
    with _FLASH_ATTENTION_PATCH_LOCK:
        original_flash_attn = getattr(module, _MOSS_FLASH_ATTN_SYMBOL)
        original_has_flash = getattr(module, "HAS_FLASH_ATTN", _MISSING)
        setattr(module, _MOSS_FLASH_ATTN_SYMBOL, sglang_flash_attn_varlen_func)
        setattr(module, "HAS_FLASH_ATTN", True)
        try:
            yield module
        finally:
            setattr(module, _MOSS_FLASH_ATTN_SYMBOL, original_flash_attn)
            if original_has_flash is _MISSING:
                if hasattr(module, "HAS_FLASH_ATTN"):
                    delattr(module, "HAS_FLASH_ATTN")
            else:
                setattr(module, "HAS_FLASH_ATTN", original_has_flash)


class SGLangCodecDecodeBackend:
    """Patch MOSS varlen attention to SGLang, then use offline decode."""

    name = "sglang"

    def __init__(self, processor: Any) -> None:
        self._processor = processor
        self._sglang_flash_attn_varlen_func: Callable[..., Any] | None = None

    def _get_sglang_flash_attn_varlen_func(self) -> Callable[..., Any]:
        if self._sglang_flash_attn_varlen_func is None:
            self._sglang_flash_attn_varlen_func = _load_sglang_flash_attn_varlen_func()
        return self._sglang_flash_attn_varlen_func

    def decode_rows(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        module = _audio_tokenizer_module(self._processor)
        _check_patchable_remote_flash_attention(module)
        flash_attn_varlen_func = self._get_sglang_flash_attn_varlen_func()
        with (
            _patch_remote_flash_attention(module, flash_attn_varlen_func),
            torch_profile_range("moss.vocoder.backend.sglang_decode"),
        ):
            return [
                torch.as_tensor(wav).detach().to("cpu", torch.float32).contiguous()
                for wav in self._processor.decode_audio_codes(codes_list)
            ]


__all__ = [
    "NONSTREAM_VOCODER_BACKEND_ENV",
    "NonStreamingVocoderBackend",
    "ProcessorDecodeBackend",
    "SGLangCodecDecodeBackend",
    "SessionDecodeBackend",
    "resolve_nonstream_vocoder_backend",
]
