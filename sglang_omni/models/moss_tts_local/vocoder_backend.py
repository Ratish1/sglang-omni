# SPDX-License-Identifier: Apache-2.0
"""Backend labels for MOSS-TTS Local vocoder experiments."""

from __future__ import annotations

from enum import Enum


class MossTTSLocalVocoderBackend(str, Enum):
    PROCESSOR = "processor"
    PROCESSOR_SDPA = "processor-sdpa"
    PROCESSOR_FLASH2_UPSTREAM = "processor-flash2-upstream"
    SGLANG_PATCH_EXPERIMENTAL = "sglang-patch-experimental"
    OWNED_EXPERIMENTAL = "owned-experimental"
    SESSION_OFFLINE = "session-offline"


_BACKEND_ALIASES = {
    "": MossTTSLocalVocoderBackend.PROCESSOR,
    "default": MossTTSLocalVocoderBackend.PROCESSOR,
    "processor": MossTTSLocalVocoderBackend.PROCESSOR,
    "processor-sdpa": MossTTSLocalVocoderBackend.PROCESSOR_SDPA,
    "processor_sdpa": MossTTSLocalVocoderBackend.PROCESSOR_SDPA,
    "processor-flash2-upstream": MossTTSLocalVocoderBackend.PROCESSOR_FLASH2_UPSTREAM,
    "processor_flash2_upstream": MossTTSLocalVocoderBackend.PROCESSOR_FLASH2_UPSTREAM,
    "flash2-upstream": MossTTSLocalVocoderBackend.PROCESSOR_FLASH2_UPSTREAM,
    "flash2_upstream": MossTTSLocalVocoderBackend.PROCESSOR_FLASH2_UPSTREAM,
    "sglang-patch": MossTTSLocalVocoderBackend.SGLANG_PATCH_EXPERIMENTAL,
    "sglang_patch": MossTTSLocalVocoderBackend.SGLANG_PATCH_EXPERIMENTAL,
    "sglang-patch-experimental": MossTTSLocalVocoderBackend.SGLANG_PATCH_EXPERIMENTAL,
    "sglang_patch_experimental": MossTTSLocalVocoderBackend.SGLANG_PATCH_EXPERIMENTAL,
    "owned": MossTTSLocalVocoderBackend.OWNED_EXPERIMENTAL,
    "owned-pytorch": MossTTSLocalVocoderBackend.OWNED_EXPERIMENTAL,
    "owned_pytorch": MossTTSLocalVocoderBackend.OWNED_EXPERIMENTAL,
    "owned-experimental": MossTTSLocalVocoderBackend.OWNED_EXPERIMENTAL,
    "owned_experimental": MossTTSLocalVocoderBackend.OWNED_EXPERIMENTAL,
    "session-offline": MossTTSLocalVocoderBackend.SESSION_OFFLINE,
    "session_offline": MossTTSLocalVocoderBackend.SESSION_OFFLINE,
}


def parse_moss_tts_local_vocoder_backend(
    value: str | None,
) -> MossTTSLocalVocoderBackend:
    normalized = (value or "").strip().lower()
    backend = _BACKEND_ALIASES.get(normalized)
    if backend is None:
        valid = ", ".join(sorted(_BACKEND_ALIASES))
        raise ValueError(
            f"unsupported MOSS-TTS Local vocoder backend {value!r}; "
            f"expected one of: {valid}"
        )
    return backend


def moss_tts_local_vocoder_backend_choices() -> list[str]:
    return [backend.value for backend in MossTTSLocalVocoderBackend]


def is_experimental_moss_tts_local_vocoder_backend(
    backend: MossTTSLocalVocoderBackend,
) -> bool:
    return backend in {
        MossTTSLocalVocoderBackend.SGLANG_PATCH_EXPERIMENTAL,
        MossTTSLocalVocoderBackend.OWNED_EXPERIMENTAL,
    }


__all__ = [
    "MossTTSLocalVocoderBackend",
    "is_experimental_moss_tts_local_vocoder_backend",
    "moss_tts_local_vocoder_backend_choices",
    "parse_moss_tts_local_vocoder_backend",
]
