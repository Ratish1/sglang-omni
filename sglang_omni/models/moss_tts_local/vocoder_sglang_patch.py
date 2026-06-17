# SPDX-License-Identifier: Apache-2.0
"""SGLang FlashAttention patch for the upstream MOSS audio tokenizer."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_EXPECTED_ATTENTION_CLASS = "MossAudioTokenizerMultiheadAttention"
_PATCH_STATE_ATTR = "_sglang_omni_moss_vocoder_patch_state"


@dataclass(frozen=True)
class MossTTSLocalSGLangVocoderPatchInfo:
    attention_modules: int
    python_modules: int


@dataclass
class _ModulePatchState:
    flash_attn_varlen_func: Any
    has_flash_attn: Any


def _load_sglang_flash_attn_varlen_func() -> Any:
    try:
        from sglang.jit_kernel.flash_attention import (
            flash_attn_varlen_func as sglang_flash_attn_varlen_func,
        )
    except Exception as exc:
        raise RuntimeError(
            "SGLang FlashAttention is not importable; install a compatible "
            "sglang package before enabling the MOSS vocoder SGLang patch"
        ) from exc
    return sglang_flash_attn_varlen_func


def _iter_moss_attention_modules(codec: Any) -> list[Any]:
    decoder = getattr(codec, "decoder", None)
    if decoder is None:
        raise RuntimeError("MOSS vocoder SGLang patch requires codec.decoder")
    modules = getattr(decoder, "modules", None)
    if not callable(modules):
        raise RuntimeError("MOSS vocoder SGLang patch requires decoder.modules()")
    attention_modules = [
        module
        for module in modules()
        if module.__class__.__name__ == _EXPECTED_ATTENTION_CLASS
        and callable(getattr(module, "_run_flash_attention", None))
        and callable(getattr(module, "resolve_attention_implementation", None))
    ]
    if not attention_modules:
        raise RuntimeError(
            "MOSS vocoder SGLang patch found no " f"{_EXPECTED_ATTENTION_CLASS} modules"
        )
    return attention_modules


def _module_for_attention(attention_module: Any) -> ModuleType:
    module_name = attention_module.__class__.__module__
    python_module = sys.modules.get(module_name)
    if python_module is None:
        raise RuntimeError(
            f"MOSS vocoder SGLang patch cannot find loaded module {module_name!r}"
        )
    if "flash_attn_varlen_func" not in vars(python_module):
        raise RuntimeError(
            "MOSS vocoder SGLang patch requires remote tokenizer module global "
            f"flash_attn_varlen_func in {module_name!r}"
        )
    if "HAS_FLASH_ATTN" not in vars(python_module):
        raise RuntimeError(
            "MOSS vocoder SGLang patch requires remote tokenizer module global "
            f"HAS_FLASH_ATTN in {module_name!r}"
        )
    return python_module


def install_moss_tts_local_sglang_vocoder_patch(
    codec: Any,
) -> MossTTSLocalSGLangVocoderPatchInfo:
    """Route the upstream MOSS vocoder FlashAttention hook to SGLang."""
    attention_modules = _iter_moss_attention_modules(codec)
    python_modules = {_module_for_attention(module) for module in attention_modules}
    sglang_flash_attn_varlen_func = _load_sglang_flash_attn_varlen_func()
    for python_module in python_modules:
        if not hasattr(python_module, _PATCH_STATE_ATTR):
            setattr(
                python_module,
                _PATCH_STATE_ATTR,
                _ModulePatchState(
                    flash_attn_varlen_func=python_module.flash_attn_varlen_func,
                    has_flash_attn=python_module.HAS_FLASH_ATTN,
                ),
            )
        python_module.flash_attn_varlen_func = sglang_flash_attn_varlen_func
        python_module.HAS_FLASH_ATTN = True
    info = MossTTSLocalSGLangVocoderPatchInfo(
        attention_modules=len(attention_modules),
        python_modules=len(python_modules),
    )
    logger.info(
        "MOSS-TTS Local vocoder SGLang patch installed "
        "attention_modules=%d python_modules=%d",
        info.attention_modules,
        info.python_modules,
    )
    return info


def uninstall_moss_tts_local_sglang_vocoder_patch(
    codec: Any,
) -> MossTTSLocalSGLangVocoderPatchInfo:
    """Restore tokenizer module globals changed by the SGLang patch."""
    attention_modules = _iter_moss_attention_modules(codec)
    python_modules = {_module_for_attention(module) for module in attention_modules}
    restored = 0
    for python_module in python_modules:
        state = getattr(python_module, _PATCH_STATE_ATTR, None)
        if state is None:
            continue
        python_module.flash_attn_varlen_func = state.flash_attn_varlen_func
        python_module.HAS_FLASH_ATTN = state.has_flash_attn
        delattr(python_module, _PATCH_STATE_ATTR)
        restored += 1
    return MossTTSLocalSGLangVocoderPatchInfo(
        attention_modules=len(attention_modules),
        python_modules=restored,
    )


__all__ = [
    "MossTTSLocalSGLangVocoderPatchInfo",
    "install_moss_tts_local_sglang_vocoder_patch",
    "uninstall_moss_tts_local_sglang_vocoder_patch",
]
