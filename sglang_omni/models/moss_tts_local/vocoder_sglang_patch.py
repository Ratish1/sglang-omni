# SPDX-License-Identifier: Apache-2.0
"""SGLang FlashAttention patch for the upstream MOSS audio tokenizer."""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_EXPECTED_ATTENTION_CLASS = "MossAudioTokenizerMultiheadAttention"
_PATCH_STATE_ATTR = "_sglang_omni_moss_vocoder_patch_state"
_PATCH_LOCK = threading.RLock()


@dataclass(frozen=True)
class MossTTSLocalSGLangVocoderPatchInfo:
    attention_modules: int
    python_modules: int
    ref_count: int = 0
    invocation_count: int = 0
    attention_implementations: Mapping[str, int] = field(default_factory=dict)


@dataclass
class _ModulePatchState:
    flash_attn_varlen_func: Any
    has_flash_attn: Any
    sglang_flash_attn_varlen_func: Any
    adapter: Any
    ref_count: int = 0
    invocation_count: int = 0


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
            f"MOSS vocoder SGLang patch found no {_EXPECTED_ATTENTION_CLASS} modules"
        )
    return attention_modules


def _attention_implementation_counts(attention_modules: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attention_module in attention_modules:
        implementation = str(
            getattr(attention_module, "attention_implementation", "unknown")
        )
        counts[implementation] = counts.get(implementation, 0) + 1
    return counts


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


def _reject_non_default_argument(name: str, value: Any, default: Any) -> None:
    if value != default:
        raise NotImplementedError(
            "MOSS vocoder SGLang patch does not support "
            f"{name}={value!r}; expected {default!r}"
        )


def _make_flash_attn_varlen_adapter(state: _ModulePatchState):
    def flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=None,
        max_seqlen_k=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=None,
        softcap=0.0,
        seqused_q=None,
        seqused_k=None,
        **kwargs,
    ):
        _reject_non_default_argument("dropout_p", dropout_p, 0.0)
        _reject_non_default_argument("alibi_slopes", alibi_slopes, None)
        _reject_non_default_argument("deterministic", deterministic, False)
        _reject_non_default_argument("return_attn_probs", return_attn_probs, False)
        _reject_non_default_argument("block_table", block_table, None)
        if kwargs:
            raise TypeError(
                "MOSS vocoder SGLang patch received unsupported "
                f"flash_attn_varlen_func arguments: {sorted(kwargs)}"
            )
        state.invocation_count += 1
        return state.sglang_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
        )

    return flash_attn_varlen_func


def _patch_info(
    attention_modules: list[Any],
    python_modules: set[ModuleType],
) -> MossTTSLocalSGLangVocoderPatchInfo:
    states = [
        getattr(python_module, _PATCH_STATE_ATTR, None)
        for python_module in python_modules
    ]
    ref_count = max(
        (int(getattr(state, "ref_count", 0)) for state in states if state is not None),
        default=0,
    )
    invocation_count = sum(
        int(getattr(state, "invocation_count", 0))
        for state in states
        if state is not None
    )
    return MossTTSLocalSGLangVocoderPatchInfo(
        attention_modules=len(attention_modules),
        python_modules=len(python_modules),
        ref_count=ref_count,
        invocation_count=invocation_count,
        attention_implementations=_attention_implementation_counts(attention_modules),
    )


def install_moss_tts_local_sglang_vocoder_patch(
    codec: Any,
) -> MossTTSLocalSGLangVocoderPatchInfo:
    """Route the upstream MOSS vocoder FlashAttention hook to SGLang."""
    attention_modules = _iter_moss_attention_modules(codec)
    python_modules = {_module_for_attention(module) for module in attention_modules}
    sglang_flash_attn_varlen_func = _load_sglang_flash_attn_varlen_func()
    with _PATCH_LOCK:
        for python_module in python_modules:
            state = getattr(python_module, _PATCH_STATE_ATTR, None)
            if state is None:
                state = _ModulePatchState(
                    flash_attn_varlen_func=python_module.flash_attn_varlen_func,
                    has_flash_attn=python_module.HAS_FLASH_ATTN,
                    sglang_flash_attn_varlen_func=sglang_flash_attn_varlen_func,
                    adapter=None,
                )
                state.adapter = _make_flash_attn_varlen_adapter(state)
                setattr(python_module, _PATCH_STATE_ATTR, state)
            state.ref_count += 1
            state.sglang_flash_attn_varlen_func = sglang_flash_attn_varlen_func
            python_module.flash_attn_varlen_func = state.adapter
            python_module.HAS_FLASH_ATTN = True
        info = _patch_info(attention_modules, python_modules)
    logger.info(
        "MOSS-TTS Local vocoder SGLang patch installed "
        "attention_modules=%d python_modules=%d ref_count=%d "
        "attention_implementations=%s",
        info.attention_modules,
        info.python_modules,
        info.ref_count,
        dict(info.attention_implementations),
    )
    return info


def uninstall_moss_tts_local_sglang_vocoder_patch(
    codec: Any,
) -> MossTTSLocalSGLangVocoderPatchInfo:
    """Restore tokenizer module globals changed by the SGLang patch."""
    attention_modules = _iter_moss_attention_modules(codec)
    python_modules = {_module_for_attention(module) for module in attention_modules}
    with _PATCH_LOCK:
        invocation_count = 0
        ref_count = 0
        for python_module in python_modules:
            state = getattr(python_module, _PATCH_STATE_ATTR, None)
            if state is None:
                continue
            invocation_count += int(state.invocation_count)
            state.ref_count = max(0, state.ref_count - 1)
            ref_count = max(ref_count, state.ref_count)
            if state.ref_count > 0:
                continue
            python_module.flash_attn_varlen_func = state.flash_attn_varlen_func
            python_module.HAS_FLASH_ATTN = state.has_flash_attn
            delattr(python_module, _PATCH_STATE_ATTR)
        return MossTTSLocalSGLangVocoderPatchInfo(
            attention_modules=len(attention_modules),
            python_modules=len(python_modules),
            ref_count=ref_count,
            invocation_count=invocation_count,
            attention_implementations=_attention_implementation_counts(
                attention_modules
            ),
        )


def get_moss_tts_local_sglang_vocoder_patch_info(
    codec: Any,
) -> MossTTSLocalSGLangVocoderPatchInfo:
    """Return current patch counters for the loaded MOSS tokenizer module."""
    attention_modules = _iter_moss_attention_modules(codec)
    python_modules = {_module_for_attention(module) for module in attention_modules}
    with _PATCH_LOCK:
        return _patch_info(attention_modules, python_modules)


__all__ = [
    "MossTTSLocalSGLangVocoderPatchInfo",
    "get_moss_tts_local_sglang_vocoder_patch_info",
    "install_moss_tts_local_sglang_vocoder_patch",
    "uninstall_moss_tts_local_sglang_vocoder_patch",
]
