# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import sglang_omni.models.moss_tts_local.vocoder_sglang_patch as vocoder_sglang_patch
from sglang_omni.models.moss_tts_local.vocoder_sglang_patch import (
    install_moss_tts_local_sglang_vocoder_patch,
    uninstall_moss_tts_local_sglang_vocoder_patch,
)


def _fake_codec(module_name: str):
    def original_flash_attn(*args, **kwargs):
        return ("original", args, kwargs)

    fake_module = ModuleType(module_name)
    fake_module.flash_attn_varlen_func = original_flash_attn
    fake_module.HAS_FLASH_ATTN = False

    attention_cls = type(
        "MossAudioTokenizerMultiheadAttention",
        (),
        {
            "__module__": module_name,
            "_run_flash_attention": lambda self: None,
            "resolve_attention_implementation": lambda self: "sdpa",
        },
    )
    attention = attention_cls()
    decoder = SimpleNamespace(modules=lambda: iter([decoder, attention]))
    codec = SimpleNamespace(decoder=decoder)
    return codec, fake_module, original_flash_attn


def test_sglang_patch_installs_and_restores_remote_attention_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec, fake_module, original_flash_attn = _fake_codec(
        "fake_moss_audio_tokenizer_patchable"
    )
    sglang_flash_attn = object()
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(
        vocoder_sglang_patch,
        "_load_sglang_flash_attn_varlen_func",
        lambda: sglang_flash_attn,
    )

    info = install_moss_tts_local_sglang_vocoder_patch(codec)

    assert info.attention_modules == 1
    assert info.python_modules == 1
    assert fake_module.flash_attn_varlen_func is sglang_flash_attn
    assert fake_module.HAS_FLASH_ATTN is True

    second_info = install_moss_tts_local_sglang_vocoder_patch(codec)
    assert second_info == info
    assert fake_module.flash_attn_varlen_func is sglang_flash_attn

    restored = uninstall_moss_tts_local_sglang_vocoder_patch(codec)

    assert restored.attention_modules == 1
    assert restored.python_modules == 1
    assert fake_module.flash_attn_varlen_func is original_flash_attn
    assert fake_module.HAS_FLASH_ATTN is False


def test_sglang_patch_rejects_unexpected_codec_shape() -> None:
    codec = SimpleNamespace(decoder=SimpleNamespace(modules=lambda: iter([])))

    with pytest.raises(RuntimeError, match="found no MossAudioTokenizer"):
        install_moss_tts_local_sglang_vocoder_patch(codec)
