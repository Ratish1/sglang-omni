# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

import sglang_omni.models.moss_tts_local.vocoder_sglang_patch as vocoder_sglang_patch
from sglang_omni.models.moss_tts_local.vocoder_sglang_patch import (
    get_moss_tts_local_sglang_vocoder_patch_info,
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
    attention.attention_implementation = "flash_attention_2"
    decoder = SimpleNamespace(modules=lambda: iter([decoder, attention]))
    codec = SimpleNamespace(decoder=decoder)
    return codec, fake_module, original_flash_attn


def test_sglang_patch_installs_and_restores_remote_attention_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codec, fake_module, original_flash_attn = _fake_codec(
        "fake_moss_audio_tokenizer_patchable"
    )
    sglang_calls = []

    def sglang_flash_attn(*args, **kwargs):
        sglang_calls.append((args, kwargs))
        return "sglang-result"

    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(
        vocoder_sglang_patch,
        "_load_sglang_flash_attn_varlen_func",
        lambda: sglang_flash_attn,
    )

    info = install_moss_tts_local_sglang_vocoder_patch(codec)

    assert info.attention_modules == 1
    assert info.python_modules == 1
    assert info.ref_count == 1
    assert dict(info.attention_implementations) == {"flash_attention_2": 1}
    assert fake_module.flash_attn_varlen_func is not original_flash_attn
    assert fake_module.HAS_FLASH_ATTN is True

    output = fake_module.flash_attn_varlen_func(
        "q",
        "k",
        "v",
        "cu_q",
        "cu_k",
        max_seqlen_q=7,
        max_seqlen_k=11,
        softmax_scale=0.25,
        causal=True,
        window_size=(125, 0),
        seqused_q="seq_q",
        seqused_k="seq_k",
        softcap=0.0,
    )
    patch_info = get_moss_tts_local_sglang_vocoder_patch_info(codec)
    assert output == "sglang-result"
    assert patch_info.invocation_count == 1
    assert sglang_calls == [
        (
            ("q", "k", "v", "cu_q", "cu_k"),
            {
                "max_seqlen_q": 7,
                "max_seqlen_k": 11,
                "seqused_q": "seq_q",
                "seqused_k": "seq_k",
                "softmax_scale": 0.25,
                "causal": True,
                "window_size": (125, 0),
                "softcap": 0.0,
            },
        )
    ]

    with pytest.raises(NotImplementedError, match="dropout_p"):
        fake_module.flash_attn_varlen_func("q", "k", "v", "cu_q", "cu_k", dropout_p=0.1)

    second_info = install_moss_tts_local_sglang_vocoder_patch(codec)
    assert second_info.ref_count == 2
    assert fake_module.flash_attn_varlen_func is not original_flash_attn

    still_patched = uninstall_moss_tts_local_sglang_vocoder_patch(codec)
    assert still_patched.ref_count == 1
    assert still_patched.invocation_count == 1
    assert fake_module.flash_attn_varlen_func is not original_flash_attn
    assert fake_module.HAS_FLASH_ATTN is True

    restored = uninstall_moss_tts_local_sglang_vocoder_patch(codec)

    assert restored.attention_modules == 1
    assert restored.python_modules == 1
    assert restored.ref_count == 0
    assert restored.invocation_count == 1
    assert fake_module.flash_attn_varlen_func is original_flash_attn
    assert fake_module.HAS_FLASH_ATTN is False


def test_sglang_patch_rejects_unexpected_codec_shape() -> None:
    codec = SimpleNamespace(decoder=SimpleNamespace(modules=lambda: iter([])))

    with pytest.raises(RuntimeError, match="found no MossAudioTokenizer"):
        install_moss_tts_local_sglang_vocoder_patch(codec)
