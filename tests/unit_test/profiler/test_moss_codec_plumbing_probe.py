# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.debug.moss_codec_plumbing_probe import (
    _applied_candidate,
    _iter_projected_transformers,
    _iter_self_attention_modules,
    _iter_transformer_layers,
    summarize_candidate_cases,
)


def _case(candidate: str, *, max_abs: float, speedup: float) -> dict:
    return {
        "candidate": candidate,
        "parity": {
            "shape_equal": True,
            "max_abs": max_abs,
            "mean_abs": max_abs,
            "max_rel": max_abs,
        },
        "timing": {
            "candidate_speedup_pct": speedup,
        },
    }


def test_plumbing_probe_accepts_only_exact_parity_and_consistent_speedup() -> None:
    rows = summarize_candidate_cases(
        [
            _case("good", max_abs=0.0, speedup=4.0),
            _case("good", max_abs=0.0, speedup=3.5),
            _case("drift", max_abs=0.1, speedup=30.0),
            _case("mixed", max_abs=0.0, speedup=6.0),
            _case("mixed", max_abs=0.0, speedup=-1.0),
        ],
        min_speedup_pct=3.0,
    )

    by_candidate = {row["candidate"]: row for row in rows}

    assert by_candidate["good"]["accepted"] is True
    assert by_candidate["good"]["parity_pass"] == 2
    assert by_candidate["drift"]["accepted"] is False
    assert by_candidate["drift"]["parity_fail"] == 1
    assert by_candidate["mixed"]["accepted"] is False
    assert by_candidate["mixed"]["min_speedup_pct"] < 0.0


class _FakeModule:
    def forward(self, x):
        return x


class _FakeLayer:
    def __init__(self) -> None:
        self.self_attn = _FakeModule()

    def forward(self, x):
        return self.self_attn.forward(x)


class _FakeProjectedTransformer:
    def __init__(self) -> None:
        self.input_proj = _FakeModule()
        self.transformer = SimpleNamespace(layers=[_FakeLayer(), _FakeLayer()])
        self.output_proj = _FakeModule()

    def forward(self, x, input_lengths):
        return x, input_lengths


def _fake_processor() -> SimpleNamespace:
    codec = SimpleNamespace(decoder=[_FakeProjectedTransformer()])
    return SimpleNamespace(audio_tokenizer=codec)


def test_plumbing_probe_discovers_compile_targets() -> None:
    processor = _fake_processor()
    codec = processor.audio_tokenizer

    assert len(_iter_projected_transformers(codec)) == 1
    assert len(_iter_transformer_layers(codec)) == 2
    assert len(_iter_self_attention_modules(codec)) == 2


def test_compile_candidate_patches_forward_temporarily(monkeypatch) -> None:
    processor = _fake_processor()
    layers = _iter_transformer_layers(processor.audio_tokenizer)
    originals = [layer.forward for layer in layers]

    def fake_compile(fn, **kwargs):
        def compiled(*args, **inner_kwargs):
            return fn(*args, **inner_kwargs)

        compiled._compile_kwargs = kwargs  # type: ignore[attr-defined]
        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    monkeypatch.setattr(
        "scripts.debug.moss_codec_plumbing_probe._set_torch_compile_config",
        lambda: None,
    )

    with _applied_candidate(
        "compile_transformer_layers",
        processor=processor,
        compile_mode="reduce-overhead",
    ) as stats:
        patched = [layer.forward for layer in layers]
        assert stats["compiled_count"] == 2
        assert stats["compile_mode"] == "reduce-overhead"
        assert patched != originals
        assert all(
            getattr(forward, "_compile_kwargs")["mode"] == "reduce-overhead"
            for forward in patched
        )

    assert [layer.forward for layer in layers] == originals
