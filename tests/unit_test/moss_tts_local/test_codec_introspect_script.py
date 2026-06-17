# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "debug"
    / "moss_tts_local_codec_introspect.py"
)


@pytest.fixture(scope="module")
def introspect_module():
    spec = importlib.util.spec_from_file_location(
        "moss_tts_local_codec_introspect", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_group_state_dict_keys_collapses_layer_indices(introspect_module) -> None:
    items = [
        ("decoder.layers.0.attn.q_proj.weight", {"shape": [64, 64]}),
        ("decoder.layers.1.attn.q_proj.weight", {"shape": [64, 64]}),
        ("decoder.layers.1.mlp.fc.weight", {"shape": [256, 64]}),
        ("decoder.final_norm.weight", {"shape": [64]}),
    ]

    groups = introspect_module.group_state_dict_keys(items)

    q_proj = next(
        group
        for group in groups
        if group["template"] == "decoder.layers.{layer}.attn.q_proj.weight"
    )
    assert q_proj["count"] == 2
    assert q_proj["layer_indices"] == [0, 1]
    assert q_proj["layer_count"] == 2

    final_norm = next(
        group for group in groups if group["template"] == "decoder.final_norm.*"
    )
    assert final_norm["count"] == 1
    assert final_norm["layer_indices"] == []


def test_json_safe_serializes_tensor_like_values(introspect_module) -> None:
    class TensorLike:
        shape = (2, 3)
        dtype = "float32"
        device = "cpu"

    payload = {
        "tensor": TensorLike(),
        "path": Path("x/y"),
        "long": "a" * 12,
    }

    safe = introspect_module.json_safe(payload, max_string=8)

    assert safe["tensor"] == {
        "shape": [2, 3],
        "dtype": "float32",
        "device": "cpu",
    }
    assert safe["path"] == "x/y"
    assert safe["long"].startswith("aaaaaaaa...<truncated")


def test_parse_int_list_rejects_empty(introspect_module) -> None:
    assert introspect_module.parse_int_list("5, 10") == [5, 10]
    with pytest.raises(Exception):
        introspect_module.parse_int_list(" , ")


def test_waveform_stats_compares_common_aligned_region(introspect_module) -> None:
    left = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    right = torch.tensor([0.0, 1.5, 2.0])

    stats = introspect_module.waveform_stats(left, right)

    comparison = stats["comparison"]
    assert comparison["same_shape"] is False
    assert comparison["comparable"] is True
    assert comparison["common_shape"] == [1, 3]
    assert comparison["max_abs_delta"] == pytest.approx(0.5)


def test_capture_decode_frame_records_shapes_without_changing_result(
    introspect_module,
) -> None:
    class FakeCodec:
        def _decode_frame(self, codes, lengths):
            return SimpleNamespace(audio=codes.float(), audio_lengths=lengths)

        def named_modules(self):
            return [("", self)]

    codec = FakeCodec()
    codes = torch.zeros((2, 1, 4), dtype=torch.long)
    lengths = torch.tensor([4], dtype=torch.long)

    report = introspect_module.capture_decode_frame_calls(
        codec,
        lambda: codec._decode_frame(codes, lengths),
    )
    result = codec._decode_frame(codes, lengths)

    assert report["ok"] is True
    assert report["count"] == 1
    assert report["records"][0]["inputs"][0]["shape"] == [2, 1, 4]
    assert report["records"][0]["output"]["audio"]["shape"] == [2, 1, 4]
    assert torch.equal(result.audio, codes.float())


def test_decoder_stage_config_summary_reports_transformer_contract(
    introspect_module,
) -> None:
    config = SimpleNamespace(
        decoder_kwargs=[
            {
                "module_type": "Transformer",
                "d_model": 1280,
                "num_heads": 20,
                "num_layers": 32,
                "dim_feedforward": 5120,
                "input_dimension": 768,
                "output_dimension": 1280,
                "context_duration": 10.0,
                "causal": True,
                "positional_embedding": "rope",
            },
            {"module_type": "PatchedPretransform", "patch_size": 2},
            {
                "module_type": "Transformer",
                "d_model": 768,
                "num_heads": 12,
                "num_layers": 12,
                "dim_feedforward": 3072,
                "input_dimension": 640,
                "output_dimension": 768,
                "context_duration": 10.0,
                "causal": True,
                "positional_embedding": "rope",
            },
        ]
    )

    summary = introspect_module.decoder_stage_config_summary(config)

    assert summary["present"] is True
    assert summary["stage_count"] == 3
    assert summary["transformer_stage_count"] == 2
    assert summary["transformer_layer_count"] == 44
    assert summary["stages"][0] == {
        "stage_index": 0,
        "module_type": "Transformer",
        "d_model": 1280,
        "heads": 20,
        "head_dim": 64,
        "layers": 32,
        "ffn": 5120,
        "input_dimension": 768,
        "output_dimension": 1280,
        "context_duration": 10.0,
        "patch_size": None,
        "causal": True,
        "positional_embedding": "rope",
    }
    assert summary["stages"][1]["patch_size"] == 2


def test_codec_decoder_forward_contracts_discovers_helpers(introspect_module) -> None:
    class MossAudioTokenizerTransformerLayer(torch.nn.Module):
        def forward(self, hidden_states, attention_mask=None):
            hidden_states = self._attention_block(hidden_states, attention_mask)
            return self._feed_forward(hidden_states)

        def _attention_block(self, hidden_states, attention_mask=None):
            del attention_mask
            return hidden_states + 1

        def _feed_forward(self, hidden_states):
            return hidden_states * 2

    class MossAudioTokenizerMultiheadAttention(torch.nn.Module):
        def forward(self, query, key, value, *, is_causal=True):
            packed = self._pack_qkv(query, key, value)
            return packed if is_causal else packed

        def _pack_qkv(self, query, key, value):
            return query + key + value

    root = torch.nn.Module()
    root.layer = MossAudioTokenizerTransformerLayer()
    root.attn = MossAudioTokenizerMultiheadAttention()

    contracts = introspect_module.codec_decoder_forward_contracts(root, max_lines=20)

    assert contracts["present"] is True
    layer = contracts["classes"]["MossAudioTokenizerTransformerLayer"]
    assert layer["module_count"] == 1
    assert layer["module_examples"] == ["layer"]
    assert set(layer["forward_helper_methods"]) == {
        "_attention_block",
        "_feed_forward",
    }
    assert "attention_mask" in layer["methods"]["forward"]["signature"]
    assert "_attention_block" in layer["methods"]
    assert (
        "hidden_states + 1"
        in layer["methods"]["_attention_block"]["source_snippet"]["text"]
    )

    attn = contracts["classes"]["MossAudioTokenizerMultiheadAttention"]
    assert attn["forward_helper_methods"] == ["_pack_qkv"]
    assert "is_causal" in attn["methods"]["forward"]["signature"]
    assert "MossAudioTokenizerRotaryEmbedding" in contracts["missing_target_classes"]
