# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from sglang_omni.models.moss_tts_local.vocoder_introspection import (
    summarize_moss_tts_local_vocoder,
)


class _FakeLayerScale(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class _FakeAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, context: int) -> None:
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.causal = True
        self.context = context
        self.max_period = 10_000.0
        self.in_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, **_: object) -> torch.Tensor:
        return self.out_proj(self.in_proj(x)[..., : self.embed_dim])


class _FakeLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, ffn_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = _FakeAttention(hidden_size, num_heads, context=5)
        self.layer_scale_1 = _FakeLayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, ffn_size),
            nn.GELU(),
            nn.Linear(ffn_size, hidden_size),
        )
        self.layer_scale_2 = _FakeLayerScale(hidden_size)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        x = x + self.layer_scale_1(self.self_attn(self.norm1(x), **kwargs))
        return x + self.layer_scale_2(self.ffn(self.norm2(x)))


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positional_embedding = "rope"
        self.max_period = 10_000.0
        self.positional_scale = 1.0
        self.layers = nn.ModuleList([_FakeLayer(16, 4, 64), _FakeLayer(16, 4, 64)])

    def resolve_attention_implementation(self, _: torch.Tensor) -> str:
        return "flash_attention_2"

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class _FakeProjectedTransformer(nn.Module):
    context_duration = 2.5

    def __init__(self) -> None:
        super().__init__()
        self.is_streaming = False
        self.input_proj = nn.Linear(8, 16)
        self.transformer = _FakeTransformer()
        self.output_proj = nn.Linear(16, 10)

    def forward(
        self, x: torch.Tensor, input_lengths: torch.Tensor, **kwargs: object
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x.transpose(1, 2))
        x = self.transformer(x, **kwargs)
        return self.output_proj(x).transpose(1, 2), input_lengths


class _FakePatch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 2
        self.downsample_ratio = 2
        self.is_downsample = False
        self.module_type = "PatchedPretransform"

    def decode(
        self, x: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, dh, length = x.shape
        patch = self.patch_size
        d = dh // patch
        return (
            x.reshape(b, d, patch, length)
            .permute(0, 1, 3, 2)
            .reshape(b, d, length * patch),
            input_lengths * patch,
        )

    def forward(
        self, x: torch.Tensor, input_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode(x, input_lengths)


class _FakeCodec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(sampling_rate=48_000)
        self.decoder = nn.ModuleList([_FakeProjectedTransformer(), _FakePatch()])

    def _decode_frame(self, codes: torch.Tensor, codes_lengths: torch.Tensor):
        return SimpleNamespace(audio=codes.float(), audio_lengths=codes_lengths)


class _FakeProcessor:
    def __init__(self) -> None:
        self.model_config = SimpleNamespace(n_vq=12, audio_vocab_size=1024)
        self.audio_tokenizer = _FakeCodec()

    def decode_audio_codes(self, codes_list, *, return_stereo: bool = True):
        return [codes.float().T for codes in codes_list]


def test_vocoder_introspection_summarizes_decoder_topology() -> None:
    summary = summarize_moss_tts_local_vocoder(_FakeProcessor())

    decoder = summary["decoder"]
    assert decoder["stage_count"] == 2
    assert decoder["transformer_stage_count"] == 1
    assert decoder["transformer_layer_count"] == 2

    transformer_stage = decoder["stages"][0]
    assert transformer_stage["module_type"] == "Transformer"
    assert transformer_stage["input_dimension"] == 8
    assert transformer_stage["d_model"] == 16
    assert transformer_stage["output_dimension"] == 10
    assert transformer_stage["layers"] == 2
    assert transformer_stage["heads"] == 4
    assert transformer_stage["head_dim"] == 4
    assert transformer_stage["ffn"] == 64
    assert transformer_stage["context"] == 5
    assert transformer_stage["context_duration"] == 2.5

    first_layer = transformer_stage["transformer"]["first_layer"]
    assert first_layer["norm1"]["class_name"] == "LayerNorm"
    assert first_layer["layer_scale_1"]["shape"] == [16]
    assert first_layer["self_attn"]["in_proj"]["weight"]["shape"] == [48, 16]

    patch_stage = decoder["stages"][1]
    assert patch_stage["module_type"] == "PatchedPretransform"
    assert patch_stage["patch_size"] == 2
    assert patch_stage["methods"]["decode"]["present"] is True

    processor = summary["processor"]
    assert processor["model_config_status"]["present"] is True
    sources = {item["label"]: item for item in processor["config_sources"]}
    assert sources["processor.model_config"]["values"]["n_vq"] == 12
    assert sources["audio_tokenizer.config"]["values"]["sampling_rate"] == 48_000


def test_vocoder_introspection_reports_missing_processor_model_config() -> None:
    processor = _FakeProcessor()
    del processor.model_config

    summary = summarize_moss_tts_local_vocoder(processor)

    processor_summary = summary["processor"]
    assert processor_summary["model_config"] is None
    assert processor_summary["model_config_status"]["present"] is False
    sources = {item["label"]: item for item in processor_summary["config_sources"]}
    assert sources["processor.model_config"]["present"] is False
    assert sources["audio_tokenizer.config"]["present"] is True
    assert sources["audio_tokenizer.config"]["values"]["sampling_rate"] == 48_000


def test_vocoder_introspection_groups_decoder_state_dict_by_stage_layer() -> None:
    summary = summarize_moss_tts_local_vocoder(_FakeProcessor())
    groups = summary["decoder"]["state_dict_groups"]

    in_proj_group = groups["0.transformer.layers.{layer}.self_attn.in_proj.weight"]
    assert in_proj_group["count"] == 2
    assert in_proj_group["layers"] == [0, 1]
    assert in_proj_group["layer_count"] == 2
    assert in_proj_group["examples"][0]["shape"] == [48, 16]

    input_proj_group = groups["0.input_proj.*"]
    assert input_proj_group["count"] == 2
    assert {tuple(item["shape"]) for item in input_proj_group["examples"]} == {
        (16, 8),
        (16,),
    }
