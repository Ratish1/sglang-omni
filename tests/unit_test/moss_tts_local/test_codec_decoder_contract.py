# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_omni.models.moss_tts_local.codec_decoder_contract import (
    EXPECTED_PATCH_FACTORS,
    EXPECTED_TRANSFORMER_CONTEXT_DURATIONS,
    EXPECTED_TRANSFORMER_FFN_DIMS,
    EXPECTED_TRANSFORMER_HEADS,
    EXPECTED_TRANSFORMER_INPUT_DIMS,
    EXPECTED_TRANSFORMER_LAYER_COUNTS,
    EXPECTED_TRANSFORMER_MODEL_DIMS,
    EXPECTED_TRANSFORMER_OUTPUT_DIMS,
    MossCodecDecoderContract,
    MossCodecDecoderContractError,
    MossPatchStageSpec,
    MossTransformerStageSpec,
    build_moss_codec_decoder_contract,
    plan_moss_nonstream_decode_chunks,
)


def _decoder_kwargs() -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []
    for index in range(6):
        stages.append(
            {
                "module_type": "Transformer",
                "input_dimension": EXPECTED_TRANSFORMER_INPUT_DIMS[index],
                "d_model": EXPECTED_TRANSFORMER_MODEL_DIMS[index],
                "output_dimension": EXPECTED_TRANSFORMER_OUTPUT_DIMS[index],
                "num_layers": EXPECTED_TRANSFORMER_LAYER_COUNTS[index],
                "num_heads": EXPECTED_TRANSFORMER_HEADS[index],
                "dim_feedforward": EXPECTED_TRANSFORMER_FFN_DIMS[index],
                "context_duration": EXPECTED_TRANSFORMER_CONTEXT_DURATIONS[index],
                "causal": True,
                "conv_layout": True,
                "norm": "layer_norm",
                "gating": "none",
                "positional_embedding": "rope",
                "layer_scale": 0.01,
                "max_period": 10000,
            }
        )
        stages.append(
            {
                "module_type": "PatchedPretransform",
                "patch_size": EXPECTED_PATCH_FACTORS[index],
            }
        )
    return stages


def _state_shapes(contract: MossCodecDecoderContract) -> dict[str, tuple[int, ...]]:
    state_dict: dict[str, tuple[int, ...]] = {}
    for stage in contract.transformer_stages:
        prefix = f"{stage.stage_index}"
        state_dict[f"{prefix}.input_proj.weight"] = (
            stage.d_model,
            stage.input_dimension,
        )
        state_dict[f"{prefix}.output_proj.weight"] = (
            stage.output_dimension,
            stage.d_model,
        )
        for layer in range(stage.num_layers):
            layer_prefix = f"{prefix}.transformer.layers.{layer}"
            state_dict[f"{layer_prefix}.self_attn.in_proj.weight"] = (
                3 * stage.d_model,
                stage.d_model,
            )
            state_dict[f"{layer_prefix}.self_attn.out_proj.weight"] = (
                stage.d_model,
                stage.d_model,
            )
            state_dict[f"{layer_prefix}.ffn.0.weight"] = (
                stage.dim_feedforward,
                stage.d_model,
            )
            state_dict[f"{layer_prefix}.ffn.2.weight"] = (
                stage.d_model,
                stage.dim_feedforward,
            )
            for norm in ("norm1", "norm2"):
                state_dict[f"{layer_prefix}.{norm}.weight"] = (stage.d_model,)
                state_dict[f"{layer_prefix}.{norm}.bias"] = (stage.d_model,)
            for scale in ("layer_scale_1", "layer_scale_2"):
                state_dict[f"{layer_prefix}.{scale}.scale"] = (stage.d_model,)
    return state_dict


class MossAudioTokenizerPatchedPretransform(torch.nn.Module):
    def __init__(self, patch_size: int, *, is_downsample: bool = False) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.is_downsample = is_downsample


class _Scale(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.full((channels,), 0.01))


class _Attention(torch.nn.Module):
    def __init__(self, spec: MossTransformerStageSpec) -> None:
        super().__init__()
        self.embed_dim = spec.d_model
        self.num_heads = spec.num_heads
        self.causal = spec.causal
        self.in_proj = torch.nn.Linear(spec.d_model, 3 * spec.d_model, bias=False)
        self.out_proj = torch.nn.Linear(spec.d_model, spec.d_model, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, spec: MossTransformerStageSpec) -> None:
        super().__init__()
        self.self_attn = _Attention(spec)
        self.norm1 = torch.nn.LayerNorm(spec.d_model)
        self.norm2 = torch.nn.LayerNorm(spec.d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(spec.d_model, spec.dim_feedforward, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(spec.dim_feedforward, spec.d_model, bias=False),
        )
        self.layer_scale_1 = _Scale(spec.d_model)
        self.layer_scale_2 = _Scale(spec.d_model)


class _Transformer(torch.nn.Module):
    def __init__(self, spec: MossTransformerStageSpec) -> None:
        super().__init__()
        self.positional_embedding = spec.positional_embedding
        self.max_period = spec.max_period
        self.layers = torch.nn.ModuleList(
            [_Layer(spec) for _ in range(spec.num_layers)]
        )


class MossAudioTokenizerProjectedTransformer(torch.nn.Module):
    def __init__(self, spec: MossTransformerStageSpec) -> None:
        super().__init__()
        self.input_proj = torch.nn.Linear(
            spec.input_dimension, spec.d_model, bias=False
        )
        self.transformer = _Transformer(spec)
        self.output_proj = torch.nn.Linear(
            spec.d_model, spec.output_dimension, bias=False
        )


def _decoder_modules(
    contract: MossCodecDecoderContract,
) -> torch.nn.ModuleList:
    modules: list[torch.nn.Module] = []
    for stage in contract.stages:
        if isinstance(stage, MossTransformerStageSpec):
            modules.append(MossAudioTokenizerProjectedTransformer(stage))
        else:
            modules.append(MossAudioTokenizerPatchedPretransform(stage.patch_size))
    return torch.nn.ModuleList(modules)


def test_contract_builds_from_remote_code_decoder_kwargs_shape() -> None:
    config = SimpleNamespace(decoder_kwargs=_decoder_kwargs())

    contract = build_moss_codec_decoder_contract(config)

    assert len(contract.stages) == 12
    assert len(contract.transformer_stages) == 6
    assert len(contract.patch_stages) == 6
    assert contract.transformer_layer_count == 92
    assert contract.n_vq == 12
    assert contract.max_decode_chunk_frames == 100
    assert contract.sample_rate == 48000
    assert contract.frame_rate == 12.5
    assert contract.frame_samples == 3840
    assert [type(stage) for stage in contract.stages[::2]] == [
        MossTransformerStageSpec
    ] * 6
    assert [type(stage) for stage in contract.stages[1::2]] == [MossPatchStageSpec] * 6

    contract.validate_state_dict(_state_shapes(contract))


def test_contract_validates_remote_module_tree_shape() -> None:
    contract = build_moss_codec_decoder_contract({"decoder_kwargs": _decoder_kwargs()})

    contract.validate_module_tree(_decoder_modules(contract))


def test_contract_rejects_downsample_patch_module() -> None:
    contract = build_moss_codec_decoder_contract({"decoder_kwargs": _decoder_kwargs()})
    decoder = _decoder_modules(contract)
    decoder[1] = MossAudioTokenizerPatchedPretransform(2, is_downsample=True)

    with pytest.raises(MossCodecDecoderContractError, match="decode, not downsample"):
        contract.validate_module_tree(decoder)


def test_contract_rejects_later_layer_attention_mismatch() -> None:
    contract = build_moss_codec_decoder_contract({"decoder_kwargs": _decoder_kwargs()})
    decoder = _decoder_modules(contract)
    transformer = decoder[0].transformer
    transformer.layers[1].self_attn.num_heads = 10

    with pytest.raises(MossCodecDecoderContractError, match="layer 1"):
        contract.validate_module_tree(decoder)


def test_decode_chunk_planner_matches_processor_chunk_duration() -> None:
    assert plan_moss_nonstream_decode_chunks(25) == [25]
    assert plan_moss_nonstream_decode_chunks(100) == [100]
    assert plan_moss_nonstream_decode_chunks(300) == [100, 100, 100]


def test_contract_rejects_wrong_patch_factor() -> None:
    decoder_kwargs = _decoder_kwargs()
    decoder_kwargs[1]["patch_size"] = 3

    with pytest.raises(MossCodecDecoderContractError, match="patch factors"):
        build_moss_codec_decoder_contract({"decoder_kwargs": decoder_kwargs})


def test_contract_rejects_unsupported_norm_or_gating() -> None:
    decoder_kwargs = _decoder_kwargs()
    decoder_kwargs[0]["norm"] = "rms_norm"

    with pytest.raises(MossCodecDecoderContractError, match="norm"):
        build_moss_codec_decoder_contract({"decoder_kwargs": decoder_kwargs})

    decoder_kwargs = _decoder_kwargs()
    decoder_kwargs[0]["gating"] = "silu"

    with pytest.raises(MossCodecDecoderContractError, match="gating"):
        build_moss_codec_decoder_contract({"decoder_kwargs": decoder_kwargs})


def test_contract_rejects_unsupported_stage_kind() -> None:
    decoder_kwargs = _decoder_kwargs()
    decoder_kwargs[3]["module_type"] = "ConvTranspose"

    with pytest.raises(MossCodecDecoderContractError, match="unsupported"):
        build_moss_codec_decoder_contract({"decoder_kwargs": decoder_kwargs})


def test_contract_rejects_wrong_layer_count() -> None:
    decoder_kwargs = _decoder_kwargs()
    decoder_kwargs[0]["num_layers"] = 31

    with pytest.raises(MossCodecDecoderContractError, match="num_layers"):
        build_moss_codec_decoder_contract({"decoder_kwargs": decoder_kwargs})


def test_state_dict_validation_rejects_wrong_weight_shape() -> None:
    contract = build_moss_codec_decoder_contract({"decoder_kwargs": _decoder_kwargs()})
    state_dict = _state_shapes(contract)
    state_dict["0.transformer.layers.0.self_attn.in_proj.weight"] = (1280, 1280)

    with pytest.raises(MossCodecDecoderContractError, match="in_proj"):
        contract.validate_state_dict(state_dict)


def test_state_dict_validation_rejects_missing_representative_weight() -> None:
    contract = build_moss_codec_decoder_contract({"decoder_kwargs": _decoder_kwargs()})
    state_dict = _state_shapes(contract)
    del state_dict["10.transformer.layers.11.layer_scale_2.scale"]

    with pytest.raises(MossCodecDecoderContractError, match="layer_scale_2"):
        contract.validate_state_dict(state_dict)
