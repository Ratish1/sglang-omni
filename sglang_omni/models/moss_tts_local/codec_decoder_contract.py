# SPDX-License-Identifier: Apache-2.0
"""Typed contract for the MOSS-TTS Local codec decoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

MOSS_TTS_LOCAL_N_VQ = 12
MOSS_TTS_LOCAL_SAMPLE_RATE = 48000
MOSS_TTS_LOCAL_FRAME_RATE = 12.5
MOSS_TTS_LOCAL_FRAME_SAMPLES = 3840
MOSS_TTS_LOCAL_MAX_DECODE_CHUNK_FRAMES = 100

EXPECTED_PATCH_FACTORS = (2, 2, 2, 2, 2, 240)
EXPECTED_TRANSFORMER_LAYER_COUNTS = (32, 12, 12, 12, 12, 12)
EXPECTED_TRANSFORMER_INPUT_DIMS = (768, 640, 384, 384, 384, 384)
EXPECTED_TRANSFORMER_MODEL_DIMS = (1280, 768, 768, 768, 768, 768)
EXPECTED_TRANSFORMER_OUTPUT_DIMS = (1280, 768, 768, 768, 768, 240)
EXPECTED_TRANSFORMER_HEADS = (20, 12, 12, 12, 12, 12)
EXPECTED_TRANSFORMER_FFN_DIMS = (5120, 3072, 3072, 3072, 3072, 3072)
EXPECTED_TRANSFORMER_CONTEXT_DURATIONS = (10.0, 10.0, 8.0, 4.0, 2.0, 1.0)


class MossCodecDecoderContractError(ValueError):
    """Raised when a codec decoder contract does not match MOSS-TTS Local."""


@dataclass(frozen=True)
class MossTransformerStageSpec:
    stage_index: int
    input_dimension: int
    d_model: int
    output_dimension: int
    num_layers: int
    num_heads: int
    dim_feedforward: int
    context_duration: float
    causal: bool
    conv_layout: bool
    norm: str
    gating: str
    positional_embedding: str
    layer_scale: float
    max_period: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


@dataclass(frozen=True)
class MossPatchStageSpec:
    stage_index: int
    patch_size: int


CodecDecoderStageSpec = MossTransformerStageSpec | MossPatchStageSpec


@dataclass(frozen=True)
class MossCodecDecoderContract:
    stages: tuple[CodecDecoderStageSpec, ...]
    n_vq: int = MOSS_TTS_LOCAL_N_VQ
    max_decode_chunk_frames: int = MOSS_TTS_LOCAL_MAX_DECODE_CHUNK_FRAMES
    sample_rate: int = MOSS_TTS_LOCAL_SAMPLE_RATE
    frame_rate: float = MOSS_TTS_LOCAL_FRAME_RATE
    frame_samples: int = MOSS_TTS_LOCAL_FRAME_SAMPLES

    @property
    def transformer_stages(self) -> tuple[MossTransformerStageSpec, ...]:
        return tuple(
            stage
            for stage in self.stages
            if isinstance(stage, MossTransformerStageSpec)
        )

    @property
    def patch_stages(self) -> tuple[MossPatchStageSpec, ...]:
        return tuple(
            stage for stage in self.stages if isinstance(stage, MossPatchStageSpec)
        )

    @property
    def transformer_layer_count(self) -> int:
        return sum(stage.num_layers for stage in self.transformer_stages)

    @classmethod
    def from_config(cls, config: Any) -> "MossCodecDecoderContract":
        decoder_kwargs = _get_config_value(config, "decoder_kwargs")
        if decoder_kwargs is None:
            raise MossCodecDecoderContractError(
                "MOSS codec decoder config is missing decoder_kwargs"
            )
        if not isinstance(decoder_kwargs, Sequence) or isinstance(
            decoder_kwargs, (str, bytes)
        ):
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder_kwargs must be a sequence, got "
                f"{type(decoder_kwargs).__name__}"
            )

        stages: list[CodecDecoderStageSpec] = []
        for stage_index, raw_stage in enumerate(decoder_kwargs):
            if not isinstance(raw_stage, Mapping):
                raise MossCodecDecoderContractError(
                    f"MOSS codec decoder stage {stage_index} must be a mapping, "
                    f"got {type(raw_stage).__name__}"
                )
            module_type = str(raw_stage.get("module_type", ""))
            if module_type == "Transformer":
                stages.append(_parse_transformer_stage(stage_index, raw_stage))
            elif module_type == "PatchedPretransform":
                stages.append(_parse_patch_stage(stage_index, raw_stage))
            else:
                raise MossCodecDecoderContractError(
                    f"unsupported MOSS codec decoder stage {stage_index} "
                    f"module_type={module_type!r}"
                )
        contract = cls(stages=tuple(stages))
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.n_vq != MOSS_TTS_LOCAL_N_VQ:
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder n_vq must be {MOSS_TTS_LOCAL_N_VQ}, "
                f"got {self.n_vq}"
            )
        if len(self.stages) != 12:
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder must have 12 stages, got {len(self.stages)}"
            )
        if len(self.transformer_stages) != 6 or len(self.patch_stages) != 6:
            raise MossCodecDecoderContractError(
                "MOSS codec decoder must have 6 transformer stages and 6 patch stages"
            )
        for index, stage in enumerate(self.stages):
            expected_transformer = index % 2 == 0
            if expected_transformer and not isinstance(stage, MossTransformerStageSpec):
                raise MossCodecDecoderContractError(
                    f"MOSS codec decoder stage {index} must be a transformer stage"
                )
            if not expected_transformer and not isinstance(stage, MossPatchStageSpec):
                raise MossCodecDecoderContractError(
                    f"MOSS codec decoder stage {index} must be a patch stage"
                )
            if stage.stage_index != index:
                raise MossCodecDecoderContractError(
                    f"MOSS codec decoder stage index mismatch: expected {index}, "
                    f"got {stage.stage_index}"
                )

        for offset, stage in enumerate(self.transformer_stages):
            expected = {
                "input_dimension": EXPECTED_TRANSFORMER_INPUT_DIMS[offset],
                "d_model": EXPECTED_TRANSFORMER_MODEL_DIMS[offset],
                "output_dimension": EXPECTED_TRANSFORMER_OUTPUT_DIMS[offset],
                "num_layers": EXPECTED_TRANSFORMER_LAYER_COUNTS[offset],
                "num_heads": EXPECTED_TRANSFORMER_HEADS[offset],
                "dim_feedforward": EXPECTED_TRANSFORMER_FFN_DIMS[offset],
            }
            for field_name, expected_value in expected.items():
                actual = getattr(stage, field_name)
                if actual != expected_value:
                    raise MossCodecDecoderContractError(
                        f"MOSS codec decoder stage {stage.stage_index} "
                        f"{field_name} must be {expected_value}, got {actual}"
                    )
            expected_context = EXPECTED_TRANSFORMER_CONTEXT_DURATIONS[offset]
            if stage.context_duration != expected_context:
                raise MossCodecDecoderContractError(
                    f"MOSS codec decoder stage {stage.stage_index} "
                    f"context_duration must be {expected_context}, "
                    f"got {stage.context_duration}"
                )
            _validate_transformer_semantics(stage)

        patch_factors = tuple(stage.patch_size for stage in self.patch_stages)
        if patch_factors != EXPECTED_PATCH_FACTORS:
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder patch factors must be "
                f"{EXPECTED_PATCH_FACTORS}, got {patch_factors}"
            )
        if self.transformer_layer_count != 92:
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder must have 92 transformer layers, "
                f"got {self.transformer_layer_count}"
            )

    def validate_module_tree(self, decoder: Any) -> None:
        """Validate representative module semantics from the remote-code decoder."""
        children = list(decoder) if isinstance(decoder, Sequence) else None
        if children is None:
            try:
                children = list(decoder.children())
            except AttributeError as exc:
                raise MossCodecDecoderContractError(
                    "MOSS codec decoder module must expose ordered children"
                ) from exc
        if len(children) != len(self.stages):
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder module must have {len(self.stages)} stages, "
                f"got {len(children)}"
            )
        for spec, module in zip(self.stages, children):
            class_name = type(module).__name__
            if isinstance(spec, MossPatchStageSpec):
                _validate_patch_module(spec, module)
                continue
            if "ProjectedTransformer" not in class_name:
                raise MossCodecDecoderContractError(
                    f"MOSS codec decoder stage {spec.stage_index} must be "
                    f"ProjectedTransformer, got {class_name}"
                )
            _validate_projected_transformer_module(spec, module)

    def validate_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Validate representative tensor shapes for every transformer stage."""
        for stage in self.transformer_stages:
            prefix = f"{stage.stage_index}"
            _require_shape(
                state_dict,
                f"{prefix}.input_proj.weight",
                (stage.d_model, stage.input_dimension),
            )
            _require_shape(
                state_dict,
                f"{prefix}.output_proj.weight",
                (stage.output_dimension, stage.d_model),
            )
            for layer in range(stage.num_layers):
                layer_prefix = f"{prefix}.transformer.layers.{layer}"
                _require_shape(
                    state_dict,
                    f"{layer_prefix}.self_attn.in_proj.weight",
                    (3 * stage.d_model, stage.d_model),
                )
                _require_shape(
                    state_dict,
                    f"{layer_prefix}.self_attn.out_proj.weight",
                    (stage.d_model, stage.d_model),
                )
                _require_shape(
                    state_dict,
                    f"{layer_prefix}.ffn.0.weight",
                    (stage.dim_feedforward, stage.d_model),
                )
                _require_shape(
                    state_dict,
                    f"{layer_prefix}.ffn.2.weight",
                    (stage.d_model, stage.dim_feedforward),
                )
                for norm in ("norm1", "norm2"):
                    _require_shape(
                        state_dict,
                        f"{layer_prefix}.{norm}.weight",
                        (stage.d_model,),
                    )
                    _require_shape(
                        state_dict,
                        f"{layer_prefix}.{norm}.bias",
                        (stage.d_model,),
                    )
                for scale in ("layer_scale_1", "layer_scale_2"):
                    _require_shape(
                        state_dict,
                        f"{layer_prefix}.{scale}.scale",
                        (stage.d_model,),
                    )


def build_moss_codec_decoder_contract(config: Any) -> MossCodecDecoderContract:
    return MossCodecDecoderContract.from_config(config)


def plan_moss_nonstream_decode_chunks(
    frames: int,
    *,
    max_chunk_frames: int = MOSS_TTS_LOCAL_MAX_DECODE_CHUNK_FRAMES,
) -> list[int]:
    frames = int(frames)
    max_chunk_frames = int(max_chunk_frames)
    if frames < 0:
        raise ValueError(f"frames must be >= 0, got {frames}")
    if max_chunk_frames <= 0:
        raise ValueError(f"max_chunk_frames must be > 0, got {max_chunk_frames}")
    return [
        min(max_chunk_frames, frames - start)
        for start in range(0, frames, max_chunk_frames)
    ]


def _parse_transformer_stage(
    stage_index: int, raw_stage: Mapping[str, Any]
) -> MossTransformerStageSpec:
    return MossTransformerStageSpec(
        stage_index=stage_index,
        input_dimension=_required_int(raw_stage, "input_dimension", stage_index),
        d_model=_required_int(raw_stage, "d_model", stage_index),
        output_dimension=_required_int(raw_stage, "output_dimension", stage_index),
        num_layers=_required_int(raw_stage, "num_layers", stage_index),
        num_heads=_required_int(raw_stage, "num_heads", stage_index),
        dim_feedforward=_required_int(raw_stage, "dim_feedforward", stage_index),
        context_duration=_required_float(raw_stage, "context_duration", stage_index),
        causal=_required_bool(raw_stage, "causal", stage_index),
        conv_layout=_required_bool(raw_stage, "conv_layout", stage_index),
        norm=str(_required(raw_stage, "norm", stage_index)),
        gating=str(_required(raw_stage, "gating", stage_index)),
        positional_embedding=str(
            _required(raw_stage, "positional_embedding", stage_index)
        ),
        layer_scale=_required_float(raw_stage, "layer_scale", stage_index),
        max_period=_required_int(raw_stage, "max_period", stage_index),
    )


def _parse_patch_stage(
    stage_index: int, raw_stage: Mapping[str, Any]
) -> MossPatchStageSpec:
    return MossPatchStageSpec(
        stage_index=stage_index,
        patch_size=_required_int(raw_stage, "patch_size", stage_index),
    )


def _validate_transformer_semantics(stage: MossTransformerStageSpec) -> None:
    if not stage.causal:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage.stage_index} must be causal"
        )
    if not stage.conv_layout:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage.stage_index} must use conv_layout"
        )
    if stage.norm != "layer_norm":
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage.stage_index} norm must be "
            f"'layer_norm', got {stage.norm!r}"
        )
    if stage.gating != "none":
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage.stage_index} gating must be "
            f"'none', got {stage.gating!r}"
        )
    if stage.positional_embedding != "rope":
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage.stage_index} positional_embedding "
            f"must be 'rope', got {stage.positional_embedding!r}"
        )
    if stage.d_model % stage.num_heads != 0 or stage.head_dim != 64:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage.stage_index} must use head_dim=64, "
            f"got d_model={stage.d_model}, num_heads={stage.num_heads}"
        )
    if stage.layer_scale != 0.01:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage.stage_index} layer_scale must be "
            f"0.01, got {stage.layer_scale}"
        )
    if stage.max_period != 10000:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage.stage_index} max_period must be "
            f"10000, got {stage.max_period}"
        )


def _validate_patch_module(spec: MossPatchStageSpec, module: Any) -> None:
    class_name = type(module).__name__
    if "PatchedPretransform" not in class_name:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} must be "
            f"PatchedPretransform, got {class_name}"
        )
    patch_size = int(getattr(module, "patch_size", -1))
    if patch_size != spec.patch_size:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} patch_size must be "
            f"{spec.patch_size}, got {patch_size}"
        )
    if bool(getattr(module, "is_downsample", True)):
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} must decode, not downsample"
        )


def _validate_projected_transformer_module(
    spec: MossTransformerStageSpec, module: Any
) -> None:
    _validate_linear_module(
        module,
        "input_proj",
        in_features=spec.input_dimension,
        out_features=spec.d_model,
        require_bias=False,
    )
    _validate_linear_module(
        module,
        "output_proj",
        in_features=spec.d_model,
        out_features=spec.output_dimension,
        require_bias=False,
    )
    transformer = getattr(module, "transformer", None)
    if transformer is None:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} is missing transformer"
        )
    positional_embedding = getattr(transformer, "positional_embedding", None)
    if positional_embedding != spec.positional_embedding:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} positional_embedding "
            f"must be {spec.positional_embedding!r}, got {positional_embedding!r}"
        )
    max_period = float(getattr(transformer, "max_period", -1.0))
    if max_period != float(spec.max_period):
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} max_period must be "
            f"{spec.max_period}, got {max_period}"
        )
    layers = list(getattr(transformer, "layers", []))
    if len(layers) != spec.num_layers:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} transformer layers "
            f"must be {spec.num_layers}, got {len(layers)}"
        )
    for layer_index, layer in enumerate(layers):
        _validate_transformer_layer_module(spec, layer, layer_index=layer_index)


def _validate_transformer_layer_module(
    spec: MossTransformerStageSpec, layer: Any, *, layer_index: int
) -> None:
    for norm in ("norm1", "norm2"):
        module = getattr(layer, norm, None)
        if type(module).__name__ != "LayerNorm":
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder stage {spec.stage_index} layer {layer_index} "
                f"{norm} must be LayerNorm"
            )
        normalized_shape = tuple(getattr(module, "normalized_shape", ()))
        if normalized_shape != (spec.d_model,):
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder stage {spec.stage_index} layer {layer_index} "
                f"{norm} normalized_shape must be {(spec.d_model,)}, "
                f"got {normalized_shape}"
            )

    self_attn = getattr(layer, "self_attn", None)
    if self_attn is None:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} layer {layer_index} "
            "is missing self_attn"
        )
    embed_dim = int(getattr(self_attn, "embed_dim", -1))
    num_heads = int(getattr(self_attn, "num_heads", -1))
    causal = bool(getattr(self_attn, "causal", False))
    if embed_dim != spec.d_model or num_heads != spec.num_heads:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} layer {layer_index} "
            f"self_attn must use embed_dim={spec.d_model}, "
            f"num_heads={spec.num_heads}"
        )
    if causal != spec.causal:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} layer {layer_index} "
            f"self_attn causal must be {spec.causal}"
        )
    _validate_linear_module(
        self_attn,
        "in_proj",
        in_features=spec.d_model,
        out_features=3 * spec.d_model,
        require_bias=False,
    )
    _validate_linear_module(
        self_attn,
        "out_proj",
        in_features=spec.d_model,
        out_features=spec.d_model,
        require_bias=False,
    )

    ffn = list(getattr(layer, "ffn", []))
    if len(ffn) != 3 or type(ffn[1]).__name__ != "GELU":
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {spec.stage_index} layer {layer_index} "
            "ffn must be Linear -> GELU -> Linear"
        )
    _validate_linear_instance(
        ffn[0],
        in_features=spec.d_model,
        out_features=spec.dim_feedforward,
        require_bias=False,
        label=f"stage {spec.stage_index} layer {layer_index} ffn.0",
    )
    _validate_linear_instance(
        ffn[2],
        in_features=spec.dim_feedforward,
        out_features=spec.d_model,
        require_bias=False,
        label=f"stage {spec.stage_index} layer {layer_index} ffn.2",
    )

    for scale in ("layer_scale_1", "layer_scale_2"):
        module = getattr(layer, scale, None)
        tensor = getattr(module, "scale", None)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (
            spec.d_model,
        ):
            raise MossCodecDecoderContractError(
                f"MOSS codec decoder stage {spec.stage_index} layer {layer_index} "
                f"{scale}.scale must have shape {(spec.d_model,)}"
            )


def _validate_linear_module(
    parent: Any,
    name: str,
    *,
    in_features: int,
    out_features: int,
    require_bias: bool,
) -> None:
    module = getattr(parent, name, None)
    _validate_linear_instance(
        module,
        in_features=in_features,
        out_features=out_features,
        require_bias=require_bias,
        label=name,
    )


def _validate_linear_instance(
    module: Any,
    *,
    in_features: int,
    out_features: int,
    require_bias: bool,
    label: str,
) -> None:
    if type(module).__name__ != "Linear":
        raise MossCodecDecoderContractError(f"{label} must be Linear")
    actual_in = int(getattr(module, "in_features", -1))
    actual_out = int(getattr(module, "out_features", -1))
    if actual_in != in_features or actual_out != out_features:
        raise MossCodecDecoderContractError(
            f"{label} must be Linear({in_features}, {out_features}), "
            f"got Linear({actual_in}, {actual_out})"
        )
    has_bias = getattr(module, "bias", None) is not None
    if has_bias != require_bias:
        expected = "with bias" if require_bias else "without bias"
        actual = "with bias" if has_bias else "without bias"
        raise MossCodecDecoderContractError(f"{label} must be {expected}, got {actual}")


def _required(raw_stage: Mapping[str, Any], key: str, stage_index: int) -> Any:
    if key not in raw_stage:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage_index} is missing {key!r}"
        )
    return raw_stage[key]


def _required_int(raw_stage: Mapping[str, Any], key: str, stage_index: int) -> int:
    value = _required(raw_stage, key, stage_index)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage_index} {key!r} must be int, "
            f"got {value!r}"
        ) from exc


def _required_float(raw_stage: Mapping[str, Any], key: str, stage_index: int) -> float:
    value = _required(raw_stage, key, stage_index)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage_index} {key!r} must be float, "
            f"got {value!r}"
        ) from exc


def _required_bool(raw_stage: Mapping[str, Any], key: str, stage_index: int) -> bool:
    value = _required(raw_stage, key, stage_index)
    if not isinstance(value, bool):
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder stage {stage_index} {key!r} must be bool, "
            f"got {type(value).__name__}"
        )
    return value


def _get_config_value(config: Any, key: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(key)
    return getattr(config, key, None)


def _require_shape(
    state_dict: Mapping[str, Any], key: str, expected_shape: tuple[int, ...]
) -> None:
    if key not in state_dict:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder state_dict is missing {key!r}"
        )
    value = state_dict[key]
    shape = tuple(value.shape) if isinstance(value, torch.Tensor) else tuple(value)
    if shape != expected_shape:
        raise MossCodecDecoderContractError(
            f"MOSS codec decoder state_dict {key!r} must have shape "
            f"{expected_shape}, got {shape}"
        )


__all__ = [
    "CodecDecoderStageSpec",
    "MOSS_TTS_LOCAL_FRAME_RATE",
    "MOSS_TTS_LOCAL_FRAME_SAMPLES",
    "MOSS_TTS_LOCAL_MAX_DECODE_CHUNK_FRAMES",
    "MOSS_TTS_LOCAL_N_VQ",
    "MOSS_TTS_LOCAL_SAMPLE_RATE",
    "MossCodecDecoderContract",
    "MossCodecDecoderContractError",
    "MossPatchStageSpec",
    "MossTransformerStageSpec",
    "build_moss_codec_decoder_contract",
    "plan_moss_nonstream_decode_chunks",
]
