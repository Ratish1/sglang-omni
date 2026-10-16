# SPDX-License-Identifier: Apache-2.0
"""Stateful incremental decoder for the Qwen3-TTS speech tokenizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from sglang_omni.utils.channels_last_conv import (
    channels_last_weight,
    is_channels_last_conv_device,
)
from sglang_omni.utils.incremental_codec import (
    IncrementalCodecState,
    IncrementalCodecStateSpec,
    causal_conv1d,
    conv_stack_state_layout,
    frames_ending_at,
    incremental_conv_stack,
    incremental_transformer_layers,
    init_incremental_state,
)

if TYPE_CHECKING:
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Decoder,
        Qwen3TTSTokenizerV2DecoderTransformerModel,
    )
else:
    pass


def incremental_transformer(
    transformer: "Qwen3TTSTokenizerV2DecoderTransformerModel",
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    valid_frames: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden_states = incremental_transformer_layers(
        transformer, transformer.input_proj(hidden_states), state, valid_frames
    )
    return transformer.output_proj(transformer.norm(hidden_states))


class Qwen3TTSIncrementalDecoder:
    def __init__(self, decoder: "Qwen3TTSTokenizerV2Decoder") -> None:
        self.require_attrs(
            decoder,
            "decoder",
            "quantizer",
            "pre_conv",
            "pre_transformer",
            "upsample",
            "decoder",
            "total_upsample",
        )
        if len(decoder.decoder) < 3:
            raise TypeError("unsupported Qwen3-TTS decoder layout")
        else:
            pass
        self.require_attrs(decoder.pre_conv, "pre_conv", "conv", "padding")
        self.require_attrs(
            decoder.pre_transformer,
            "pre_transformer",
            "input_proj",
            "layers",
            "norm",
            "output_proj",
            "rotary_emb",
            "window_size",
        )
        for layer_index, layer in enumerate(decoder.pre_transformer.layers):
            self.require_attrs(
                layer,
                f"pre_transformer.layers.{layer_index}",
                "input_layernorm",
                "self_attn",
                "self_attn_layer_scale",
                "post_attention_layernorm",
                "mlp",
                "mlp_layer_scale",
            )
            self.require_attrs(
                layer.self_attn,
                f"pre_transformer.layers.{layer_index}.self_attn",
                "head_dim",
                "num_key_value_groups",
                "scaling",
                "sliding_window",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "q_norm",
                "k_norm",
            )
        for stage_index, blocks in enumerate(decoder.upsample):
            if len(blocks) != 2:
                raise TypeError("unsupported Qwen3-TTS upsample layout")
            else:
                pass
            self.require_attrs(
                blocks[0], f"upsample.{stage_index}.0", "conv", "right_pad"
            )
            self.require_attrs(
                blocks[1],
                f"upsample.{stage_index}.1",
                "dwconv",
                "norm",
                "pwconv1",
                "act",
                "pwconv2",
                "gamma",
            )
        self.require_attrs(decoder.decoder[0], "decoder.0", "conv", "padding")
        self.require_attrs(decoder.decoder[-1], "decoder.final", "conv", "padding")
        for block_index, block in enumerate(decoder.decoder[1:-2], start=1):
            if not hasattr(block, "block") or len(block.block) < 2:
                raise TypeError("unsupported Qwen3-TTS decoder block layout")
            else:
                pass
            self.require_attrs(
                block.block[1],
                f"decoder.{block_index}.1",
                "conv",
                "right_pad",
            )
            for residual_index, residual in enumerate(block.block[2:]):
                self.require_attrs(
                    residual,
                    f"decoder.{block_index}.{residual_index + 2}",
                    "act1",
                    "conv1",
                    "act2",
                    "conv2",
                )
        self.decoder = decoder
        self.total_upsample = int(decoder.total_upsample)
        # note (ratish): on CUDA the convs run channels last on weights kept here in
        # that layout; the shared modules keep theirs for the whole-utterance decode.
        if is_channels_last_conv_device(decoder.pre_conv.conv.weight.device):
            self.channels_last_weights: dict[str, torch.Tensor] | None = {
                "pre_conv": channels_last_weight(decoder.pre_conv.conv),
                "decoder.0": channels_last_weight(decoder.decoder[0].conv),
                "decoder.final": channels_last_weight(decoder.decoder[-1].conv),
            }
            for stage_index, blocks in enumerate(decoder.upsample):
                self.channels_last_weights[f"upsample.{stage_index}.transconv"] = (
                    channels_last_weight(blocks[0].conv)
                )
            for block_index, block in enumerate(decoder.decoder[1:-2], start=1):
                self.channels_last_weights[f"decoder.{block_index}.transconv"] = (
                    channels_last_weight(block.block[1].conv)
                )
                for residual_index, residual in enumerate(block.block[2:]):
                    key = f"decoder.{block_index}.residual.{residual_index}"
                    self.channels_last_weights[f"{key}.conv1"] = channels_last_weight(
                        residual.conv1.conv
                    )
                    self.channels_last_weights[f"{key}.conv2"] = channels_last_weight(
                        residual.conv2.conv
                    )
        else:
            self.channels_last_weights = None
        self._state_spec: IncrementalCodecStateSpec | None = (
            None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
        self.compiled_kernel: (
            Callable[[torch.Tensor, IncrementalCodecState], torch.Tensor] | None
        ) = None
        self.compiled_shapes: set[tuple[int, int]] = set()

    @staticmethod
    def require_attrs(module: object, path: str, *names: str) -> None:
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            raise TypeError(
                "unsupported Qwen3-TTS decoder layout; "
                f"{path} is missing {', '.join(missing)}"
            )
        else:
            pass

    def state_spec(self) -> IncrementalCodecStateSpec:
        """Describe one stream's state without running a decode.

        Note (Qihao Liu): the lazy buffers in ``decode`` only materialize once
        activations have flowed through, so a state arena cannot preallocate
        from them. This walks the validated module tree in the same order
        ``decode`` does and derives every key and shape statically.
        """
        if (
            self._state_spec is None
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            self._state_spec = (
                self.build_state_spec()
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        else:
            pass
        return (
            self._state_spec
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    def build_state_spec(self) -> IncrementalCodecStateSpec:
        decoder = self.decoder
        stack_conv, transconv = conv_stack_state_layout(decoder)
        pre_conv = decoder.pre_conv
        conv = [
            ("pre_conv", int(pre_conv.conv.in_channels), int(pre_conv.padding)),
            *stack_conv,
        ]

        transformer = decoder.pre_transformer
        attention = transformer.layers[0].self_attn
        head_dim = int(attention.head_dim)
        key_features = int(attention.k_proj.out_features)
        if key_features % head_dim:
            raise TypeError(
                "unsupported Qwen3-TTS decoder layout; k_proj width "
                f"{key_features} is not a multiple of head_dim {head_dim}"
            )
        else:
            pass
        return IncrementalCodecStateSpec(
            conv_histories=tuple(conv),
            transconv_overlaps=tuple(transconv),
            num_layers=len(transformer.layers),
            num_key_value_heads=key_features // head_dim,
            head_dim=head_dim,
            retained_context=max(0, int(transformer.window_size) - 1),
        )

    def init_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> IncrementalCodecState:
        return init_incremental_state(
            self.state_spec(), batch_size, device=device, dtype=dtype
        )

    def decode(
        self,
        codes: torch.Tensor,
        state: IncrementalCodecState,
        *,
        compiled: bool = False,
    ) -> torch.Tensor:
        if codes.ndim != 3:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires codes shaped [B, Q, T]"
            )
        else:
            pass
        if int(codes.shape[0]) < 1:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires at least one row"
            )
        else:
            pass
        fresh_frames = int(codes.shape[-1])
        if fresh_frames <= 0:
            raise ValueError(
                "Qwen3-TTS incremental codec decoding requires fresh frames"
            )
        else:
            pass
        # note (luojiaxuan): the compiled kernel is opt-in per call. Only the
        # graph runner asks for it, at capture time, on static buffers; every
        # eager decode stays off Dynamo, whose value guards would otherwise
        # recompile once per stream position and hit the recompile limit.
        shape = (int(codes.shape[0]), fresh_frames)
        if compiled and shape not in self.compiled_shapes:
            raise RuntimeError(
                f"Qwen3-TTS incremental codec shape {shape} was not precompiled"
            )
        else:
            pass
        kernel = self.compiled_kernel if compiled else self.decode_tensors
        waveform = kernel(codes, state)
        expected_samples = fresh_frames * self.total_upsample
        if int(waveform.shape[-1]) != expected_samples:
            raise RuntimeError(
                "Qwen3-TTS incremental codec decoder returned the wrong sample count"
            )
        else:
            pass
        state.transformer_context_length = min(
            self.state_spec().retained_context,
            state.transformer_context_length + fresh_frames,
        )
        state.advance(fresh_frames)
        return waveform

    def tail_frames(self, emit_frames: int) -> int:
        """How many transformer outputs the last emit_frames frames' samples and the conv-stack
        state depend on, walked back from the waveform: a causal conv adds its history, and a
        transposed conv of kernel k and stride s maps n samples to (n + k - 1) // s inputs.
        """
        decoder = self.decoder
        samples = int(emit_frames) * self.total_upsample + int(
            decoder.decoder[-1].padding
        )
        for decoder_block in reversed(decoder.decoder[1:-2]):
            for residual_unit in reversed(decoder_block.block[2:]):
                samples += int(residual_unit.conv2.padding) + int(
                    residual_unit.conv1.padding
                )
            transconv = decoder_block.block[1].conv
            samples = (samples + int(transconv.kernel_size[0]) - 1) // int(
                transconv.stride[0]
            )
        samples += int(decoder.decoder[0].padding)
        for blocks in reversed(decoder.upsample):
            samples += int(blocks[1].dwconv.padding)
            transconv = blocks[0].conv
            samples = (samples + int(transconv.kernel_size[0]) - 1) // int(
                transconv.stride[0]
            )
        return samples

    def decode_tail(
        self,
        codes: torch.Tensor,
        valid_frames: torch.Tensor,
        state: IncrementalCodecState,
        emit_frames: int,
    ) -> torch.Tensor:
        """Decode row i's valid_frames[i] frames with the conv stack on the last
        tail_frames(emit_frames) outputs only, and return the last emit_frames frames' samples.
        Needs arena state (per-row positions, full-width K/V), tail_frames <= valid_frames[i].
        """
        if state.frame_positions is None:
            raise ValueError("Qwen3-TTS tail decoding needs per-row frame positions")
        else:
            pass
        tail_frames = self.tail_frames(emit_frames)
        hidden_states = self.frame_tensors(codes, state, valid_frames)
        time_dim = 1 if self.channels_last_weights is not None else 2
        window = frames_ending_at(hidden_states, valid_frames, tail_frames, time_dim)
        waveform = self.conv_stack_tensors(window, state)
        state.transformer_context_length = self.state_spec().retained_context
        state.frame_positions = state.frame_positions + valid_frames
        return waveform[..., -int(emit_frames) * self.total_upsample :]

    def decode_tensors(
        self,
        codes: torch.Tensor,
        state: IncrementalCodecState,
    ) -> torch.Tensor:
        """The tensor-only decode step: every op here is traceable and capturable.

        The frame counter and retained-context bookkeeping live in ``decode``:
        they are the Python scalars that change every step, and inside the
        traced region each distinct value would force a recompile. With
        channels_last_weights, activations flow as (B, L, C) from the quantizer
        to the final conv.
        """
        return self.conv_stack_tensors(self.frame_tensors(codes, state), state)

    def frame_tensors(
        self,
        codes: torch.Tensor,
        state: IncrementalCodecState,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Quantizer, pre_conv and transformer in the decoder's layout; with valid_frames,
        row i's pre_conv history and kept K/V end at valid_frames[i]."""
        channels_last_weights = self.channels_last_weights
        is_channels_last = channels_last_weights is not None
        hidden_states = self.decoder.quantizer.decode(codes)
        if is_channels_last:
            hidden_states = hidden_states.transpose(1, 2)
        else:
            pass
        prior_history = (
            state.conv_histories["pre_conv"] if valid_frames is not None else None
        )
        quantized = hidden_states
        hidden_states = causal_conv1d(
            self.decoder.pre_conv,
            hidden_states,
            state,
            "pre_conv",
            channels_last_weights,
        )
        if valid_frames is not None:
            history_size = int(prior_history.shape[-1])
            inputs = quantized.transpose(1, 2) if is_channels_last else quantized
            state.conv_histories["pre_conv"] = frames_ending_at(
                torch.cat((prior_history, inputs), dim=-1),
                history_size + valid_frames,
                history_size,
                -1,
            )
        else:
            pass
        if is_channels_last:
            return incremental_transformer(
                self.decoder.pre_transformer, hidden_states, state, valid_frames
            )
        else:
            return incremental_transformer(
                self.decoder.pre_transformer,
                hidden_states.transpose(1, 2),
                state,
                valid_frames,
            ).permute(0, 2, 1)

    def conv_stack_tensors(
        self, hidden_states: torch.Tensor, state: IncrementalCodecState
    ) -> torch.Tensor:
        """Everything after the transformer, from its outputs to the clamped waveform."""
        return incremental_conv_stack(
            self.decoder,
            hidden_states,
            state,
            self.channels_last_weights,
            first_decode=False,
            fused_activations=False,
        )

    def precompile(self, codes: torch.Tensor, state: IncrementalCodecState) -> None:
        """Trace the tensor-only step on these inputs and admit their shape.

        ``decode`` routes a shape through the compiled kernel only after it was
        precompiled here, so an unforeseen shape at serving time takes the eager
        path instead of a multi-second compile on a request's critical path.

        Pass the codes and state of a real decode, in its grad-disabling
        context: Dynamo guards on both, and the step advances the state.
        """
        if self.compiled_kernel is None:
            self.compiled_kernel = torch.compile(
                self.decode_tensors, dynamic=False, fullgraph=True
            )
        else:
            pass
        shape = (int(codes.shape[0]), int(codes.shape[-1]))
        if shape in self.compiled_shapes:
            return
        else:
            pass
        # note (luojiaxuan): one trace per (batch, width) pair, past the default
        # 8; ``decode`` never compiles a new shape, so the limit only matters here.
        with torch._dynamo.config.patch(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            recompile_limit=max(
                torch._dynamo.config.recompile_limit, 64
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        ):
            self.compiled_kernel(codes, state)
        self.compiled_shapes.add(shape)
