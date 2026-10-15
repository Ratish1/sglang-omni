# SPDX-License-Identifier: Apache-2.0
"""Stateful incremental decoder for the Qwen3-TTS speech tokenizer."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang_omni.utils.channels_last_conv import (
    channels_last_weight,
    is_channels_last_conv_device,
)
from sglang_omni.utils.incremental_codec import (
    IncrementalCodecState,
    IncrementalCodecStateSpec,
    apply_activation,
    causal_conv1d,
    causal_transconv1d,
    frames_ending_at,
    incremental_convnext,
    incremental_residual_unit,
)

if TYPE_CHECKING:
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2CausalConvNet,
        Qwen3TTSTokenizerV2CausalTransConvNet,
        Qwen3TTSTokenizerV2Decoder,
        Qwen3TTSTokenizerV2DecoderAttention,
        Qwen3TTSTokenizerV2DecoderTransformerModel,
    )
else:
    pass


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    def rotate_half(value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def repeat_kv(hidden_states: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return hidden_states
    else:
        pass
    batch, heads, length, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, heads, groups, length, head_dim
    )
    return hidden_states.reshape(batch, heads * groups, length, head_dim)


def incremental_attention(
    attention: "Qwen3TTSTokenizerV2DecoderAttention",
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    state: IncrementalCodecState,
    layer_index: int,
    key_positions: torch.Tensor,
    query_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    head_dim = int(attention.head_dim)
    query = attention.q_norm(attention.q_proj(hidden_states)).view(
        *input_shape, -1, head_dim
    )
    key = attention.k_norm(attention.k_proj(hidden_states)).view(
        *input_shape, -1, head_dim
    )
    value = attention.v_proj(hidden_states).view(*input_shape, -1, head_dim)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    query, key = apply_rotary_pos_emb(
        query, key, position_embeddings[0], position_embeddings[1]
    )

    prior_key = state.transformer_keys.get(layer_index)
    prior_value = state.transformer_values.get(layer_index)
    if (prior_key is None) != (prior_value is None):
        raise RuntimeError(f"incomplete transformer state for layer {layer_index}")
    else:
        pass
    if prior_key is not None:
        key = torch.cat((prior_key, key), dim=-2)
        value = torch.cat((prior_value, value), dim=-2)
    else:
        pass

    if int(key.shape[-2]) != int(key_positions.shape[-1]):
        raise RuntimeError(
            f"incremental transformer state length {int(key.shape[-2])} does not "
            f"match {int(key_positions.shape[-1])} key positions"
        )
    else:
        pass
    repeated_key = repeat_kv(key, int(attention.num_key_value_groups))
    repeated_value = repeat_kv(value, int(attention.num_key_value_groups))
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * float(
        attention.scaling
    )
    keys_by_row = key_positions.unsqueeze(1)
    queries_by_row = query_positions.unsqueeze(2)
    allowed = keys_by_row <= queries_by_row
    sliding_window = int(attention.sliding_window)
    if sliding_window > 0:
        allowed &= keys_by_row > (queries_by_row - sliding_window)
    else:
        pass
    # Note (Qihao Liu): a right-aligned K/V buffer that a stream has not filled
    # yet holds zeros whose nominal absolute position is negative. Those slots
    # would otherwise satisfy both tests above for an early query, so mask them
    # explicitly. This is what lets a cold and a warm stream share one cohort.
    allowed &= (key_positions >= 0).unsqueeze(1)
    scores = scores.masked_fill(
        ~allowed.unsqueeze(1),
        torch.finfo(scores.dtype).min,
    )
    probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probabilities, repeated_value)
    output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return attention.o_proj(output), key, value


def incremental_transformer(
    transformer: "Qwen3TTSTokenizerV2DecoderTransformerModel",
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    valid_frames: torch.Tensor | None = None,
) -> torch.Tensor:
    hidden_states = transformer.input_proj(hidden_states)
    batch_size = int(hidden_states.shape[0])
    fresh_frames = int(hidden_states.shape[1])
    device = hidden_states.device
    frame_positions = state.row_frame_positions(batch_size, device)
    # Note (Qihao Liu): the retained buffer is right-aligned: slot j of a buffer
    # of length P holds absolute position frame_position - P + j. Deriving P
    # from the buffer rather than from transformer_context_length keeps this
    # correct when an arena hands over a full-width buffer that a cold stream
    # has not filled.
    prior_key = state.transformer_keys.get(0)
    prior_length = 0 if prior_key is None else int(prior_key.shape[-2])
    key_offsets = torch.arange(
        -prior_length, fresh_frames, device=device, dtype=torch.long
    )
    query_offsets = torch.arange(fresh_frames, device=device, dtype=torch.long)
    key_positions = frame_positions.unsqueeze(1) + key_offsets.unsqueeze(0)
    query_positions = frame_positions.unsqueeze(1) + query_offsets.unsqueeze(0)
    position_embeddings = transformer.rotary_emb(hidden_states, query_positions)

    next_keys: dict[int, torch.Tensor] = {}
    next_values: dict[int, torch.Tensor] = {}
    window_size = int(transformer.window_size)
    retained_context = max(0, window_size - 1)
    for layer_index, layer in enumerate(transformer.layers):
        residual = hidden_states
        normalized = layer.input_layernorm(hidden_states)
        attended, key, value = incremental_attention(
            layer.self_attn,
            normalized,
            position_embeddings,
            state,
            layer_index,
            key_positions,
            query_positions,
        )
        hidden_states = residual + layer.self_attn_layer_scale(attended)
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + layer.mlp_layer_scale(hidden_states)
        if valid_frames is not None:
            retained_end = prior_length + valid_frames
            next_keys[layer_index] = frames_ending_at(
                key, retained_end, retained_context, -2
            )
            next_values[layer_index] = frames_ending_at(
                value, retained_end, retained_context, -2
            )
        else:
            next_keys[layer_index] = (
                key[..., -retained_context:, :].clone()
                if retained_context
                else key[..., :0, :].clone()
            )
            next_values[layer_index] = (
                value[..., -retained_context:, :].clone()
                if retained_context
                else value[..., :0, :].clone()
            )

    state.transformer_keys = next_keys
    state.transformer_values = next_values
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
        conv: list[tuple[str, int, int]] = []
        transconv: list[tuple[str, int, int]] = []

        def add_conv(module: "Qwen3TTSTokenizerV2CausalConvNet", key: str) -> None:
            conv.append((key, int(module.conv.in_channels), int(module.padding)))

        def add_transconv(
            module: "Qwen3TTSTokenizerV2CausalTransConvNet", key: str
        ) -> None:
            transconv.append(
                (key, int(module.conv.out_channels), int(module.right_pad))
            )

        add_conv(decoder.pre_conv, "pre_conv")
        for stage_index, blocks in enumerate(decoder.upsample):
            add_transconv(blocks[0], f"upsample.{stage_index}.transconv")
            add_conv(blocks[1].dwconv, f"upsample.{stage_index}.convnext.dwconv")
        add_conv(decoder.decoder[0], "decoder.0")
        for block_index, decoder_block in enumerate(decoder.decoder[1:-2], start=1):
            add_transconv(decoder_block.block[1], f"decoder.{block_index}.transconv")
            for residual_index, residual in enumerate(decoder_block.block[2:]):
                key = f"decoder.{block_index}.residual.{residual_index}"
                add_conv(residual.conv1, f"{key}.conv1")
                add_conv(residual.conv2, f"{key}.conv2")
        add_conv(decoder.decoder[-1], "decoder.final")

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
        """Allocate a zeroed state with full-width buffers.

        Note (Qihao Liu): unlike the lazily grown state, the Transformer K/V
        buffers start at their retained width. A stream that has not filled
        them yet reads zeros at negative nominal positions, which
        ``_incremental_attention`` masks out, so a cold and a warm stream share
        one execution shape.
        """
        spec = self.state_spec()
        state = IncrementalCodecState(
            frame_positions=torch.zeros(batch_size, device=device, dtype=torch.long),
            transformer_context_length=spec.retained_context,
        )
        for key, channels, length in spec.conv_histories:
            state.conv_histories[key] = torch.zeros(
                batch_size, channels, length, device=device, dtype=dtype
            )
        for key, channels, length in spec.transconv_overlaps:
            state.transconv_overlaps[key] = torch.zeros(
                batch_size, channels, length, device=device, dtype=dtype
            )
        for layer_index in range(spec.num_layers):
            shape = (
                batch_size,
                spec.num_key_value_heads,
                spec.retained_context,
                spec.head_dim,
            )
            state.transformer_keys[layer_index] = torch.zeros(
                shape, device=device, dtype=dtype
            )
            state.transformer_values[layer_index] = torch.zeros(
                shape, device=device, dtype=dtype
            )
        return state

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
        channels_last_weights = self.channels_last_weights
        is_channels_last = channels_last_weights is not None
        for stage_index, blocks in enumerate(self.decoder.upsample):
            if len(blocks) != 2:
                raise TypeError("unsupported Qwen3-TTS upsample layout")
            else:
                pass
            hidden_states = causal_transconv1d(
                blocks[0],
                hidden_states,
                state,
                f"upsample.{stage_index}.transconv",
                channels_last_weights,
            )
            hidden_states = incremental_convnext(
                blocks[1],
                hidden_states,
                state,
                f"upsample.{stage_index}.convnext",
                is_channels_last,
            )

        waveform = causal_conv1d(
            self.decoder.decoder[0],
            hidden_states,
            state,
            "decoder.0",
            channels_last_weights,
        )
        for block_index, decoder_block in enumerate(
            self.decoder.decoder[1:-2], start=1
        ):
            if len(decoder_block.block) < 2:
                raise TypeError("unsupported Qwen3-TTS decoder block layout")
            else:
                pass
            waveform = apply_activation(
                decoder_block.block[0], waveform, is_channels_last
            )
            waveform = causal_transconv1d(
                decoder_block.block[1],
                waveform,
                state,
                f"decoder.{block_index}.transconv",
                channels_last_weights,
            )
            for residual_index, residual_unit in enumerate(decoder_block.block[2:]):
                waveform = incremental_residual_unit(
                    residual_unit,
                    waveform,
                    state,
                    f"decoder.{block_index}.residual.{residual_index}",
                    channels_last_weights,
                )
        waveform = apply_activation(
            self.decoder.decoder[-2], waveform, is_channels_last
        )
        waveform = causal_conv1d(
            self.decoder.decoder[-1],
            waveform,
            state,
            "decoder.final",
            channels_last_weights,
        )
        if is_channels_last:
            waveform = waveform.transpose(1, 2)
        else:
            pass
        return waveform.clamp(min=-1, max=1)

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
