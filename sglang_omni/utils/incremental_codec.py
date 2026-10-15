# SPDX-License-Identifier: Apache-2.0
"""Per-stream state and causal conv steps shared by the incremental codec decoders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import torch
import torch.nn.functional as F

from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
)


class CausalConv(Protocol):
    """A stride-1 causal conv: the conv and the history length it pads on the left."""

    conv: torch.nn.Conv1d
    padding: int


class CausalTransConv(Protocol):
    """A causal transposed conv: the conv and the samples it trims on the right."""

    conv: torch.nn.ConvTranspose1d
    right_pad: int


class ConvNeXtBlock(Protocol):
    dwconv: CausalConv
    norm: torch.nn.LayerNorm
    pwconv1: torch.nn.Linear
    act: torch.nn.Module
    pwconv2: torch.nn.Linear
    gamma: torch.nn.Parameter


class DecoderResidualUnit(Protocol):
    act1: torch.nn.Module
    conv1: CausalConv
    act2: torch.nn.Module
    conv2: CausalConv


@dataclass(frozen=True)
class IncrementalCodecStateSpec:
    """Static description of one stream's incremental state.

    conv_histories and transconv_overlaps are (key, channels, length) triples in the order
    the decoder visits them. Zero-length entries are kept so every consumer sees the same
    key set regardless of kernel size.
    """

    conv_histories: tuple[tuple[str, int, int], ...]
    transconv_overlaps: tuple[tuple[str, int, int], ...]
    num_layers: int
    num_key_value_heads: int
    head_dim: int
    retained_context: int

    def bytes_per_stream(self, dtype: torch.dtype) -> int:
        itemsize = torch.empty((), dtype=dtype).element_size()
        elements = sum(channels * length for _, channels, length in self.conv_histories)
        elements += sum(
            channels * length for _, channels, length in self.transconv_overlaps
        )
        elements += (
            2
            * self.num_layers
            * self.num_key_value_heads
            * self.retained_context
            * self.head_dim
        )
        return int(elements * itemsize)


@dataclass
class IncrementalCodecState:
    frame_position: int = 0
    transformer_context_length: int = 0
    transformer_keys: dict[int, torch.Tensor] = field(default_factory=dict)
    transformer_values: dict[int, torch.Tensor] = field(default_factory=dict)
    conv_histories: dict[str, torch.Tensor] = field(default_factory=dict)
    transconv_overlaps: dict[str, torch.Tensor] = field(default_factory=dict)
    # Note (Qihao Liu): absolute frame position per batch row. None means
    # every row shares frame_position, which is the single-request case.
    # A cohort assembled from a state arena sets this so streams at different
    # playback positions can execute in one launch.
    frame_positions: torch.Tensor | None = None

    def row_frame_positions(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.frame_positions is None:
            return torch.full(
                (batch_size,), self.frame_position, device=device, dtype=torch.long
            )
        else:
            pass
        if int(self.frame_positions.shape[0]) != batch_size:
            raise ValueError(
                "incremental codec state carries "
                f"{int(self.frame_positions.shape[0])} row positions for a batch "
                f"of {batch_size}"
            )
        else:
            pass
        return self.frame_positions.to(device=device, dtype=torch.long)

    def advance(self, fresh_frames: int) -> None:
        self.frame_position += fresh_frames
        if self.frame_positions is not None:
            self.frame_positions = self.frame_positions + fresh_frames
        else:
            pass

    def clone(self) -> IncrementalCodecState:
        return IncrementalCodecState(
            frame_position=self.frame_position,
            transformer_context_length=self.transformer_context_length,
            frame_positions=(
                None if self.frame_positions is None else self.frame_positions.clone()
            ),
            transformer_keys={
                key: value.clone() for key, value in self.transformer_keys.items()
            },
            transformer_values={
                key: value.clone() for key, value in self.transformer_values.items()
            },
            conv_histories={
                key: value.clone() for key, value in self.conv_histories.items()
            },
            transconv_overlaps={
                key: value.clone() for key, value in self.transconv_overlaps.items()
            },
        )


class IncrementalCodecDecoder(Protocol):
    """What a state arena needs from a decoder: its state layout and a zeroed state."""

    def state_spec(self) -> IncrementalCodecStateSpec: ...

    def init_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> IncrementalCodecState: ...


def conv_history(
    state: IncrementalCodecState,
    key: str,
    hidden_states: torch.Tensor,
    batch_size: int,
    channels: int,
    history_size: int,
) -> torch.Tensor:
    """The conv's (B, C, history) state, zeros before the stream's first decode."""
    history = state.conv_histories.get(key)
    if history is None:
        return hidden_states.new_zeros(batch_size, channels, history_size)
    elif history.shape[:-1] != (batch_size, channels):
        raise ValueError(f"incremental causal Conv1d state shape changed for {key}")
    else:
        return history


def incremental_causal_conv1d(
    module: CausalConv,
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    key: str,
) -> torch.Tensor:
    conv = module.conv
    stride = int(conv.stride[0])
    if stride != 1:
        raise ValueError(f"incremental causal Conv1d requires stride=1, got {stride}")
    else:
        pass
    history_size = int(module.padding)
    history = conv_history(
        state,
        key,
        hidden_states,
        int(hidden_states.shape[0]),
        int(hidden_states.shape[1]),
        history_size,
    )

    combined = torch.cat((history, hidden_states), dim=-1)
    output = conv(combined).contiguous()
    if output.shape[-1] != hidden_states.shape[-1]:
        raise RuntimeError(
            f"incremental causal Conv1d changed temporal length for {key}"
        )
    else:
        pass
    state.conv_histories[key] = (
        combined[..., -history_size:].clone()
        if history_size
        else combined[..., :0].clone()
    )
    return output


def incremental_causal_transconv1d(
    module: CausalTransConv,
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    key: str,
) -> torch.Tensor:
    conv = module.conv
    stride = int(conv.stride[0])
    right_pad = int(module.right_pad)
    output = F.conv_transpose1d(
        hidden_states,
        conv.weight,
        bias=None,
        stride=conv.stride,
        padding=conv.padding,
        output_padding=conv.output_padding,
        groups=conv.groups,
        dilation=conv.dilation,
    )
    overlap = state.transconv_overlaps.get(key)
    if overlap is not None:
        if overlap.shape[:-1] != output.shape[:-1]:
            raise ValueError(
                f"incremental causal ConvTranspose1d state shape changed for {key}"
            )
        else:
            pass
        overlap_length = int(overlap.shape[-1])
        output[..., :overlap_length] += overlap
    else:
        pass

    emit_length = int(hidden_states.shape[-1]) * stride
    emitted = output[..., :emit_length]
    tail = output[..., emit_length:]
    if int(tail.shape[-1]) != right_pad:
        raise RuntimeError(
            f"incremental causal ConvTranspose1d produced the wrong overlap for {key}"
        )
    else:
        pass
    state.transconv_overlaps[key] = tail.clone()
    if conv.bias is not None:
        emitted = emitted + conv.bias.view(1, -1, 1)
    else:
        pass
    return emitted.contiguous()


def channels_last_causal_conv1d(
    module: CausalConv,
    weight: torch.Tensor,
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    key: str,
) -> torch.Tensor:
    """incremental_causal_conv1d of a (B, L, C) activation with its channels-last
    weight; the history keeps its (B, C, history) layout."""
    conv = module.conv
    stride = int(conv.stride[0])
    if stride != 1:
        raise ValueError(f"incremental causal Conv1d requires stride=1, got {stride}")
    else:
        pass
    history_size = int(module.padding)
    batch_size, length, channels = hidden_states.shape
    history = conv_history(
        state, key, hidden_states, batch_size, channels, history_size
    )

    if history_size:
        combined = torch.cat((history.transpose(1, 2), hidden_states), dim=1)
    else:
        combined = hidden_states
    dilation = int(conv.dilation[0])
    if dilation == 1:
        padded = combined
    else:
        padded = F.pad(combined, (0, 0, 0, (-combined.shape[1]) % dilation))
    output = channels_last_conv1d(padded, conv, weight, length)
    if output.shape[1] != length:
        raise RuntimeError(
            f"incremental causal Conv1d changed temporal length for {key}"
        )
    else:
        pass
    state.conv_histories[key] = (
        combined[:, combined.shape[1] - history_size :]
        .transpose(1, 2)
        .clone(memory_format=torch.contiguous_format)
    )
    return output.contiguous()


def channels_last_causal_transconv1d(
    module: CausalTransConv,
    weight: torch.Tensor,
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    key: str,
) -> torch.Tensor:
    """incremental_causal_transconv1d of a (B, L, C) activation with its channels-last
    weight; the overlap keeps its (B, C, overlap) layout."""
    conv = module.conv
    stride = int(conv.stride[0])
    right_pad = int(module.right_pad)
    output = channels_last_conv_transpose1d(hidden_states, conv, weight, None)
    overlap = state.transconv_overlaps.get(key)
    if overlap is not None:
        if overlap.shape[:-1] != (output.shape[0], output.shape[2]):
            raise ValueError(
                f"incremental causal ConvTranspose1d state shape changed for {key}"
            )
        else:
            pass
        output[:, : int(overlap.shape[-1])] += overlap.transpose(1, 2)
    else:
        pass

    emit_length = int(hidden_states.shape[1]) * stride
    emitted = output[:, :emit_length]
    tail = output[:, emit_length:]
    if int(tail.shape[1]) != right_pad:
        raise RuntimeError(
            f"incremental causal ConvTranspose1d produced the wrong overlap for {key}"
        )
    else:
        pass
    state.transconv_overlaps[key] = tail.transpose(1, 2).clone(
        memory_format=torch.contiguous_format
    )
    if conv.bias is not None:
        emitted = emitted + conv.bias
    else:
        pass
    return emitted.contiguous()


def causal_conv1d(
    module: CausalConv,
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    key: str,
    channels_last_weights: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if channels_last_weights is None:
        return incremental_causal_conv1d(module, hidden_states, state, key)
    else:
        return channels_last_causal_conv1d(
            module, channels_last_weights[key], hidden_states, state, key
        )


def causal_transconv1d(
    module: CausalTransConv,
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    key: str,
    channels_last_weights: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    if channels_last_weights is None:
        return incremental_causal_transconv1d(module, hidden_states, state, key)
    else:
        return channels_last_causal_transconv1d(
            module, channels_last_weights[key], hidden_states, state, key
        )


def apply_activation(
    module: torch.nn.Module, hidden_states: torch.Tensor, is_channels_last: bool
) -> torch.Tensor:
    """Run a (B, C, L) activation module on the decoder's layout."""
    if is_channels_last:
        return module(hidden_states.transpose(1, 2)).transpose(1, 2)
    else:
        return module(hidden_states)


def frames_ending_at(
    sequence: torch.Tensor, end_frames: torch.Tensor, count: int, dim: int
) -> torch.Tensor:
    """Each row's count entries of sequence along dim that end at end_frames[row]."""
    index = (end_frames - count).unsqueeze(1) + torch.arange(
        count, device=sequence.device
    )
    shape = [1] * sequence.ndim
    shape[0], shape[dim] = int(sequence.shape[0]), count
    expanded = list(sequence.shape)
    expanded[dim] = count
    return sequence.gather(dim, index.view(shape).expand(expanded))


def incremental_convnext(
    module: ConvNeXtBlock,
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    key: str,
    is_channels_last: bool,
) -> torch.Tensor:
    residual = hidden_states
    if is_channels_last:
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel,
        # faster than cuDNN's channels-last depthwise and free of its per-shape setup.
        hidden_states = incremental_causal_conv1d(
            module.dwconv, hidden_states.transpose(1, 2), state, f"{key}.dwconv"
        ).transpose(1, 2)
    else:
        hidden_states = incremental_causal_conv1d(
            module.dwconv, hidden_states, state, f"{key}.dwconv"
        ).permute(0, 2, 1)
    hidden_states = module.norm(hidden_states)
    hidden_states = module.pwconv1(hidden_states)
    hidden_states = module.act(hidden_states)
    hidden_states = module.pwconv2(hidden_states)
    hidden_states = module.gamma * hidden_states
    if is_channels_last:
        return residual + hidden_states
    else:
        return residual + hidden_states.permute(0, 2, 1)


def incremental_residual_unit(
    module: DecoderResidualUnit,
    hidden_states: torch.Tensor,
    state: IncrementalCodecState,
    key: str,
    channels_last_weights: dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    is_channels_last = channels_last_weights is not None
    residual = hidden_states
    hidden_states = apply_activation(module.act1, hidden_states, is_channels_last)
    hidden_states = causal_conv1d(
        module.conv1, hidden_states, state, f"{key}.conv1", channels_last_weights
    )
    hidden_states = apply_activation(module.act2, hidden_states, is_channels_last)
    hidden_states = causal_conv1d(
        module.conv2, hidden_states, state, f"{key}.conv2", channels_last_weights
    )
    return hidden_states + residual
