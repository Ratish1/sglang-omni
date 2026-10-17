# SPDX-License-Identifier: Apache-2.0
"""The code2wav conv stack decoded incrementally from per-stream slots in a codec state pool:
each causal conv reads its history from the rows' slots and stores the newest, and each
transposed conv adds the slots' overlap and stores the samples the next frames still add to."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F

from sglang_omni.scheduling.codec_state_pool import CodecStatePool
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
)


class CausalConv(Protocol):
    """A stride-1 causal conv: the conv and the history length it pads on the left."""

    conv: torch.nn.Conv1d
    padding: int


class CausalTransConv(Protocol):
    """A causal transposed conv: the conv and the samples its whole-sequence forward trims
    from the start and the end."""

    conv: torch.nn.ConvTranspose1d
    left_pad: int
    right_pad: int


class CodecConvStack(Protocol):
    """The conv stack after the transformer: upsample stages of a transposed conv and a
    ConvNeXt block, then decoder.0, decoder blocks (activation, transposed conv, residual
    units), a final activation and a final conv."""

    upsample: torch.nn.ModuleList
    decoder: torch.nn.ModuleList


def conv_stack_state_layout(
    codec: CodecConvStack,
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """The conv stack's (key, channels, length) conv histories and transposed-conv overlaps,
    in the order its decode visits them. Qwen3-TTS's speech tokenizer decoder has the same conv
    stack and lays out its state from this too."""
    conv: list[tuple[str, int, int]] = []
    transconv: list[tuple[str, int, int]] = []

    def add_conv(module: CausalConv, key: str) -> None:
        conv.append((key, int(module.conv.in_channels), int(module.padding)))

    def add_transconv(module: CausalTransConv, key: str) -> None:
        transconv.append((key, int(module.conv.out_channels), int(module.right_pad)))

    for stage_index, blocks in enumerate(codec.upsample):
        add_transconv(blocks[0], f"upsample.{stage_index}.transconv")
        add_conv(blocks[1].dwconv, f"upsample.{stage_index}.convnext.dwconv")
    add_conv(codec.decoder[0], "decoder.0")
    for block_index, decoder_block in enumerate(codec.decoder[1:-2], start=1):
        add_transconv(decoder_block.block[1], f"decoder.{block_index}.transconv")
        for residual_index, residual in enumerate(decoder_block.block[2:]):
            key = f"decoder.{block_index}.residual.{residual_index}"
            add_conv(residual.conv1, f"{key}.conv1")
            add_conv(residual.conv2, f"{key}.conv2")
    add_conv(codec.decoder[-1], "decoder.final")
    return conv, transconv


@dataclass(frozen=True, kw_only=True)
class SlotDecode:
    """The pool slots a decode's rows read and write, and whether every row starts its stream."""

    pool: CodecStatePool
    slot_index: torch.Tensor
    is_cold: bool


def with_history(
    frames: torch.Tensor, slots: SlotDecode, key: str, history_samples: int
) -> torch.Tensor:
    """The (B, history + L, C) frames with each slot's history in front (zeros when the rows
    start their streams); the last history samples become the slots' history."""
    length = int(frames.shape[1])
    if history_samples == 0:
        return frames
    elif slots.is_cold:
        padded = F.pad(frames, (0, 0, history_samples, 0))
    else:
        history = slots.pool.histories[key].index_select(0, slots.slot_index)
        padded = torch.cat((history, frames), dim=1)
    slots.pool.histories[key].index_copy_(0, slots.slot_index, padded[:, length:])
    return padded


def slot_causal_conv(
    module: CausalConv, frames: torch.Tensor, slots: SlotDecode, key: str
) -> torch.Tensor:
    """The causal conv's (B, L, C_out) outputs for the rows' new (B, L, C) frames."""
    conv = module.conv
    length = int(frames.shape[1])
    padded = with_history(frames, slots, key, int(module.padding))
    padded = F.pad(padded, (0, 0, 0, (-length) % int(conv.dilation[0])))
    return channels_last_conv1d(padded, conv, conv.weight, length)


def slot_causal_transconv(
    module: CausalTransConv,
    frames: torch.Tensor,
    slots: SlotDecode,
    key: str,
    *,
    left_trim: int,
) -> torch.Tensor:
    """The transposed conv's emitted (B, L x stride - left_trim, C_out) samples: the slots'
    overlap added in front, the samples the next frames still add to kept as their overlap,
    and left_trim samples dropped, as the whole-sequence forward drops them once at its start.
    The bias is added after the overlap, so overlapping samples carry it once."""
    conv = module.conv
    output = channels_last_conv_transpose1d(frames, conv, conv.weight, None)
    emitted_samples = int(frames.shape[1]) * int(conv.stride[0])
    if key not in slots.pool.overlaps:
        pass
    elif slots.is_cold:
        slots.pool.overlaps[key].index_copy_(
            0, slots.slot_index, output[:, emitted_samples:]
        )
    else:
        overlap_samples = int(output.shape[1]) - emitted_samples
        output[:, :overlap_samples] += slots.pool.overlaps[key].index_select(
            0, slots.slot_index
        )
        slots.pool.overlaps[key].index_copy_(
            0, slots.slot_index, output[:, emitted_samples:]
        )
    return output[:, left_trim:emitted_samples] + conv.bias


def channels_last_activation(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """A (B, C, L) activation module run on (B, L, C) activations."""
    return module(hidden_states.transpose(1, 2)).transpose(1, 2)


def slot_conv_stack(
    codec: CodecConvStack,
    hidden_states: torch.Tensor,
    slots: SlotDecode,
    *,
    first_decode: bool,
) -> torch.Tensor:
    """Everything after the transformer, from its (B, L, C) outputs to the clamped (B, 1,
    samples) waveform, on channels-last activations. A stream's first decode drops each
    transposed conv's left pad, as the whole-sequence forward does."""
    for stage_index, blocks in enumerate(codec.upsample):
        transconv, convnext = blocks[0], blocks[1]
        hidden_states = slot_causal_transconv(
            transconv,
            hidden_states,
            slots,
            f"upsample.{stage_index}.transconv",
            left_trim=int(transconv.left_pad) if first_decode else 0,
        )
        padded = with_history(
            hidden_states,
            slots,
            f"upsample.{stage_index}.convnext.dwconv",
            int(convnext.dwconv.padding),
        )
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel, faster
        # than cuDNN's channels-last depthwise and free of its per-shape setup.
        normed = convnext.norm(
            convnext.dwconv.conv(padded.transpose(1, 2).contiguous()).transpose(1, 2)
        )
        hidden_states = hidden_states + convnext.gamma * convnext.pwconv2(
            convnext.act(convnext.pwconv1(normed))
        )
    hidden_states = slot_causal_conv(
        codec.decoder[0], hidden_states, slots, "decoder.0"
    )
    for block_index, decoder_block in enumerate(codec.decoder[1:-2], start=1):
        transconv = decoder_block.block[1]
        hidden_states = slot_causal_transconv(
            transconv,
            channels_last_activation(decoder_block.block[0], hidden_states),
            slots,
            f"decoder.{block_index}.transconv",
            left_trim=int(transconv.left_pad) if first_decode else 0,
        )
        for residual_index, unit in enumerate(decoder_block.block[2:]):
            key = f"decoder.{block_index}.residual.{residual_index}"
            output = slot_causal_conv(
                unit.conv1,
                channels_last_activation(unit.act1, hidden_states),
                slots,
                f"{key}.conv1",
            )
            output = slot_causal_conv(
                unit.conv2,
                channels_last_activation(unit.act2, output),
                slots,
                f"{key}.conv2",
            )
            hidden_states = output + hidden_states
    waveform = slot_causal_conv(
        codec.decoder[-1],
        channels_last_activation(codec.decoder[-2], hidden_states),
        slots,
        "decoder.final",
    )
    return waveform.transpose(1, 2).clamp(min=-1, max=1)
