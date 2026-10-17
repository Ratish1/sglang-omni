# SPDX-License-Identifier: Apache-2.0
"""The code2wav conv stack decoded incrementally from per-stream slots in a codec state pool:
each producer writes its frames into the next causal conv's input, and the slots' histories and
overlaps are exchanged in place."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from sglang_omni.kernels.codec_state import history_exchange, overlap_exchange
from sglang_omni.scheduling.codec_state_pool import CodecStatePool
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
)
from sglang_omni.utils.snake_beta import fused_bias_snake_beta


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
    in the order its decode visits them."""
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


def padded_input(
    module: CausalConv, frames_like: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A causal conv's (B, history + L + tail, C) input for frames shaped as frames_like (B, L,
    C), and the rows its producer writes the frames into."""
    batch_size, length, channels = frames_like.shape
    history_samples = int(module.padding)
    tail_samples = (-(history_samples + length)) % int(module.conv.dilation[0])
    padded = frames_like.new_empty(
        batch_size, history_samples + length + tail_samples, channels
    )
    return (padded, padded[:, history_samples : history_samples + length])


def padded_conv(
    module: CausalConv,
    weight: torch.Tensor,
    padded: torch.Tensor,
    slots: SlotDecode,
    key: str,
    length: int,
    *,
    add_bias: bool,
) -> torch.Tensor:
    """The causal conv of a padded_input whose frames the producer wrote, with the slots'
    history in front and their newest history kept."""
    if key in slots.pool.histories:
        history_exchange(
            padded,
            slots.pool.histories[key],
            slots.slot_index,
            new_samples=length,
            is_cold=slots.is_cold,
        )
    else:
        pass
    return channels_last_conv1d(padded, module.conv, weight, length, add_bias=add_bias)


def transconv_frames(
    module: CausalTransConv,
    weight: torch.Tensor,
    hidden_states: torch.Tensor,
    slots: SlotDecode,
    key: str,
    *,
    left_trim: int,
) -> torch.Tensor:
    """A transposed conv's emitted (B, L, C) frames without its bias: the slots' overlap added in
    front, the samples the next frames still add to kept as their overlap, and left_trim samples
    dropped, as the whole-sequence forward drops them once at its start."""
    output = channels_last_conv_transpose1d(hidden_states, module.conv, weight, None)
    emitted_samples = int(hidden_states.shape[1]) * int(module.conv.stride[0])
    if key in slots.pool.overlaps:
        overlap_exchange(
            output,
            slots.pool.overlaps[key],
            slots.slot_index,
            emitted_samples=emitted_samples,
            is_cold=slots.is_cold,
        )
    else:
        pass
    return output[:, left_trim:emitted_samples]


def pool_conv_stack(
    codec: CodecConvStack,
    hidden_states: torch.Tensor,
    slots: SlotDecode,
    channels_last_weights: dict[str, torch.Tensor],
    *,
    first_decode: bool,
) -> torch.Tensor:
    """Everything after the transformer, from its (B, L, C) outputs to the clamped (B, 1,
    samples) waveform, on bf16 channels-last activations with every SnakeBeta fused. A stream's
    first decode drops each transposed conv's left pad, as the whole-sequence forward does.
    """
    stages = len(codec.upsample)
    for stage_index, blocks in enumerate(codec.upsample):
        transconv, convnext = blocks[0], blocks[1]
        frames = transconv_frames(
            transconv,
            channels_last_weights[f"upsample.{stage_index}.transconv"],
            hidden_states,
            slots,
            f"upsample.{stage_index}.transconv",
            left_trim=int(transconv.left_pad) if first_decode else 0,
        )
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel, faster
        # than cuDNN's channels-last depthwise and free of its per-shape setup.
        batch_size, length, channels = frames.shape
        history_samples = int(convnext.dwconv.padding)
        depthwise_input = frames.new_empty(
            batch_size, channels, history_samples + length
        )
        residual = depthwise_input[:, :, history_samples:].transpose(1, 2)
        torch.add(frames, transconv.conv.bias, out=residual)
        history_exchange(
            depthwise_input.transpose(1, 2),
            slots.pool.histories[f"upsample.{stage_index}.convnext.dwconv"],
            slots.slot_index,
            new_samples=length,
            is_cold=slots.is_cold,
        )
        hidden_states = convnext.dwconv.conv(depthwise_input).transpose(1, 2)
        hidden_states = convnext.pwconv2(
            convnext.act(convnext.pwconv1(convnext.norm(hidden_states)))
        )
        scaled = convnext.gamma * hidden_states
        if stage_index + 1 < stages:
            hidden_states = residual + scaled
        else:
            padded, frame_rows = padded_input(codec.decoder[0], scaled)
            torch.add(residual, scaled, out=frame_rows)
    hidden_states = padded_conv(
        codec.decoder[0],
        channels_last_weights["decoder.0"],
        padded,
        slots,
        "decoder.0",
        length,
        add_bias=True,
    )
    blocks = codec.decoder[1:-2]
    activated = blocks[0].block[0](hidden_states.transpose(1, 2)).transpose(1, 2)
    for block_index, decoder_block in enumerate(blocks, start=1):
        transconv = decoder_block.block[1]
        frames = transconv_frames(
            transconv,
            channels_last_weights[f"decoder.{block_index}.transconv"],
            activated,
            slots,
            f"decoder.{block_index}.transconv",
            left_trim=int(transconv.left_pad) if first_decode else 0,
        )
        units = decoder_block.block[2:]
        length = int(frames.shape[1])
        residual = frames.new_empty(frames.shape)
        padded, frame_rows = padded_input(units[0].conv1, frames)
        fused_bias_snake_beta(
            frames, transconv.conv.bias, units[0].act1, frame_rows, sum_out=residual
        )
        for residual_index, unit in enumerate(units):
            key = f"decoder.{block_index}.residual.{residual_index}"
            conv1_output = padded_conv(
                unit.conv1,
                channels_last_weights[f"{key}.conv1"],
                padded,
                slots,
                f"{key}.conv1",
                length,
                add_bias=False,
            )
            padded, frame_rows = padded_input(unit.conv2, frames)
            fused_bias_snake_beta(
                conv1_output, unit.conv1.conv.bias, unit.act2, frame_rows
            )
            conv2_output = padded_conv(
                unit.conv2,
                channels_last_weights[f"{key}.conv2"],
                padded,
                slots,
                f"{key}.conv2",
                length,
                add_bias=False,
            )
            if residual_index + 1 < len(units):
                next_unit = units[residual_index + 1]
                padded, frame_rows = padded_input(next_unit.conv1, frames)
                fused_bias_snake_beta(
                    conv2_output,
                    unit.conv2.conv.bias,
                    next_unit.act1,
                    frame_rows,
                    residual=residual,
                    sum_out=residual,
                )
            elif block_index < len(blocks):
                activated = conv2_output.new_empty(conv2_output.shape)
                fused_bias_snake_beta(
                    conv2_output,
                    unit.conv2.conv.bias,
                    blocks[block_index].block[0],
                    activated,
                    residual=residual,
                )
            else:
                padded, frame_rows = padded_input(codec.decoder[-1], frames)
                fused_bias_snake_beta(
                    conv2_output,
                    unit.conv2.conv.bias,
                    codec.decoder[-2],
                    frame_rows,
                    residual=residual,
                )
    waveform = padded_conv(
        codec.decoder[-1],
        channels_last_weights["decoder.final"],
        padded,
        slots,
        "decoder.final",
        length,
        add_bias=True,
    )
    return waveform.transpose(1, 2).clamp(min=-1, max=1)
