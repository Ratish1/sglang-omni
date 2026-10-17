# SPDX-License-Identifier: Apache-2.0
"""The code2wav conv stack decoded incrementally from per-stream slots in a codec state pool:
each causal conv's padded input takes its history from the rows' slots and gives back the
newest, and each transposed conv adds the slots' overlap and stores the samples the next frames
still add to, both exchanged in place."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.kernels.codec_state import history_exchange, overlap_exchange
from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
)


def conv_stack_state_layout(
    codec: Qwen3OmniMoeCode2Wav,
) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
    """The conv stack's (key, channels, length) conv histories and transposed-conv overlaps,
    in the order slot_conv_stack visits them."""
    conv_histories: list[tuple[str, int, int]] = []
    transconv_overlaps: list[tuple[str, int, int]] = []

    def add_conv(module: Qwen3OmniMoeCausalConvNet, state_key: str) -> None:
        conv_histories.append((state_key, module.conv.in_channels, module.padding))

    def add_transconv(module: Qwen3OmniMoeCausalTransConvNet, state_key: str) -> None:
        transconv_overlaps.append(
            (state_key, module.conv.out_channels, module.right_pad)
        )

    for stage_index, blocks in enumerate(codec.upsample):
        add_transconv(blocks[0], f"upsample.{stage_index}.transconv")
        add_conv(blocks[1].dwconv, f"upsample.{stage_index}.convnext.dwconv")
    add_conv(codec.decoder[0], "decoder.0")
    for block_index, decoder_block in enumerate(codec.decoder[1:-2], start=1):
        add_transconv(decoder_block.block[1], f"decoder.{block_index}.transconv")
        for residual_index, residual in enumerate(decoder_block.block[2:]):
            residual_key = f"decoder.{block_index}.residual.{residual_index}"
            add_conv(residual.conv1, f"{residual_key}.conv1")
            add_conv(residual.conv2, f"{residual_key}.conv2")
    add_conv(codec.decoder[-1], "decoder.final")
    return conv_histories, transconv_overlaps


@dataclass(frozen=True, kw_only=True)
class SlotDecode:
    """The pool slots a decode's rows read and write, and whether every row starts its stream."""

    pool: CodecStatePool
    slot_index: torch.Tensor
    is_cold: bool


def slot_causal_conv(
    module: Qwen3OmniMoeCausalConvNet,
    frames: torch.Tensor,
    slot_decode: SlotDecode,
    state_key: str,
) -> torch.Tensor:
    """The causal conv's (B, L, C_out) outputs for the rows' new (B, L, C) frames, its padded
    input's history exchanged with the slots in place."""
    conv = module.conv
    histories = slot_decode.pool.histories
    batch_size, length, channels = frames.shape
    history_samples = module.padding
    tail_samples = (-length) % conv.dilation[0]
    padded = frames.new_empty(
        batch_size, history_samples + length + tail_samples, channels
    )
    padded[:, history_samples : history_samples + length].copy_(frames)
    if state_key in histories:
        history_exchange(
            padded,
            histories[state_key],
            slot_decode.slot_index,
            new_samples=length,
            is_cold=slot_decode.is_cold,
        )
    else:
        padded[:, history_samples + length :].zero_()
    return channels_last_conv1d(padded, conv, conv.weight, length)


def slot_causal_transconv(
    module: Qwen3OmniMoeCausalTransConvNet,
    frames: torch.Tensor,
    slot_decode: SlotDecode,
    state_key: str,
    *,
    left_trim: int,
) -> torch.Tensor:
    """The transposed conv's emitted samples after left_trim, the slots' overlap added in
    front and the samples the next frames add to kept; the bias follows the overlap, so
    overlapping samples carry it once."""
    conv = module.conv
    overlaps = slot_decode.pool.overlaps
    output = channels_last_conv_transpose1d(frames, conv, conv.weight, None)
    emitted_samples = frames.shape[1] * conv.stride[0]
    if state_key in overlaps:
        overlap_exchange(
            output,
            overlaps[state_key],
            slot_decode.slot_index,
            emitted_samples=emitted_samples,
            is_cold=slot_decode.is_cold,
        )
    else:
        pass
    return output[:, left_trim:emitted_samples] + conv.bias


def channels_last_activation(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """A (B, C, L) activation module run on (B, L, C) activations."""
    return module(hidden_states.transpose(1, 2)).transpose(1, 2)


def slot_conv_stack(
    codec: Qwen3OmniMoeCode2Wav,
    hidden_states: torch.Tensor,
    slot_decode: SlotDecode,
) -> torch.Tensor:
    """Everything after the transformer, from its (B, L, C) outputs to the clamped (B, 1,
    samples) waveform, on channels-last activations. A stream's first decode drops each
    transposed conv's left pad, as the whole-sequence forward does."""
    for stage_index, blocks in enumerate(codec.upsample):
        transconv, convnext = blocks[0], blocks[1]
        hidden_states = slot_causal_transconv(
            transconv,
            hidden_states,
            slot_decode,
            f"upsample.{stage_index}.transconv",
            left_trim=transconv.left_pad if slot_decode.is_cold else 0,
        )
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel, faster
        # than cuDNN's channels-last depthwise and free of its per-shape setup.
        batch_size, length, channels = hidden_states.shape
        history_samples = convnext.dwconv.padding
        depthwise_input = hidden_states.new_empty(
            batch_size, channels, history_samples + length
        )
        depthwise_input[:, :, history_samples:].copy_(hidden_states.transpose(1, 2))
        history_exchange(
            depthwise_input.transpose(1, 2),
            slot_decode.pool.histories[f"upsample.{stage_index}.convnext.dwconv"],
            slot_decode.slot_index,
            new_samples=length,
            is_cold=slot_decode.is_cold,
        )
        normed = convnext.norm(convnext.dwconv.conv(depthwise_input).transpose(1, 2))
        hidden_states = hidden_states + convnext.gamma * convnext.pwconv2(
            convnext.act(convnext.pwconv1(normed))
        )
    hidden_states = slot_causal_conv(
        codec.decoder[0], hidden_states, slot_decode, "decoder.0"
    )
    for block_index, decoder_block in enumerate(codec.decoder[1:-2], start=1):
        transconv = decoder_block.block[1]
        hidden_states = slot_causal_transconv(
            transconv,
            channels_last_activation(decoder_block.block[0], hidden_states),
            slot_decode,
            f"decoder.{block_index}.transconv",
            left_trim=transconv.left_pad if slot_decode.is_cold else 0,
        )
        for residual_index, unit in enumerate(decoder_block.block[2:]):
            residual_key = f"decoder.{block_index}.residual.{residual_index}"
            output = slot_causal_conv(
                unit.conv1,
                channels_last_activation(unit.act1, hidden_states),
                slot_decode,
                f"{residual_key}.conv1",
            )
            output = slot_causal_conv(
                unit.conv2,
                channels_last_activation(unit.act2, output),
                slot_decode,
                f"{residual_key}.conv2",
            )
            hidden_states = output + hidden_states
    waveform = slot_causal_conv(
        codec.decoder[-1],
        channels_last_activation(codec.decoder[-2], hidden_states),
        slot_decode,
        "decoder.final",
    )
    return waveform.transpose(1, 2).clamp(min=-1, max=1)
