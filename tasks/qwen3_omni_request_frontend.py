# SPDX-License-Identifier: Apache-2.0
"""Request-owned audio frontend experiment with the native batched layer stack."""

from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.vision import prepare_vision_attention_metadata
from torch.nn import functional as F
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder


@dataclass(kw_only=True)
class AudioFrontend:
    hidden_states: torch.Tensor
    cumulative_sequence_lengths: torch.Tensor


def request_frontend(
    encoder: Qwen3OmniAudioEncoder,
    input_features: torch.Tensor,
    audio_feature_lengths: torch.Tensor,
) -> AudioFrontend:
    tower = encoder.audio_tower
    padded_features, chunk_lengths = hf_modeling.chunk_and_pad_features(
        input_features, audio_feature_lengths, tower.n_window
    )
    valid_indices = hf_modeling.get_valid_indices(chunk_lengths, tower.n_window).to(
        encoder.device, non_blocking=True
    )
    cumulative_sequence_lengths = hf_modeling.get_audio_cu_seqlens(
        chunk_lengths, audio_feature_lengths, tower.n_window_infer, tower.n_window
    )
    convolution_outputs = []
    for chunk in padded_features.unsqueeze(1).split(tower.conv_chunksize, dim=0):
        hidden_states = F.gelu(tower.conv2d1(chunk))
        hidden_states = F.gelu(tower.conv2d2(hidden_states))
        convolution_outputs.append(F.gelu(tower.conv2d3(hidden_states)))
    convolution_output = torch.cat(convolution_outputs, dim=0)
    chunks, channels, frequencies, frames = convolution_output.shape
    hidden_states = tower.conv_out(
        convolution_output.permute(0, 3, 1, 2)
        .contiguous()
        .view(chunks, frames, channels * frequencies)
    )
    hidden_states = hidden_states + tower.positional_embedding.positional_embedding[
        :frames
    ].unsqueeze(0).to(hidden_states.dtype)
    return AudioFrontend(
        hidden_states=torch.index_select(
            hidden_states.reshape(-1, hidden_states.shape[-1]), 0, valid_indices
        ),
        cumulative_sequence_lengths=cumulative_sequence_lengths,
    )


def batched_request_frontend(
    encoder: Qwen3OmniAudioEncoder,
    input_features: torch.Tensor,
    audio_feature_lengths: torch.Tensor,
    request_clip_counts: list[int],
) -> AudioFrontend:
    assert audio_feature_lengths.device.type == "cpu"
    assert sum(request_clip_counts) == audio_feature_lengths.numel()
    assert all(count > 0 for count in request_clip_counts)
    assert input_features.shape[-1] == int(audio_feature_lengths.sum())
    request_lengths = audio_feature_lengths.split(request_clip_counts)
    request_features = input_features.to(
        device=encoder.device, dtype=encoder.audio_tower.dtype
    ).split([int(lengths.sum()) for lengths in request_lengths], dim=-1)
    frontends = [
        request_frontend(encoder, features, lengths)
        for features, lengths in zip(request_features, request_lengths, strict=True)
    ]
    segment_lengths = torch.cat(
        [frontend.cumulative_sequence_lengths.diff() for frontend in frontends]
    )
    return AudioFrontend(
        hidden_states=torch.cat([frontend.hidden_states for frontend in frontends]),
        cumulative_sequence_lengths=F.pad(segment_lengths, (1, 0)).cumsum(
            0, dtype=torch.int32
        ),
    )


def request_frontend_forward(
    encoder: Qwen3OmniAudioEncoder,
    input_features: torch.Tensor,
    audio_feature_lengths: torch.Tensor,
    request_clip_counts: list[int],
) -> torch.Tensor:
    frontend = batched_request_frontend(
        encoder, input_features, audio_feature_lengths, request_clip_counts
    )
    segment_lengths = frontend.cumulative_sequence_lengths.diff().tolist()
    cumulative_sequence_lengths = frontend.cumulative_sequence_lengths.to(
        encoder.device, non_blocking=True
    )
    max_sequence_length = max(segment_lengths)
    encoder.segment_splits.value = segment_lengths
    encoder.segment_splits.attention_metadata = (
        prepare_vision_attention_metadata(
            cumulative_sequence_lengths,
            encoder.device,
            max_seqlen=max_sequence_length,
        )
        if encoder.packed_attention is not None
        else None
    )
    try:
        hidden_states = frontend.hidden_states
        for layer in encoder.audio_tower.layers:
            hidden_states = layer(
                hidden_states,
                cumulative_sequence_lengths,
                max_seqlen=max_sequence_length,
            )[0]
        tower = encoder.audio_tower
        return tower.proj2(tower.act(tower.proj1(tower.ln_post(hidden_states))))
    finally:
        encoder.segment_splits.value = None
        encoder.segment_splits.attention_metadata = None
