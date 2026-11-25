# SPDX-License-Identifier: Apache-2.0
"""Request-owned convolution and projection for batched audio encoding."""

import torch
from torch.nn import functional as F
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
)


def encode_request_frontends(
    tower: Qwen3OmniMoeAudioEncoder,
    padded_features: torch.Tensor,
    audio_feature_lengths: torch.Tensor,
    request_clip_counts: list[int],
    valid_indices: torch.Tensor,
    output_frames: int,
) -> torch.Tensor:
    request_lengths = [
        lengths.tolist() for lengths in audio_feature_lengths.split(request_clip_counts)
    ]
    chunk_frames = tower.n_window * 2
    request_chunks = [
        sum((length + chunk_frames - 1) // chunk_frames for length in lengths)
        for lengths in request_lengths
    ]
    request_outputs = []
    for request_features, lengths in zip(
        padded_features.split(request_chunks), request_lengths, strict=True
    ):
        width = min(max(lengths), chunk_frames)
        if width != padded_features.shape[-1]:
            request_features = (
                request_features[:, :, :width]
                .transpose(1, 2)
                .contiguous()
                .transpose(1, 2)
            )
        else:
            pass
        convolution_outputs = []
        for chunk in request_features.unsqueeze(1).split(tower.conv_chunksize, dim=0):
            hidden_states = F.gelu(tower.conv2d1(chunk))
            hidden_states = F.gelu(tower.conv2d2(hidden_states))
            convolution_outputs.append(F.gelu(tower.conv2d3(hidden_states)))
        convolution_output = torch.cat(convolution_outputs)
        chunks, channels, frequencies, frames = convolution_output.shape
        hidden_states = tower.conv_out(
            convolution_output.permute(0, 3, 1, 2)
            .contiguous()
            .view(chunks, frames, channels * frequencies)
        )
        hidden_states = hidden_states + tower.positional_embedding.positional_embedding[
            :frames
        ].unsqueeze(0).to(hidden_states.dtype)
        if frames != output_frames:
            hidden_states = F.pad(hidden_states, (0, 0, 0, output_frames - frames))
        else:
            pass
        request_outputs.append(hidden_states)
    padded_output = torch.cat(request_outputs)
    return torch.index_select(
        padded_output.reshape(-1, padded_output.shape[-1]), 0, valid_indices
    )
