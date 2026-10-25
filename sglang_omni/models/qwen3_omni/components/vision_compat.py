# SPDX-License-Identifier: Apache-2.0
"""Transformers compatibility boundary for the Qwen3-Omni vision encoder."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids


@dataclass(frozen=True, kw_only=True)
class VisionRotaryInputs:
    cos_sin_cache: torch.Tensor
    positions: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class VisionSequenceMetadata:
    sequence_patch_counts: tuple[int, ...]


class Qwen3OmniMoeVisionEncoderCompat(hf_modeling.Qwen3OmniMoeVisionEncoder):
    """HF vision encoder with the bilinear position interpolation computed on its device."""

    def prepare_position_embeddings(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | VisionRotaryInputs:
        return cos, sin

    def prepare_attention_metadata(
        self,
        sequence_patch_counts: tuple[int, ...],
    ) -> dict[str, VisionSequenceMetadata]:
        """The extra attention keywords every block receives."""
        return {}

    def interpolate_position_embeddings(
        self,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        """Bilinear position embeddings of every grid, in merge-block order, repeated per frame."""
        grid_side = self.num_grid_per_side
        merge_size = self.spatial_merge_size
        # note (ratish): the model's device and dtype properties iterate its parameters on every read.
        table = self.pos_embed.weight
        device = table.device

        heights = sorted({height for _, height, _ in grid_thw})
        widths = sorted({width for _, _, width in grid_thw})
        height_offsets = dict(zip(heights, itertools.accumulate(heights, initial=0)))
        width_offsets = dict(zip(widths, itertools.accumulate(widths, initial=0)))
        grid_metadata = []
        total_frame_patches = 0
        total_output_patches = 0
        for frames, height, width in grid_thw:
            grid_metadata.append(
                [
                    height * width,
                    frames * height * width,
                    total_frame_patches,
                    total_output_patches,
                    width,
                    height_offsets[height],
                    width_offsets[width],
                ]
            )
            total_frame_patches += height * width
            total_output_patches += frames * height * width
        # note (ratish): every grid-derived value is built on the host and copied once; nothing is read back.
        (
            grid_frame_patches,
            grid_output_patches,
            grid_frame_offsets,
            grid_output_offsets,
            grid_widths,
            grid_height_offsets,
            grid_width_offsets,
        ) = (
            torch.tensor(grid_metadata, dtype=torch.long)
            .to(device, non_blocking=True)
            .unbind(1)
        )

        patch_grids = torch.repeat_interleave(
            grid_frame_patches, output_size=total_frame_patches
        )
        merged_indices = (
            torch.arange(total_frame_patches, device=device)
            - grid_frame_offsets[patch_grids]
        )
        blocks_per_row = grid_widths[patch_grids] // merge_size
        block_indices = merged_indices // (merge_size * merge_size)
        patch_rows = (block_indices // blocks_per_row) * merge_size + (
            merged_indices // merge_size % merge_size
        )
        patch_columns = (block_indices % blocks_per_row) * merge_size + (
            merged_indices % merge_size
        )
        height_coordinates = torch.cat(
            [
                torch.linspace(
                    0, grid_side - 1, height, dtype=torch.float32, device=device
                )
                for height in heights
            ]
        )[grid_height_offsets[patch_grids] + patch_rows]
        width_coordinates = torch.cat(
            [
                torch.linspace(
                    0, grid_side - 1, width, dtype=torch.float32, device=device
                )
                for width in widths
            ]
        )[grid_width_offsets[patch_grids] + patch_columns]

        height_floor = height_coordinates.to(torch.long)
        width_floor = width_coordinates.to(torch.long)
        height_ceil = torch.clamp(height_floor + 1, max=grid_side - 1)
        width_ceil = torch.clamp(width_floor + 1, max=grid_side - 1)
        height_fraction = height_coordinates - height_floor
        width_fraction = width_coordinates - width_floor
        weight_11 = height_fraction * width_fraction
        weight_10 = height_fraction - weight_11
        weight_01 = width_fraction - weight_11
        weight_00 = 1 - height_fraction - weight_01
        tap_indices = torch.stack(
            [
                height_floor * grid_side + width_floor,
                height_floor * grid_side + width_ceil,
                height_ceil * grid_side + width_floor,
                height_ceil * grid_side + width_ceil,
            ]
        )
        tap_weights = torch.stack([weight_00, weight_01, weight_10, weight_11])
        frame_embeddings = (
            self.pos_embed(tap_indices) * tap_weights.to(table.dtype)[:, :, None]
        ).sum(dim=0)

        output_grids = torch.repeat_interleave(
            grid_output_patches, output_size=total_output_patches
        )
        frame_indices = (
            torch.arange(total_output_patches, device=device)
            - grid_output_offsets[output_grids]
        ) % grid_frame_patches[output_grids]
        return frame_embeddings[grid_frame_offsets[output_grids] + frame_indices]

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | hf_modeling.BaseModelOutputWithDeepstackFeatures:
        """Run the HF encoder with the positional interpolation computed on its device."""
        # HF may pass its device-generated interpolation tensors through
        # kwargs. This compatibility path always recomputes them from the grid.
        kwargs.pop("interp_indices", None)
        kwargs.pop("interp_weights", None)
        # note (ratish): the rotary positions and attention lengths are derived on the host
        # from the grid; the device receives one copy of each and is never read back.
        host_grid_thw = grid_thw.cpu()
        position_ids = get_vision_position_ids(
            host_grid_thw, self.spatial_merge_size, kwargs=kwargs
        )
        host_cu_seqlens = get_vision_cu_seqlens(host_grid_thw, kwargs=kwargs)
        sequence_patch_counts = tuple(
            (host_cu_seqlens[1:] - host_cu_seqlens[:-1]).tolist()
        )
        max_seqlen = max(sequence_patch_counts)
        attention_metadata_arguments = self.prepare_attention_metadata(
            sequence_patch_counts
        )
        device = hidden_states.device
        cu_seqlens = host_cu_seqlens.to(device, non_blocking=True)
        position_ids = position_ids.to(device, non_blocking=True)

        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.interpolate_position_embeddings(host_grid_thw.tolist())
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
        cos, sin = self.rotary_pos_emb(hidden_states, position_ids)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        position_embeddings = self.prepare_position_embeddings(cos, sin)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
                **attention_metadata_arguments,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger_index = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_index](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
            else:
                pass

        merged_hidden_states = self.merger(hidden_states)

        return hf_modeling.BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_feature_lists,
        )
