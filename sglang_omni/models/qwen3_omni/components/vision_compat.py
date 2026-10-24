# SPDX-License-Identifier: Apache-2.0
"""Transformers compatibility boundary for the Qwen3-Omni vision encoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling
from transformers.processing_utils import Unpack
from transformers.utils.generic import TransformersKwargs, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids

# note (ratish): below this many grid rows the per-row loop is faster (crossover measured on H20).
VECTORIZED_POS_EMBED_MIN_IMAGES = 6


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

    def fast_pos_embed_interpolate_from_list(
        self,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        """Bilinear position embeddings per grid row, in merge-block order, repeated per frame."""
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim
        # note (ratish): the model's device and dtype properties iterate its parameters on every read.
        dtype = self.pos_embed.weight.dtype
        device = self.pos_embed.weight.device

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=dtype)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def fast_pos_embed_interpolate_vectorized(
        self,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        """fast_pos_embed_interpolate_from_list over all grid rows at once, equal bit for bit."""
        num_grid_per_side = self.num_grid_per_side
        m = self.spatial_merge_size
        dtype = self.pos_embed.weight.dtype
        device = self.pos_embed.weight.device

        ts = [g[0] for g in grid_thw]
        hs = [g[1] for g in grid_thw]
        ws = [g[2] for g in grid_thw]
        num_images = len(grid_thw)

        hw_list = [h * w for h, w in zip(hs, ws)]
        thw_list = [t * s for t, s in zip(ts, hw_list)]
        total_hw = sum(hw_list)
        total_out = sum(thw_list)

        def exclusive_prefix(sizes: list[int]) -> torch.Tensor:
            out, acc = [], 0
            for s in sizes:
                out.append(acc)
                acc += s
            return torch.tensor(out, device=device, dtype=torch.long)

        hw_off = exclusive_prefix(hw_list)
        thw_off = exclusive_prefix(thw_list)
        image_arange = torch.arange(num_images, device=device)

        base_image_id = torch.repeat_interleave(
            image_arange, torch.tensor(hw_list, device=device)
        )
        base_local = torch.arange(total_hw, device=device) - hw_off[base_image_id]
        w_of = torch.tensor(ws, device=device)[base_image_id]
        row = base_local // w_of
        col = base_local % w_of

        uniq_h, inv_h = torch.unique(
            torch.tensor(hs, device=device), return_inverse=True
        )
        uniq_w, inv_w = torch.unique(
            torch.tensor(ws, device=device), return_inverse=True
        )
        h_luts = [
            torch.linspace(0, num_grid_per_side - 1, h, device=device)
            for h in uniq_h.tolist()
        ]
        w_luts = [
            torch.linspace(0, num_grid_per_side - 1, w, device=device)
            for w in uniq_w.tolist()
        ]
        h_lut_off = exclusive_prefix([len(x) for x in h_luts])
        w_lut_off = exclusive_prefix([len(x) for x in w_luts])
        h_idxs = torch.cat(h_luts)[h_lut_off[inv_h[base_image_id]] + row]
        w_idxs = torch.cat(w_luts)[w_lut_off[inv_w[base_image_id]] + col]

        h_floor = h_idxs.to(torch.long)
        w_floor = w_idxs.to(torch.long)
        h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
        w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor
        w11 = dh * dw
        w10 = dh - w11
        w01 = dw - w11
        w00 = 1 - dh - w01

        base_h = h_floor * num_grid_per_side
        base_h_ceil = h_ceil * num_grid_per_side
        indices = torch.stack(
            [
                base_h + w_floor,
                base_h + w_ceil,
                base_h_ceil + w_floor,
                base_h_ceil + w_ceil,
            ],
            dim=0,
        )
        weights = torch.stack([w00, w01, w10, w11], dim=0).to(dtype=dtype)
        embeds = self.pos_embed(indices) * weights[:, :, None]
        base_embeds = embeds.sum(dim=0)

        out_image_id = torch.repeat_interleave(
            image_arange, torch.tensor(thw_list, device=device)
        )
        pos_in_image = torch.arange(total_out, device=device) - thw_off[out_image_id]
        hw_of_out = torch.tensor(hw_list, device=device)[out_image_id]
        frame_idx = pos_in_image // hw_of_out
        local_idx = pos_in_image % hw_of_out
        patch = base_embeds[hw_off[out_image_id] + local_idx]

        all_w = torch.tensor(ws, device=device)[out_image_id]
        rows = local_idx // all_w
        cols = local_idx % all_w
        out_within = (
            frame_idx * hw_of_out
            + ((rows // m) * (all_w // m) + (cols // m)) * m * m
            + (rows % m) * m
            + (cols % m)
        )
        merged = torch.empty_like(patch)
        merged[out_within + thw_off[out_image_id]] = patch
        return merged

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
        grid_thw_list = host_grid_thw.tolist()
        if len(grid_thw_list) >= VECTORIZED_POS_EMBED_MIN_IMAGES:
            pos_embeds = self.fast_pos_embed_interpolate_vectorized(grid_thw_list)
        else:
            pos_embeds = self.fast_pos_embed_interpolate_from_list(grid_thw_list)
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
