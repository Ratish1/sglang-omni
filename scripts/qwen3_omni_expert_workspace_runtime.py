"""Decode the pinned BF16 expert workspace with output-byte validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from scripts.qwen3_omni_talker_expert_runtime import combine_expert_contributions

# note (ratish): these are the pinned workspace.h alignment and prologue block bounds.
WORKSPACE_ALIGNMENT_BYTES = 128
MINIMUM_BLOCK_TOKENS = 32
MAXIMUM_BLOCK_TOKENS = 1024


@dataclass(kw_only=True, frozen=True)
class WorkspaceBuffer:
    offset_bytes: int
    size_bytes: int


@dataclass(kw_only=True)
class WorkspaceStages:
    activated: torch.Tensor
    projected: torch.Tensor
    has_glu_intermediate: bool


def workspace_prefix(
    tokens: int,
    hidden_channels: int,
    intermediate_channels: int,
    experts: int,
    ranks: int,
) -> dict[str, WorkspaceBuffer]:
    block_tokens = MINIMUM_BLOCK_TOKENS
    while (
        block_tokens < MAXIMUM_BLOCK_TOKENS
        and ((tokens + block_tokens - 1) // block_tokens) * experts > block_tokens
    ):
        block_tokens *= 2
    blocks = (tokens + block_tokens - 1) // block_tokens
    expanded_tokens = tokens * ranks
    sizes = {
        "permuted_row_to_unpermuted_row": expanded_tokens * 4,
        "permuted_token_selected_experts": expanded_tokens * 4,
        "blocked_expert_counts": experts * blocks * 4,
        "blocked_expert_counts_cumsum": experts * blocks * 4,
        "blocked_row_to_unpermuted_row": experts * tokens * 4,
        "expert_first_token_offset": (experts + 1) * 8,
        "permuted_token_final_scales": 0,
        "overlapped_gemm1_gemm2_inputs": expanded_tokens
        * max(hidden_channels, intermediate_channels)
        * 2,
        "overlapped_gemm1_gemm2_outputs": expanded_tokens
        * max(hidden_channels, 2 * intermediate_channels)
        * 2,
    }
    buffers = {}
    offset_bytes = 0
    for name, size_bytes in sizes.items():
        buffers[name] = WorkspaceBuffer(
            offset_bytes=offset_bytes, size_bytes=size_bytes
        )
        offset_bytes += (
            (size_bytes + WORKSPACE_ALIGNMENT_BYTES - 1)
            // WORKSPACE_ALIGNMENT_BYTES
            * WORKSPACE_ALIGNMENT_BYTES
        )
    return buffers


def workspace_view(
    workspace: torch.Tensor,
    buffer: WorkspaceBuffer,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    stop_bytes = buffer.offset_bytes + buffer.size_bytes
    assert stop_bytes <= workspace.numel()
    return workspace[buffer.offset_bytes : stop_bytes].view(dtype).reshape(shape)


def read_workspace_stages(
    workspace: torch.Tensor,
    buffers: dict[str, WorkspaceBuffer],
    indices: torch.Tensor,
    scales: torch.Tensor,
    token_rows: torch.Tensor,
    expert_ranks: torch.Tensor,
    intermediate_channels: int,
    output: torch.Tensor,
) -> WorkspaceStages:
    tokens, hidden_channels = output.shape
    ranks = indices.shape[1]
    expanded_tokens = tokens * ranks
    matches = []
    for has_glu_intermediate in (True, False):
        activation_name, projection_name = (
            ("overlapped_gemm1_gemm2_inputs", "overlapped_gemm1_gemm2_outputs")
            if has_glu_intermediate
            else ("overlapped_gemm1_gemm2_outputs", "overlapped_gemm1_gemm2_inputs")
        )
        activated = workspace_view(
            workspace,
            WorkspaceBuffer(
                offset_bytes=buffers[activation_name].offset_bytes,
                size_bytes=expanded_tokens * intermediate_channels * 2,
            ),
            torch.bfloat16,
            (expanded_tokens, intermediate_channels),
        )
        projected = workspace_view(
            workspace,
            WorkspaceBuffer(
                offset_bytes=buffers[projection_name].offset_bytes,
                size_bytes=expanded_tokens * hidden_channels * 2,
            ),
            torch.bfloat16,
            (expanded_tokens, hidden_channels),
        )
        routed = torch.empty(
            (tokens, ranks, hidden_channels), dtype=output.dtype, device=output.device
        )
        routed[token_rows, expert_ranks] = projected
        reconstructed = combine_expert_contributions(
            list(routed.unbind(dim=1)), indices, scales, "checkpoint-reduction"
        )
        if torch.equal(reconstructed.view(torch.uint8), output.view(torch.uint8)):
            matches.append(
                WorkspaceStages(
                    activated=activated,
                    projected=projected,
                    has_glu_intermediate=has_glu_intermediate,
                )
            )
        else:
            pass
    assert len(matches) == 1, ("workspace alias uniqueness", len(matches))
    return matches[0]
