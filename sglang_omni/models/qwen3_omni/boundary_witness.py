# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostics for one native speech request."""

import os
from pathlib import Path

import torch

MAX_RECORDED_CODEC_FRAMES = 128
FORCED_CODEC = os.environ.get("OMNI_WITNESS_FORCE_FIRST_DECODE_CODE")


def swap_first_decode_code(
    sampled: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    """Intervene on the held serial witness at its captured decode position."""
    if FORCED_CODEC is None or sampled.shape[0] != 1:
        return sampled
    else:
        pass
    return torch.where(positions == 136, int(FORCED_CODEC), sampled)


def record_boundary(
    request_id: str,
    boundary: str,
    fields: dict[str, torch.Tensor | int | list[int] | None],
    *,
    frame_index: int = 0,
) -> None:
    if (
        request_id != os.environ.get("OMNI_WITNESS_REQUEST_ID")
        or frame_index >= MAX_RECORDED_CODEC_FRAMES
    ):
        return
    else:
        pass
    directory = Path(os.environ["OMNI_WITNESS_DIRECTORY"])
    with (directory / f"{boundary}-{frame_index:03d}.pt").open("xb") as destination:
        torch.save(
            {
                name: (
                    value.detach().cpu().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                )
                for name, value in fields.items()
            },
            destination,
        )
