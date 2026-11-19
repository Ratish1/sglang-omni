# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostics for one native speech request."""

import os
from pathlib import Path

import torch

MAX_RECORDED_CODEC_FRAMES = 128


def is_witness(request_id: str) -> bool:
    return request_id == os.environ.get("OMNI_WITNESS_REQUEST_ID")


def record_boundary(
    request_id: str,
    boundary: str,
    fields: dict[str, torch.Tensor | float | int | list[int] | None],
    *,
    frame_index: int = 0,
) -> None:
    if not is_witness(request_id) or frame_index >= MAX_RECORDED_CODEC_FRAMES:
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
