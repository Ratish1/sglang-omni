# SPDX-License-Identifier: Apache-2.0
"""Bounded diagnostics for one native speech request."""

import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import torch

MAX_RECORDED_CODEC_FRAMES = 128


class WitnessControl(TypedDict):
    request_id: str
    output_directory: str
    norm_scope: Literal["none", "first", "decode"]
    held_input_directory: str | None


def witness_control(request_id: str) -> WitnessControl | None:
    control_file = os.environ.get("OMNI_WITNESS_BATCH_CONTROL")
    if control_file is not None:
        control = json.loads(Path(control_file).read_text())
        return control if control["request_id"] == request_id else None
    elif request_id == os.environ.get("OMNI_WITNESS_REQUEST_ID"):
        return {
            "request_id": request_id,
            "output_directory": os.environ["OMNI_WITNESS_DIRECTORY"],
            "norm_scope": (
                "first"
                if os.environ.get("OMNI_WITNESS_SWAP_FIRST_NORM") == "1"
                else "none"
            ),
            "held_input_directory": os.environ.get("OMNI_WITNESS_HELD_INPUT_DIRECTORY"),
        }
    else:
        return None


def is_witness(request_id: str) -> bool:
    return witness_control(request_id) is not None


def record_boundary(
    request_id: str,
    boundary: str,
    fields: dict[str, torch.Tensor | float | int | list[int] | None],
    *,
    frame_index: int = 0,
) -> None:
    control = witness_control(request_id)
    if control is None or frame_index >= MAX_RECORDED_CODEC_FRAMES:
        return
    else:
        pass
    directory = Path(control["output_directory"])
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
