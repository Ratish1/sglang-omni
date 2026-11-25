# SPDX-License-Identifier: Apache-2.0
"""Record selected conditioning boundaries without replacing their operands."""

import json
from os import environ
from pathlib import Path

import torch


def selected_conditioning_request(request_id: str) -> bool:
    return environ.get("OMNI_NATIVE_CONDITIONING_PROVENANCE") == "1" and request_id in {
        f"feedback-trace-native-{sample_id}"
        for sample_id in environ["OMNI_NATIVE_TRAJECTORY_SAMPLES"].split(",")
    }


def record_conditioning_boundary(
    request_id: str,
    boundary: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str | int | bool | list[int] | list[str]],
) -> None:
    assert selected_conditioning_request(request_id)
    directory = (
        Path(environ["OMNI_NATIVE_TRAJECTORY_DIRECTORY"])
        / "provenance"
        / request_id.removeprefix("feedback-trace-native-")
    )
    directory.mkdir(parents=True, exist_ok=True)
    snapshots = {
        name: tensor.detach().cpu().clone() for name, tensor in tensors.items()
    }
    with (directory / f"{boundary}.pt").open("xb") as output:
        torch.save(snapshots, output)
    with (directory / f"{boundary}.json").open("x") as output:
        json.dump({"request_id": request_id, **metadata}, output, indent=2)
        output.write("\n")
