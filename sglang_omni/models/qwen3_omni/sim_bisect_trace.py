# SPDX-License-Identifier: Apache-2.0
"""Record or force complete Talker frame histories for a SIM diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal, TypedDict

import torch


class SimTraceEvent(TypedDict, total=False):
    kind: Literal["input", "frame"]
    phase: Literal["prefill", "decode"]
    projected_input_sha256: str
    positions_sha256: str
    input_ids_sha256: str
    frame_index: int
    codes: list[int]
    end_token_id: int
    forced: bool
    chosen_code_logit: float
    best_other_code_logit: float
    residual_code_margins: list[float]


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


class SimBisectTrace:
    def __init__(self, trace_directory: Path, donor_directory: Path | None) -> None:
        self.trace_directory = trace_directory
        manifest = json.loads(Path(os.environ["SGLANG_OMNI_SIM_MANIFEST"]).read_text())
        self.request_ids: frozenset[str] = frozenset(
            case["request_id"] for case in manifest["cases"]
        )
        self.donor_frames: dict[str, list[list[int]]] = {}
        if donor_directory is not None:
            for request_id in self.request_ids:
                events = [
                    json.loads(line)
                    for line in (donor_directory / f"{request_id}.jsonl")
                    .read_text()
                    .splitlines()
                ]
                frames = [event for event in events if event["kind"] == "frame"]
                assert frames and all(
                    event["frame_index"] == index and not event["forced"]
                    for index, event in enumerate(frames)
                ), request_id
                assert frames[-1]["codes"][0] == frames[-1]["end_token_id"], request_id
                self.donor_frames[request_id] = [event["codes"] for event in frames]
        else:
            pass

    def append(self, request_id: str, event: SimTraceEvent) -> None:
        with (self.trace_directory / f"{request_id}.jsonl").open("a") as trace_file:
            trace_file.write(json.dumps(event, allow_nan=False) + "\n")

    def record_input(
        self,
        request_id: str,
        phase: Literal["prefill", "decode"],
        projected_input: torch.Tensor,
        positions: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> None:
        self.append(
            request_id,
            {
                "kind": "input",
                "phase": phase,
                "projected_input_sha256": tensor_sha256(projected_input),
                "positions_sha256": tensor_sha256(positions),
                "input_ids_sha256": tensor_sha256(input_ids),
            },
        )

    def forced_codes(self, request_id: str, frame_index: int) -> list[int] | None:
        frames = self.donor_frames.get(request_id)
        return frames[frame_index] if frames is not None else None

    def record_frame(
        self,
        request_id: str,
        frame_index: int,
        codes: torch.Tensor,
        end_token_id: int,
        unprocessed_logits: torch.Tensor,
        *,
        forced: bool,
        residual_code_margins: list[float],
    ) -> None:
        assert codes.ndim == 1 and unprocessed_logits.ndim == 1
        code_ids: list[int] = codes.tolist()
        chosen_code_logit = float(unprocessed_logits[code_ids[0]])
        alternatives = unprocessed_logits.clone()
        alternatives[code_ids[0]] = float("-inf")
        self.append(
            request_id,
            {
                "kind": "frame",
                "frame_index": frame_index,
                "codes": code_ids,
                "end_token_id": end_token_id,
                "forced": forced,
                "chosen_code_logit": chosen_code_logit,
                "best_other_code_logit": float(alternatives.max()),
                "residual_code_margins": residual_code_margins,
            },
        )
