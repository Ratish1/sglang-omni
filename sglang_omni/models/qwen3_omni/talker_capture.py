# SPDX-License-Identifier: Apache-2.0
"""Opt-in record of what the Qwen3-Omni talker computed for each request.

With SGLANG_OMNI_TALKER_CAPTURE_DIR naming a directory, every finished talker request leaves one torch file there:
the prefill rows and ids the talker started from, the input row of every decode step, and for every step the
talker's final hidden state (codec_head's input, whose logits the first code is sampled from) and the frame's 16
codes. Those are the inputs and outputs a reference talker and code predictor need to be teacher-forced on the same
steps; the request's client metadata is kept so a caller can match a file to its request.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
else:
    pass

CAPTURE_DIR_ENV = "SGLANG_OMNI_TALKER_CAPTURE_DIR"


@dataclass
class TalkerSteps:
    hidden: list[torch.Tensor] = field(default_factory=list)
    codes: list[torch.Tensor] = field(default_factory=list)


class TalkerCapture:
    """Collects each request's steps on the device and writes them once the request finishes."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.steps: dict[str, TalkerSteps] = {}

    @classmethod
    def from_environment(cls) -> TalkerCapture | None:
        directory = os.environ.get(CAPTURE_DIR_ENV)
        if directory:
            return cls(Path(directory))
        else:
            return None

    def record_step(
        self, request_ids: list[str], hidden: torch.Tensor, codes: torch.Tensor
    ) -> None:
        """One talker step of a batch: hidden is [batch, hidden_size], codes is [batch, code_groups]."""
        # note (ratish): both are views of buffers the next step overwrites in place.
        hidden = hidden.detach().clone()
        codes = codes.detach().clone()
        for row, request_id in enumerate(request_ids):
            steps = self.steps.setdefault(request_id, TalkerSteps())
            steps.hidden.append(hidden[row])
            steps.codes.append(codes[row])

    def finish(self, request_id: str, data: SGLangARRequestData) -> None:
        steps = self.steps.pop(request_id, None)
        if steps is None:
            return
        else:
            pass
        sampling = data.req.sampling_params
        history = data.decode_input_embeds or []
        payload = data.stage_payload
        record = {
            "request_id": request_id,
            "metadata": dict(payload.request.metadata) if payload is not None else {},
            "prefill_input_ids": torch.tensor(
                data.req.origin_input_ids, dtype=torch.long
            ),
            "prefill_input_embeds": data.prefill_input_embeds.detach().cpu(),
            "decode_input_embeds": (
                torch.stack(history).cpu()
                if history
                else data.prefill_input_embeds.new_empty(
                    0, data.prefill_input_embeds.shape[-1]
                ).cpu()
            ),
            "hidden": torch.stack(steps.hidden).cpu(),
            "codes": torch.stack(steps.codes).cpu(),
            "sampling": {
                "temperature": sampling.temperature,
                "top_k": sampling.top_k,
                "top_p": sampling.top_p,
                "repetition_penalty": sampling.repetition_penalty,
                "seed": sampling.sampling_seed,
            },
        }
        torch.save(record, self.directory / f"{request_id.replace('/', '_')}.pt")
