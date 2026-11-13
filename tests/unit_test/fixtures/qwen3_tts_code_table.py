# SPDX-License-Identifier: Apache-2.0
"""Code table state for Qwen3-TTS runners built without their constructor."""

from __future__ import annotations

import torch

from sglang_omni.models.qwen3_tts.code_table import Qwen3TTSCodeTable
from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.scheduling.types import SchedulerRequest


def attach_code_table(
    runner: Qwen3TTSModelRunner,
    requests: list[SchedulerRequest],
    *,
    row_frames: int,
    codebook_count: int,
    device: torch.device,
) -> Qwen3TTSCodeTable:
    """Give the runner a table, open a row for each request and stage their frame bases."""
    runner.device = device
    runner.code_table = Qwen3TTSCodeTable(
        row_count=2 * len(requests),
        row_frames=row_frames,
        codebook_count=codebook_count,
        device=device,
    )
    runner.code_step = 0
    runner.frames_written_events = []
    runner.step_frames_written = None
    runner.stream_step_target = None
    runner.step_request_ids = []
    runner.step_rows = []
    runner.step_frame_counts = []
    runner.step_opened_streams = {}
    for request in requests:
        runner.open_code_row(request.data)
    runner.stage_frame_index_bases(requests)
    return runner.code_table
