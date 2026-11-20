# SPDX-License-Identifier: Apache-2.0
"""Asynchronous CUDA event measurements for native Talker prefill."""

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(kw_only=True)
class PrefillTiming:
    start: torch.cuda.Event
    end: torch.cuda.Event
    token_count: int
    bucket_tokens: int | None
    request_ids: list[str]
    host_duration_us: float


class PrefillTimingRecorder:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.pending: deque[PrefillTiming] = deque()

    def flush_ready(self) -> None:
        rows = []
        while self.pending and self.pending[0].end.query():
            timing = self.pending.popleft()
            rows.append(
                {
                    "token_count": timing.token_count,
                    "bucket_tokens": timing.bucket_tokens,
                    "request_ids": timing.request_ids,
                    "device_duration_us": timing.start.elapsed_time(timing.end) * 1000,
                    "host_duration_us": timing.host_duration_us,
                }
            )
        if rows:
            with self.output_path.open("a") as report:
                report.writelines(f"{json.dumps(row)}\n" for row in rows)
        else:
            pass
