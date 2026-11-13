# SPDX-License-Identifier: Apache-2.0
"""Every live Qwen3-TTS request's codec frames in one device table the engine writes and the vocoder reads."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch


class Qwen3TTSCodeTable:
    """codes[row, frame, codebook] holds a request's reference frames, then its generated frames.

    The engine writes a step's frames in one launch and the vocoder gathers a cohort's in one
    read. Holders and the free rows change under lock; only the engine thread writes
    settled_step, and a reader that sees an older value only waits when it need not.
    """

    def __init__(
        self,
        *,
        row_count: int,
        row_frames: int,
        codebook_count: int,
        device: torch.device,
    ) -> None:
        self.row_frames = row_frames
        self.codes = torch.zeros(
            (row_count, row_frames, codebook_count), dtype=torch.long, device=device
        )
        self.frames = self.codes.view(-1, codebook_count)
        self.holders = [0] * row_count
        self.free_rows = list(range(row_count - 1, -1, -1))
        self.lock = threading.Lock()
        # note (ratish): every engine step up to this one has its frames written.
        self.settled_step = 0

    def free_row_count(self) -> int:
        return len(self.free_rows)

    def acquire(self) -> int:
        """A free row with one holder; admission keeps one free for each request it lets in."""
        with self.lock:
            row = self.free_rows.pop()
            self.holders[row] = 1
        return row

    def hold(self, row: int) -> None:
        with self.lock:
            self.holders[row] += 1

    def drop(self, row: int) -> None:
        with self.lock:
            self.holders[row] -= 1
            if self.holders[row] == 0:
                self.free_rows.append(row)
            else:
                pass


@dataclass(eq=False, frozen=True, kw_only=True)
class CodecFrameStep:
    """One engine step's frames for the vocoder, parallel to the step's request ids.

    rows and frame_counts give each request's table row and its generated frames after the
    step; written is recorded once the step's frames are in the table, after step's earlier
    steps on the engine's one stream.
    """

    table: Qwen3TTSCodeTable
    rows: list[int]
    frame_counts: list[int]
    written: torch.Event | None
    step: int

    def wait_written(self, stream: torch.Stream) -> None:
        """Order stream after this step's frame writes, unless the engine has settled them."""
        if self.written is None or self.step <= self.table.settled_step:
            return
        else:
            stream.wait_event(self.written)
