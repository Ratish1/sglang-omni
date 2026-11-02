# SPDX-License-Identifier: Apache-2.0
"""Every live Qwen3-TTS request's codec frames on the device, one table for the engine and vocoder."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch


class Qwen3TTSCodeTable:
    """codes[row, frame, codebook] holds a request's reference frames, then its generated frames.

    The engine writes each step's frames in one launch and the vocoder reads a cohort's in one
    gather. A row goes back to the free list once every holder has dropped it.
    """

    def __init__(
        self,
        *,
        num_rows: int,
        max_frames: int,
        num_codebooks: int,
        device: torch.device | str,
    ) -> None:
        self.max_frames = max_frames
        self.codes = torch.zeros(
            (num_rows, max_frames, num_codebooks), dtype=torch.long, device=device
        )
        self.frames = self.codes.view(-1, num_codebooks)
        self.holders = [0] * num_rows
        self.free = list(range(num_rows - 1, -1, -1))
        self.lock = threading.Lock()

    def free_rows(self) -> int:
        return len(self.free)

    def acquire(self) -> int:
        """Take a free row for one holder; admission keeps one free for each request it lets in."""
        with self.lock:
            row = self.free.pop()
            self.holders[row] = 1
        return row

    def hold(self, row: int) -> None:
        with self.lock:
            self.holders[row] += 1

    def drop(self, row: int) -> None:
        with self.lock:
            self.holders[row] -= 1
            if self.holders[row] == 0:
                self.free.append(row)
            else:
                pass


@dataclass(eq=False, frozen=True, kw_only=True)
class CodecFrameStep:
    """A step's frames for the vocoder: where they are, and the event that marks them written."""

    table: Qwen3TTSCodeTable
    ready: torch.Event | None
