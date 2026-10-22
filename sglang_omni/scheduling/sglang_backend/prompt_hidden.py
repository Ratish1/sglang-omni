# SPDX-License-Identifier: Apache-2.0
"""A request whose next stage reads its hidden states at chosen prompt positions: prefill copies
those rows to pinned host memory as it computes them, and the request's prefix match stops below
the first row it still owes."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req


@dataclass(kw_only=True)
class PromptHiddenCapture:
    """The hidden states a request owes at its ascending prompt positions, as pinned host rows."""

    positions: torch.Tensor
    hidden_states: torch.Tensor | None = field(default=None, init=False)
    is_received: torch.Tensor = field(init=False)
    is_handed_off: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.is_received = torch.zeros(self.positions.shape, dtype=torch.bool)

    def chunk_rows(self, prefix_length: int, extend_length: int) -> slice:
        """The rows at the positions an extend of extend_length tokens after prefix_length
        computes."""
        bounds = torch.tensor([prefix_length, prefix_length + extend_length])
        start, end = torch.searchsorted(self.positions, bounds).tolist()
        return slice(start, end)

    def is_owed_in(self, prefix_length: int, extend_length: int) -> bool:
        rows = self.chunk_rows(prefix_length, extend_length)
        return not self.is_handed_off and not bool(self.is_received[rows].all())

    def first_owed_position(self) -> int | None:
        """The first position whose row is owed, None once all are received or handed off."""
        if self.is_handed_off or bool(self.is_received.all()):
            return None
        else:
            return int(self.positions[~self.is_received][0])

    def add_chunk(self, prefix_length: int, chunk_hidden_states: torch.Tensor) -> None:
        """Copy the chunk's rows at the captured positions into the pinned host rows without
        blocking; the stream orders the copy before the host reads the chunk's sampled token.
        """
        rows = self.chunk_rows(prefix_length, chunk_hidden_states.shape[0])
        if rows.start == rows.stop:
            return
        else:
            pass
        if self.hidden_states is None:
            self.hidden_states = torch.empty(
                (self.positions.numel(), chunk_hidden_states.shape[1]),
                dtype=chunk_hidden_states.dtype,
                pin_memory=True,
            )
        else:
            pass
        offsets = (self.positions[rows] - prefix_length).to(
            device=chunk_hidden_states.device
        )
        self.hidden_states[rows].copy_(
            chunk_hidden_states.index_select(0, offsets), non_blocking=True
        )
        self.is_received[rows] = True

    def take(self) -> torch.Tensor:
        """The rows once every position is received; the request owes none afterwards."""
        if not bool(self.is_received.all()):
            raise RuntimeError(
                "a prompt hidden row is missing when the next stage starts"
            )
        else:
            pass
        self.is_handed_off = True
        return self.hidden_states


class PromptHiddenRequest(Req):
    """A request that owes the next stage prompt hidden states: its prefix match stops below the
    first owed row, as a logprob start keeps the positions it reports out of the match.
    """

    prompt_hidden: PromptHiddenCapture | None = None

    def _compute_max_prefix_len(  # noqa: leading-underscore  # the Req method it overrides
        self, input_len: int
    ) -> int:
        max_prefix_len = super()._compute_max_prefix_len(input_len)
        first_owed_position = (
            None
            if self.prompt_hidden is None
            else self.prompt_hidden.first_owed_position()
        )
        if first_owed_position is None:
            return max_prefix_len
        else:
            return min(max_prefix_len, first_owed_position)
