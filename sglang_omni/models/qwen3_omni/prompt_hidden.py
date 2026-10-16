# SPDX-License-Identifier: Apache-2.0
"""Request-owned selected Thinker states for multimodal prompt positions."""

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, kw_only=True)
class PromptHiddenStates:
    positions: torch.Tensor
    hidden_states: torch.Tensor


@dataclass(kw_only=True)
class PromptHiddenCapture:
    positions: torch.Tensor
    hidden_states: torch.Tensor | None = field(default=None, init=False)
    received: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.positions = self.positions.detach().to(device="cpu", copy=True)
        self.received = torch.zeros(self.positions.shape, dtype=torch.bool)

    @property
    def needs_capture(self) -> bool:
        return not bool(self.received.all())

    def add_extend(self, *, prefix_length: int, hidden_states: torch.Tensor) -> None:
        if prefix_length < 0 or hidden_states.ndim != 2:
            raise ValueError(
                "Prompt capture requires a nonnegative prefix and 2D states"
            )
        else:
            pass
        missing = (
            ~self.received
            & (self.positions >= prefix_length)
            & (self.positions < prefix_length + hidden_states.shape[0])
        )
        if not bool(missing.any()):
            return
        else:
            pass
        if self.hidden_states is None:
            self.hidden_states = torch.empty(
                (self.positions.numel(), hidden_states.shape[1]),
                dtype=hidden_states.dtype,
                device="cpu",
            )
        elif (
            self.hidden_states.shape[1] != hidden_states.shape[1]
            or self.hidden_states.dtype != hidden_states.dtype
        ):
            raise ValueError("Prompt capture width and dtype must match across chunks")
        else:
            pass
        source_positions = (self.positions[missing] - prefix_length).to(
            device=hidden_states.device
        )
        self.hidden_states[missing] = hidden_states.detach()[source_positions].to(
            device="cpu"
        )
        self.received[missing] = True

    def take_complete(self) -> PromptHiddenStates | None:
        if self.needs_capture:
            raise RuntimeError(
                "Missing multimodal prompt hidden states before Talker start"
            )
        elif self.hidden_states is None:
            return None
        else:
            hidden_states = self.hidden_states
            self.hidden_states = None
            return PromptHiddenStates(
                positions=self.positions, hidden_states=hidden_states
            )
