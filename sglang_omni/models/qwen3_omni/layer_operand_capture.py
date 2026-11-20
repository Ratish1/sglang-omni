# SPDX-License-Identifier: Apache-2.0
"""Keep request-scoped layer operands live through a decode graph replay."""

from dataclasses import dataclass, field

import torch


@dataclass(kw_only=True)
class LayerOperandRecorder:
    step_counter: torch.Tensor
    capture_attention_geometry: bool = False
    input_norm_override: torch.Tensor | None = None
    use_input_norm_override: torch.Tensor | None = None
    graph_snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)
    eager_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)
    current_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)

    def begin(self, positions: torch.Tensor) -> None:
        self.current_snapshot = {
            "step": self.step_counter.clone(),
            "positions": positions.movedim(-1, 0).clone(),
        }
        if torch.cuda.is_current_stream_capturing():
            self.graph_snapshots.append(self.current_snapshot)
        else:
            self.eager_snapshot = self.current_snapshot

    def record(self, name: str, tensor: torch.Tensor) -> None:
        self.current_snapshot[name] = tensor.clone()

    def read(self, expected_step: int, row_index: int) -> dict[str, torch.Tensor]:
        snapshots = [
            snapshot
            for snapshot in (self.eager_snapshot, *self.graph_snapshots)
            if snapshot and int(snapshot["step"]) == expected_step
        ]
        assert len(snapshots) == 1, (expected_step, len(snapshots))
        return {
            name: tensor[row_index].detach().cpu().clone()
            for name, tensor in snapshots[0].items()
            if name != "step"
        }
