"""Retain fresh decode operands across eager calls and captured graph replays."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

MOE_BATCH_FIELDS = frozenset(
    (
        "mlp_normalized",
        "router_logits",
        "expert_indices",
        "expert_weights",
        "routed_output",
    )
)


@dataclass(kw_only=True)
class LayerOperandRecorder:
    step_counter: torch.Tensor
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

    def read(
        self, expected_step: int, row: int, *, retain_moe_batches: bool = False
    ) -> dict[str, torch.Tensor]:
        snapshots = [
            snapshot
            for snapshot in (self.eager_snapshot, *self.graph_snapshots)
            if snapshot and int(snapshot["step"]) == expected_step
        ]
        assert len(snapshots) == 1, (expected_step, len(snapshots))
        snapshot = snapshots[0]
        tensors = {
            name: tensor[row].detach().cpu().clone()
            for name, tensor in snapshot.items()
            if name != "step"
        }
        if retain_moe_batches:
            tensors.update(
                {
                    f"batch_{name}": snapshot[name].detach().cpu().clone()
                    for name in MOE_BATCH_FIELDS
                }
            )
            tensors["batch_selected_row"] = torch.tensor(row)
        else:
            pass
        return tensors
