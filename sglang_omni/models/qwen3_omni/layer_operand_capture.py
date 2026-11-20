# SPDX-License-Identifier: Apache-2.0
"""Keep request-scoped layer operands live through a decode graph replay."""

from dataclasses import dataclass, field

import torch


@dataclass(kw_only=True)
class LayerOperandRecorder:
    step_counter: torch.Tensor
    capture_attention_geometry: bool = False
    prefill_token_index: torch.Tensor | None = None
    records_packed_prefill_experts: bool = False
    is_recording_prefill: bool = False
    input_norm_override: torch.Tensor | None = None
    use_input_norm_override: torch.Tensor | None = None
    graph_snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)
    prefill_graph_snapshots: dict[int, dict[str, torch.Tensor]] = field(
        default_factory=dict
    )
    eager_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)
    eager_prefill_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)
    current_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)

    def records(self, is_decode: bool, is_extend: bool) -> bool:
        return is_decode or (is_extend and self.prefill_token_index is not None)

    def begin(self, positions: torch.Tensor, is_prefill: bool) -> None:
        self.is_recording_prefill = is_prefill
        self.current_snapshot = {"step": self.step_counter.clone()}
        self.record("positions", positions.movedim(-1, 0))
        is_capturing = torch.cuda.is_current_stream_capturing()
        if is_prefill and is_capturing:
            bucket_tokens = positions.shape[-1]
            self.prefill_graph_snapshots[bucket_tokens] = self.current_snapshot
        elif is_prefill:
            self.eager_prefill_snapshot = self.current_snapshot
        elif is_capturing:
            self.graph_snapshots.append(self.current_snapshot)
        else:
            self.eager_snapshot = self.current_snapshot

    def record(self, name: str, tensor: torch.Tensor) -> None:
        if self.is_recording_prefill:
            assert self.prefill_token_index is not None
            token_index = self.prefill_token_index.clamp(max=tensor.shape[0] - 1)
            self.current_snapshot[name] = tensor.index_select(0, token_index)
        else:
            self.current_snapshot[name] = tensor.clone()

    def record_packed_prefill(self, name: str, tensor: torch.Tensor) -> None:
        assert self.is_recording_prefill and self.records_packed_prefill_experts
        self.current_snapshot[f"packed_{name}"] = tensor.clone()

    def read(
        self,
        expected_step: int,
        row_index: int,
        *,
        captured_rows: int | None = None,
    ) -> dict[str, torch.Tensor]:
        snapshots = [
            snapshot
            for snapshot in (self.eager_snapshot, *self.graph_snapshots)
            if snapshot
            and (
                captured_rows is None or snapshot["positions"].size(0) == captured_rows
            )
            and int(snapshot["step"]) == expected_step
        ]
        assert len(snapshots) == 1, (expected_step, len(snapshots))
        return {
            name: tensor[row_index].detach().cpu().clone()
            for name, tensor in snapshots[0].items()
            if name != "step"
        }

    def read_prefill(
        self, expected_step: int, bucket_tokens: int | None
    ) -> dict[str, torch.Tensor]:
        """Use the executed shape because graphs can share snapshot addresses."""
        snapshot = (
            self.eager_prefill_snapshot
            if bucket_tokens is None
            else self.prefill_graph_snapshots[bucket_tokens]
        )
        assert int(snapshot["step"]) == expected_step
        return {
            name: (tensor if name.startswith("packed_") else tensor[0])
            .detach()
            .cpu()
            .clone()
            for name, tensor in snapshot.items()
            if name != "step"
        }
