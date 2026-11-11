"""Retain fresh decode operands across eager calls and captured graph replays."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig

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
class MoeTacticChoice:
    operation: str
    tensor_shapes: list[list[int]]
    tactic: int
    capturing: bool
    tuning: bool


def observe_moe_tactics(choices: list[MoeTacticChoice]) -> None:
    tuner = AutoTuner.get()
    original_choose_one = tuner.choose_one

    def choose_one(
        custom_op: str,
        runners: list[TunableRunner],
        tuning_config: TuningConfig,
        inputs: list[torch.Tensor | None],
        **kernel_arguments: int,
    ) -> tuple[TunableRunner, int | tuple[int, ...]]:
        runner, tactic = original_choose_one(
            custom_op, runners, tuning_config, inputs, **kernel_arguments
        )
        if custom_op in ("trtllm::fused_moe::gemm1", "trtllm::fused_moe::gemm2"):
            assert isinstance(tactic, int)
            choices.append(
                MoeTacticChoice(
                    operation=custom_op,
                    tensor_shapes=[
                        list(tensor.shape) if tensor is not None else []
                        for tensor in inputs
                    ],
                    tactic=tactic,
                    capturing=torch.cuda.is_current_stream_capturing(),
                    tuning=tuner.is_tuning_mode,
                )
            )
        else:
            pass
        return runner, tactic

    tuner.choose_one = choose_one


@dataclass(kw_only=True)
class LayerOperandRecorder:
    step_counter: torch.Tensor
    retain_moe_batches: bool = False
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
        if self.retain_moe_batches and name in MOE_BATCH_FIELDS:
            self.current_snapshot[f"layout_{name}"] = torch.tensor(
                (*tensor.shape, *tensor.stride(), tensor.data_ptr() % 16), device="cpu"
            )
        else:
            pass
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
            name: (
                tensor.detach().cpu().clone()
                if name.startswith("layout_")
                else tensor[row].detach().cpu().clone()
            )
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
            tensors["batch_is_graph"] = torch.tensor(
                snapshot is not self.eager_snapshot
            )
        else:
            pass
        return tensors
