"""Retain fresh decode operands across eager calls and captured graph replays."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode

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
    maximum_prefill_token_rows: int
    retain_moe_batches: bool = False
    recording_prefill: bool = False
    prefill_buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    graph_snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)
    eager_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)
    current_snapshot: dict[str, torch.Tensor] = field(default_factory=dict)

    def captures_mode(self, forward_mode: ForwardMode) -> bool:
        return forward_mode.is_decode() or (
            self.retain_moe_batches and forward_mode.is_extend()
        )

    def begin(self, positions: torch.Tensor, *, is_prefill: bool = False) -> None:
        self.recording_prefill = is_prefill
        self.current_snapshot = {
            "step": self.step_counter.clone(),
            "positions": positions.movedim(-1, 0).clone(),
        }
        if self.retain_moe_batches:
            self.current_snapshot["layout_is_prefill"] = torch.tensor(
                is_prefill, device="cpu"
            )
        else:
            pass
        if torch.cuda.is_current_stream_capturing():
            self.graph_snapshots.append(self.current_snapshot)
        else:
            self.eager_snapshot = self.current_snapshot

    def record(self, name: str, tensor: torch.Tensor) -> None:
        if self.retain_moe_batches and (
            name in MOE_BATCH_FIELDS or name == "qkv" or name.startswith("operator_")
        ):
            self.current_snapshot[f"layout_{name}"] = torch.tensor(
                (*tensor.shape, *tensor.stride(), tensor.data_ptr() % 16), device="cpu"
            )
        else:
            pass
        if self.recording_prefill:
            assert tensor.shape[0] <= self.maximum_prefill_token_rows
            if name not in self.prefill_buffers:
                assert not torch.cuda.is_current_stream_capturing()
                self.prefill_buffers[name] = tensor.new_empty(
                    (self.maximum_prefill_token_rows, *tensor.shape[1:])
                )
            else:
                pass
            captured = self.prefill_buffers[name][: len(tensor)]
            # note (ratish): non-overlapping prefills let every graph bucket share stable storage.
            assert captured.dtype == tensor.dtype and captured.shape == tensor.shape
            captured.copy_(tensor)
            self.current_snapshot[name] = captured
            self.current_snapshot[f"fresh_{name}"] = self.step_counter.clone()
        else:
            self.current_snapshot[name] = tensor.clone()

    def read(
        self,
        expected_step: int,
        row: int,
        *,
        retain_moe_batches: bool = False,
        retain_operator_batches: bool = False,
        prefill_rows: slice | None = None,
    ) -> dict[str, torch.Tensor]:
        snapshots = [
            snapshot
            for snapshot in (self.eager_snapshot, *self.graph_snapshots)
            if snapshot and int(snapshot["step"]) == expected_step
        ]
        assert len(snapshots) == 1, (expected_step, len(snapshots))
        snapshot = snapshots[0]
        for name, tensor in snapshot.items():
            if name.startswith("fresh_"):
                assert int(tensor) == expected_step, name
            else:
                pass
        tensors = {
            name: (
                tensor.detach().cpu().clone()
                if name.startswith("layout_")
                else tensor[row].detach().cpu().clone()
            )
            for name, tensor in snapshot.items()
            if name != "step" and not name.startswith("fresh_")
        }
        if prefill_rows is not None:
            assert bool(snapshot["layout_is_prefill"])
            assert prefill_rows.start is not None and prefill_rows.stop is not None
            assert (
                0
                <= prefill_rows.start
                <= row
                < prefill_rows.stop
                <= len(snapshot["positions"])
            )
            assert all(
                f"fresh_{name}" in snapshot
                for name in snapshot
                if name not in ("positions", "step")
                and not name.startswith(("fresh_", "layout_"))
            )
            tensors.update(
                {
                    f"prefix_{name}": tensor[prefill_rows].detach().cpu().clone()
                    for name, tensor in snapshot.items()
                    if name != "step" and not name.startswith(("fresh_", "layout_"))
                }
            )
            tensors["prefix_token_rows"] = torch.tensor(
                (prefill_rows.start, prefill_rows.stop)
            )
        else:
            pass
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
        if retain_operator_batches:
            tensors.update(
                {
                    f"batch_{name}": tensor.detach().cpu().clone()
                    for name, tensor in snapshot.items()
                    if name.startswith(("operator_", "layer_")) or name == "positions"
                }
            )
        else:
            pass
        return tensors
