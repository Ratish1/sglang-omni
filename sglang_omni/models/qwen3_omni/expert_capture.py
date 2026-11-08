"""Retain complete MoE batches, including actual prefill and padding operands."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from sglang_omni.models.qwen3_omni.corpus_hooks import (
    OutputCapture,
    ProjectionCapture,
    TopKCapture,
)


class FullExpertCapture:
    def __init__(self, layers: nn.ModuleList, maximum_rows: int) -> None:
        self.snapshots: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.finite: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(layers):
            prefix = f"backbone.{layer_index}."
            for name, projection in (
                ("router", layer.mlp.gate),
                ("shared_gate", layer.mlp.shared_expert_gate),
            ):
                weight = projection.weight
                inputs = torch.empty(
                    maximum_rows,
                    weight.shape[1],
                    device=weight.device,
                    dtype=weight.dtype,
                )
                outputs = torch.empty(
                    maximum_rows,
                    weight.shape[0],
                    device=weight.device,
                    dtype=weight.dtype,
                )
                count = torch.zeros((), device=weight.device, dtype=torch.int64)
                finite = torch.zeros((), device=weight.device, dtype=torch.bool)
                self.snapshots[prefix + name + ".input"] = inputs
                self.snapshots[prefix + name + ".output"] = outputs
                self.counts[prefix + name] = count
                self.finite[prefix + name] = finite
                projection.register_forward_hook(
                    ProjectionCapture(inputs, outputs, count, finite, None)
                )
            weight = layer.mlp.gate.weight
            for name, module in (
                ("routed", layer.mlp.experts),
                ("shared", layer.mlp.shared_expert),
                ("combined", layer.mlp),
            ):
                outputs = torch.empty(
                    maximum_rows,
                    weight.shape[1],
                    device=weight.device,
                    dtype=weight.dtype,
                )
                count = torch.zeros((), device=weight.device, dtype=torch.int64)
                finite = torch.zeros((), device=weight.device, dtype=torch.bool)
                self.snapshots[prefix + name] = outputs
                self.counts[prefix + name] = count
                self.finite[prefix + name] = finite
                module.register_forward_hook(
                    OutputCapture(outputs, None, finite, count)
                )
            ids = torch.empty(
                maximum_rows,
                layer.config.num_experts_per_tok,
                device=weight.device,
                dtype=torch.int32,
            )
            scales = torch.empty_like(ids, dtype=torch.float32)
            count = torch.zeros((), device=weight.device, dtype=torch.int64)
            self.snapshots[prefix + "expert_ids"] = ids
            self.snapshots[prefix + "expert_weights"] = scales
            self.counts[prefix + "topk"] = count
            layer.mlp.topk.register_forward_hook(TopKCapture(ids, scales, None, count))

    def prepare(self) -> None:
        for count in self.counts.values():
            count.zero_()

    def save(self, path: Path, real_rows: int, query_indices: torch.Tensor) -> None:
        counts = {name: int(count.cpu()) for name, count in self.counts.items()}
        row_count = counts["backbone.0.router"]
        assert 0 < real_rows <= row_count and set(counts.values()) == {row_count}
        torch.save(
            {
                "snapshots": {
                    name: tensor[:row_count].cpu()
                    for name, tensor in self.snapshots.items()
                },
                "finite_including_padding": {
                    name: value.cpu() for name, value in self.finite.items()
                },
                "real_rows": real_rows,
                "query_indices": query_indices.cpu(),
            },
            path,
        )
