"""Retain complete MoE batches, including actual prefill and padding operands."""

from __future__ import annotations

from pathlib import Path

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch import nn

from sglang_omni.models.qwen3_omni.corpus_hooks import (
    AttentionCapture,
    OutputCapture,
    ProjectionCapture,
    TopKCapture,
)


class FullExpertCapture:
    def __init__(
        self,
        layers: nn.ModuleList,
        maximum_rows: int,
        *,
        include_attention: bool = False,
    ) -> None:
        self.maximum_rows = maximum_rows
        self.snapshots: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.finite: dict[str, torch.Tensor] = {}
        self.cache_write_checks: dict[str, torch.Tensor] = {}
        for layer_index, layer in enumerate(layers):
            prefix = f"backbone.{layer_index}."
            projections = [
                ("router", layer.mlp.gate),
                ("shared_gate", layer.mlp.shared_expert_gate),
            ]
            if include_attention:
                projections.extend(
                    [
                        ("qkv", layer.self_attn.qkv_proj),
                        ("attention_projection", layer.self_attn.o_proj),
                        ("shared_up", layer.mlp.shared_expert.gate_up_proj),
                        ("shared_down", layer.mlp.shared_expert.down_proj),
                    ]
                )
            else:
                pass
            for name, projection in projections:
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
            if include_attention:
                attention = layer.self_attn
                for name, width in (
                    ("query", attention.q_size),
                    ("key", attention.kv_size),
                    ("value", attention.kv_size),
                ):
                    self.snapshots[prefix + name] = torch.empty(
                        maximum_rows, width, device=weight.device, dtype=weight.dtype
                    )
                count = torch.zeros((), device=weight.device, dtype=torch.int64)
                self.counts[prefix + "attention"] = count
                write_check = torch.zeros(3, device=weight.device, dtype=torch.int64)
                self.cache_write_checks[prefix] = write_check
                attention.attn.register_forward_hook(
                    AttentionCapture(
                        None,
                        self.snapshots[prefix + "query"],
                        self.snapshots[prefix + "key"],
                        self.snapshots[prefix + "value"],
                        count,
                        write_check,
                    )
                )
            else:
                pass

    def prepare(self) -> None:
        for count in self.counts.values():
            count.zero_()

    def save(
        self,
        path: Path,
        real_rows: int,
        query_indices: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        counts = {name: int(count.cpu()) for name, count in self.counts.items()}
        row_count = counts["backbone.0.router"]
        assert 0 < real_rows <= row_count and set(counts.values()) == {row_count}
        assert row_count <= self.maximum_rows, (row_count, self.maximum_rows)
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
                "cache_write_checks": {
                    name: checks.cpu()
                    for name, checks in self.cache_write_checks.items()
                },
                "query_indices": query_indices.cpu(),
                "positions": forward_batch.positions[:real_rows].cpu(),
                "mrope_positions": (
                    forward_batch.mrope_positions[:, :real_rows].cpu()
                    if forward_batch.mrope_positions is not None
                    else torch.empty(0, device="cpu")
                ),
                "extend_lengths": (
                    list(forward_batch.extend_seq_lens_cpu)
                    if forward_batch.forward_mode.is_extend()
                    else [1] * len(query_indices)
                ),
                "prefix_lengths": (
                    list(forward_batch.extend_prefix_lens_cpu)
                    if forward_batch.forward_mode.is_extend()
                    else []
                ),
            },
            path,
        )
