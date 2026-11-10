"""Copy actual model operation boundaries into fixed-address observation tensors."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from torch import nn


class ProjectionCapture:
    def __init__(
        self,
        input_snapshot: torch.Tensor,
        output_snapshot: torch.Tensor,
        row_count: torch.Tensor,
        finite: torch.Tensor,
        query_indices: torch.Tensor | None,
    ) -> None:
        self.input_snapshot = input_snapshot
        self.output_snapshot = output_snapshot
        self.row_count = row_count
        self.finite = finite
        self.query_indices = query_indices

    def __call__(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor],
        output: tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        input_rows = arguments[0].reshape(-1, self.input_snapshot.shape[-1])
        output_rows = output[0].reshape(-1, self.output_snapshot.shape[-1])
        self.row_count.fill_(input_rows.shape[0])
        self.finite.copy_(output_rows.isfinite().all())
        if self.query_indices is None:
            captured_rows = min(input_rows.shape[0], self.input_snapshot.shape[0])
            self.input_snapshot[:captured_rows].copy_(input_rows[:captured_rows])
            self.output_snapshot[:captured_rows].copy_(output_rows[:captured_rows])
        else:
            self.input_snapshot.copy_(input_rows.index_select(0, self.query_indices))
            self.output_snapshot.copy_(output_rows.index_select(0, self.query_indices))


class OutputCapture:
    def __init__(
        self,
        snapshot: torch.Tensor,
        query_indices: torch.Tensor | None,
        finite: torch.Tensor,
        row_count: torch.Tensor,
    ) -> None:
        self.snapshot = snapshot
        self.query_indices = query_indices
        self.finite = finite
        self.row_count = row_count

    def __call__(
        self,
        module: nn.Module,
        arguments: (
            tuple[torch.Tensor]
            | tuple[torch.Tensor, ForwardBatch | None]
            | tuple[torch.Tensor, StandardTopKOutput]
            | tuple[torch.Tensor, torch.Tensor, torch.Tensor, ForwardBatch]
        ),
        output: torch.Tensor,
    ) -> None:
        if self.query_indices is None:
            captured_rows = min(output.shape[0], self.snapshot.shape[0])
            self.snapshot[:captured_rows].copy_(output[:captured_rows])
        else:
            self.snapshot.copy_(output.index_select(0, self.query_indices))
        self.finite.copy_(output.isfinite().all())
        self.row_count.fill_(output.shape[0])


class TopKCapture:
    def __init__(
        self,
        expert_ids: torch.Tensor,
        weights: torch.Tensor,
        query_indices: torch.Tensor | None,
        row_count: torch.Tensor,
    ) -> None:
        self.expert_ids = expert_ids
        self.weights = weights
        self.query_indices = query_indices
        self.row_count = row_count

    def __call__(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor, torch.Tensor],
        output: StandardTopKOutput,
    ) -> None:
        assert isinstance(output, StandardTopKOutput)
        if self.query_indices is None:
            captured_rows = min(output.topk_ids.shape[0], self.expert_ids.shape[0])
            self.expert_ids[:captured_rows].copy_(output.topk_ids[:captured_rows])
            self.weights[:captured_rows].copy_(output.topk_weights[:captured_rows])
        else:
            self.expert_ids.copy_(output.topk_ids.index_select(0, self.query_indices))
            self.weights.copy_(output.topk_weights.index_select(0, self.query_indices))
        self.row_count.fill_(output.topk_ids.shape[0])


def record_cache_write_checks(
    directory: Path,
    checks: list[torch.Tensor],
    forward_batch: ForwardBatch,
    request_ids: list[str],
) -> None:
    write_checks = torch.stack(checks).cpu().tolist()
    actual_tokens = (
        sum(forward_batch.extend_seq_lens_cpu)
        if forward_batch.forward_mode.is_extend()
        else len(request_ids)
    )
    with (directory / "cache-writes.jsonl").open("a") as output:
        output.write(
            json.dumps(
                {
                    "request_ids": request_ids,
                    "phase": forward_batch.forward_mode.name,
                    "actual_tokens": actual_tokens,
                    "layer_key_exact_value_exact_live_tokens": write_checks,
                }
            )
            + "\n"
        )
    assert all(
        keys == values == 1 and rows == actual_tokens
        for keys, values, rows in write_checks
    ), write_checks


class CacheWriteCapture:
    def __init__(self, checks: torch.Tensor) -> None:
        self.checks = checks

    def __call__(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor, torch.Tensor, torch.Tensor, ForwardBatch],
        output: torch.Tensor,
    ) -> None:
        query, keys, values, forward_batch = arguments
        key_buffer, value_buffer = get_token_to_kv_pool().get_kv_buffer(module.layer_id)
        assert (
            keys.untyped_storage().data_ptr() != key_buffer.untyped_storage().data_ptr()
        )
        assert (
            values.untyped_storage().data_ptr()
            != value_buffer.untyped_storage().data_ptr()
        )
        projected_keys = keys.reshape(query.shape[0], -1)
        projected_values = values.reshape(query.shape[0], -1)
        locations = forward_batch.out_cache_loc[: query.shape[0]].long()
        live_rows = locations != 0
        stored_keys = key_buffer.index_select(0, locations).reshape_as(projected_keys)
        stored_values = value_buffer.index_select(0, locations).reshape_as(
            projected_values
        )
        self.checks[0].copy_(
            ((stored_keys == projected_keys) | ~live_rows[:, None]).all()
        )
        self.checks[1].copy_(
            ((stored_values == projected_values) | ~live_rows[:, None]).all()
        )
        self.checks[2].copy_(live_rows.sum())


class AttentionCapture:
    def __init__(
        self,
        query_indices: torch.Tensor | None,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        row_count: torch.Tensor,
        write_check: torch.Tensor,
    ) -> None:
        self.query_indices = query_indices
        self.queries = queries
        self.keys = keys
        self.values = values
        self.row_count = row_count
        self.write_capture = CacheWriteCapture(write_check)

    def __call__(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor, torch.Tensor, torch.Tensor, ForwardBatch],
        output: torch.Tensor,
    ) -> None:
        query, keys, values, forward_batch = arguments
        self.write_capture(module, arguments, output)
        projected_keys = keys.reshape(query.shape[0], self.keys.shape[-1])
        projected_values = values.reshape(query.shape[0], self.values.shape[-1])
        if self.query_indices is None:
            captured_rows = min(query.shape[0], self.queries.shape[0])
            self.queries[:captured_rows].copy_(
                query.reshape(-1, self.queries.shape[-1])[:captured_rows]
            )
            self.keys[:captured_rows].copy_(projected_keys[:captured_rows])
            self.values[:captured_rows].copy_(projected_values[:captured_rows])
        else:
            self.queries.copy_(
                query.reshape(-1, self.queries.shape[-1]).index_select(
                    0, self.query_indices
                )
            )
            self.keys.copy_(projected_keys.index_select(0, self.query_indices))
            self.values.copy_(projected_values.index_select(0, self.query_indices))
        self.row_count.fill_(query.shape[0])
