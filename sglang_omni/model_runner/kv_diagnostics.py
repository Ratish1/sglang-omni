"""Observe full-prefix KV preservation without changing model inputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_token_to_kv_pool
from torch import nn

from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.scheduling.types import SchedulerRequest


class KVWriteHook:
    def __call__(
        self,
        attention: RadixAttention,
        arguments: tuple[torch.Tensor, torch.Tensor, torch.Tensor, ForwardBatch],
        output: torch.Tensor,
    ) -> None:
        query, keys, values, forward_batch = arguments
        locations = forward_batch.out_cache_loc.long()
        pool = get_token_to_kv_pool()
        stored_keys, stored_values = pool.get_kv_buffer(attention.layer_id)
        valid = locations > 0
        wrong_keys = (
            stored_keys.index_select(0, locations)
            != keys.reshape(-1, attention.tp_k_head_num, attention.head_dim)
        ).any(dim=(1, 2))
        wrong_values = (
            stored_values.index_select(0, locations)
            != values.reshape(-1, attention.tp_v_head_num, attention.v_head_dim)
        ).any(dim=(1, 2))
        attention.diagnostic_kv_write_counts.add_(
            torch.stack(
                [(wrong_keys & valid).sum(), (wrong_values & valid).sum(), valid.sum()]
            )
        )
        if query.shape[0] == 1:
            attention.diagnostic_query.copy_(
                query.view(attention.tp_q_head_num, attention.head_dim)
            )
            attention.diagnostic_attention_output.copy_(
                output.view(attention.tp_q_head_num, attention.v_head_dim)
            )
            attention.diagnostic_position.copy_(forward_batch.positions.reshape(-1)[0])
        else:
            pass


def install_kv_write_observers(layers: nn.ModuleList) -> None:
    if os.environ.get("SGLANG_OMNI_DIAGNOSTIC_KV_DIRECTORY") is None:
        return
    else:
        pass
    for layer in layers:
        attention = layer.self_attn.attn
        assert isinstance(attention, RadixAttention)
        device = layer.self_attn.q_norm.weight.device
        for name, shape, dtype in (
            ("diagnostic_kv_write_counts", (3,), torch.int64),
            (
                "diagnostic_query",
                (attention.tp_q_head_num, attention.head_dim),
                torch.bfloat16,
            ),
            (
                "diagnostic_attention_output",
                (attention.tp_q_head_num, attention.v_head_dim),
                torch.bfloat16,
            ),
            ("diagnostic_position", (), torch.int64),
        ):
            attention.register_buffer(
                name, torch.zeros(shape, dtype=dtype, device=device), persistent=False
            )
        attention.register_buffer(
            "diagnostic_attention_properties",
            torch.tensor(
                [attention.scaling, attention.logit_cap, attention.sliding_window_size],
                device=device,
                dtype=torch.float64,
            ),
            persistent=False,
        )
        attention.register_forward_hook(KVWriteHook())


@dataclass(kw_only=True)
class PrefixKVObservation:
    prefix_locations: torch.Tensor
    prefix_keys: list[torch.Tensor]
    prefix_values: list[torch.Tensor]
    checks: list[torch.Tensor] = field(default_factory=list)
    sequence_lengths: list[int] = field(default_factory=list)
    snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)


class FullPrefixKVObserver:
    def __init__(self, directory: Path, runner: SGLModelRunner, stage: str) -> None:
        self.directory = directory / stage
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.observations: dict[str, PrefixKVObservation] = {}
        backend = runner.attn_backend
        pool = runner.token_to_kv_pool
        assert isinstance(backend, FlashAttentionBackend)
        assert isinstance(pool, MHATokenToKVPool)
        assert not runner.kv_index_translator.is_translating
        assert not backend.use_mla and not backend.has_swa
        assert backend.fa_impl_ver == 3 and pool.dtype == torch.bfloat16
        assert pool.kv_cache_layout == "nhd"
        self.backend = backend
        self.pool = pool
        self.attention_tensors = {
            name: buffer
            for name, buffer in runner.model.named_buffers()
            if name.endswith(
                (
                    ".diagnostic_kv_write_counts",
                    ".diagnostic_query",
                    ".diagnostic_attention_output",
                    ".diagnostic_position",
                    ".diagnostic_attention_properties",
                )
            )
        }
        assert len(self.attention_tensors) == 5 * pool.layer_num, (
            stage,
            len(self.attention_tensors),
            pool.layer_num,
        )
        metadata = {
            "pool_class": type(pool).__name__,
            "allocator_class": type(runner.token_to_kv_pool_allocator).__name__,
            "backend_class": type(backend).__name__,
            "page_size": runner.page_size,
            "is_translating": runner.kv_index_translator.is_translating,
            "layer_count": pool.layer_num,
            "pool_tokens": pool.size,
            "buffer_shape": list(pool.k_buffer[0].shape),
            "key_stride": list(pool.k_buffer[0].stride()),
            "value_stride": list(pool.v_buffer[0].stride()),
            "same_storage": pool.k_buffer[0].untyped_storage().data_ptr()
            == pool.v_buffer[0].untyped_storage().data_ptr(),
            "key_storage_offset": pool.k_buffer[0].storage_offset(),
            "value_storage_offset": pool.v_buffer[0].storage_offset(),
        }
        (self.directory / "runtime.json").write_text(json.dumps(metadata, indent=2))

    def observe(
        self, forward_batch: ForwardBatch, requests: list[SchedulerRequest]
    ) -> None:
        sequence_lengths = forward_batch.seq_lens[: len(requests)].tolist()
        request_pool_indices = forward_batch.req_pool_indices[: len(requests)]
        request_table = self.runner.req_to_token_pool.req_to_token
        maximum_sequence_length = max(sequence_lengths)
        token_offsets = torch.arange(
            maximum_sequence_length, device=request_pool_indices.device
        )
        request_locations = request_table.index_select(0, request_pool_indices)[
            :, :maximum_sequence_length
        ]
        metadata = self.backend.forward_metadata
        page_locations = metadata.page_table[: len(requests)].index_select(
            1, token_offsets // self.runner.page_size
        )
        read_locations = (
            page_locations * self.runner.page_size
            + token_offsets % self.runner.page_size
        )
        decode = forward_batch.forward_mode.is_decode()
        for row_index, request in enumerate(requests):
            sequence_length = sequence_lengths[row_index]
            locations = request_locations[row_index, :sequence_length].long()
            if decode:
                observation = self.observations[request.request_id]
            else:
                observation = PrefixKVObservation(
                    prefix_locations=locations.clone(),
                    prefix_keys=[
                        buffer.index_select(0, locations)
                        for buffer in self.pool.k_buffer
                    ],
                    prefix_values=[
                        buffer.index_select(0, locations)
                        for buffer in self.pool.v_buffer
                    ],
                )
                assert request.request_id not in self.observations
                self.observations[request.request_id] = observation
            prefix_length = observation.prefix_locations.numel()
            checks = [
                (locations[:prefix_length] == observation.prefix_locations).all(),
                (read_locations[row_index, :sequence_length] == locations).all(),
                metadata.cache_seqlens_int32[row_index] == sequence_length,
                (locations > 0).all(),
                (locations < self.pool.size + self.runner.page_size).all(),
            ]
            if decode:
                checks.extend(
                    [
                        forward_batch.positions[row_index] == sequence_length - 1,
                        forward_batch.out_cache_loc[row_index] == locations[-1],
                    ]
                )
            else:
                checks.extend([torch.ones_like(checks[0])] * 2)
            for key_buffer, value_buffer, expected_keys, expected_values in zip(
                self.pool.k_buffer,
                self.pool.v_buffer,
                observation.prefix_keys,
                observation.prefix_values,
            ):
                checks.extend(
                    [
                        (
                            key_buffer.index_select(0, observation.prefix_locations)
                            == expected_keys
                        ).all(),
                        (
                            value_buffer.index_select(0, observation.prefix_locations)
                            == expected_values
                        ).all(),
                    ]
                )
            observation.checks.append(torch.stack(checks))
            observation.sequence_lengths.append(sequence_length)
            if len(observation.snapshots) < 4:
                snapshot = {
                    "request_locations": locations.clone(),
                    "read_locations": read_locations[
                        row_index, :sequence_length
                    ].clone(),
                    "sequence_length": forward_batch.seq_lens[row_index].clone(),
                    "request_pool_index": request_pool_indices[row_index].clone(),
                }
                if decode and len(requests) == 1:
                    snapshot.update(
                        {
                            name: tensor.clone()
                            for name, tensor in self.attention_tensors.items()
                        }
                    )
                    for layer_index, (key_buffer, value_buffer) in enumerate(
                        zip(self.pool.k_buffer, self.pool.v_buffer)
                    ):
                        snapshot[f"layer_{layer_index}_keys"] = key_buffer.index_select(
                            0, locations
                        )
                        snapshot[f"layer_{layer_index}_values"] = (
                            value_buffer.index_select(0, locations)
                        )
                else:
                    pass
                observation.snapshots.append(snapshot)
            else:
                pass

    def finish(self, request_id: str) -> None:
        observation = self.observations.pop(request_id)
        checks = torch.stack(observation.checks).cpu()
        report = {
            "request_id": request_id,
            "prefix_tokens": observation.prefix_locations.numel(),
            "forward_count": len(observation.checks),
            "sequence_lengths": observation.sequence_lengths,
            "failed_checks": (~checks).nonzero().tolist(),
            "write_counts": {
                name: tensor.cpu().tolist()
                for name, tensor in self.attention_tensors.items()
                if name.endswith(".diagnostic_kv_write_counts")
            },
            "check_columns": [
                "prefix_locations",
                "read_locations",
                "sequence_length",
                "positive_locations",
                "bounded_locations",
                "decode_position",
                "decode_write_location",
                *[
                    f"layer_{layer_index}_{kind}_preserved"
                    for layer_index in range(self.pool.layer_num)
                    for kind in ("key", "value")
                ],
            ],
        }
        ordinal = len(list(self.directory.glob("request-*.pt")))
        torch.save(
            {
                "request_id": request_id,
                "prefix_locations": observation.prefix_locations.cpu(),
                "prefix_keys": [tensor.cpu() for tensor in observation.prefix_keys],
                "prefix_values": [tensor.cpu() for tensor in observation.prefix_values],
                "snapshots": [
                    {name: tensor.cpu() for name, tensor in snapshot.items()}
                    for snapshot in observation.snapshots
                ],
            },
            self.directory / f"request-{ordinal}.pt",
        )
        with (self.directory / f"checks-{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
