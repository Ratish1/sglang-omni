"""Observe prompt and generated KV preservation without changing model inputs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.model_runner.sglang_model_runner import SGLModelRunner
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass(kw_only=True)
class PrefixKVObservation:
    prefix_locations: torch.Tensor
    prefix_keys: list[torch.Tensor]
    prefix_values: list[torch.Tensor]
    checks: list[torch.Tensor] = field(default_factory=list)
    sequence_lengths: list[int] = field(default_factory=list)
    snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)
    write_conflicts: list[tuple[int, int, list[str]]] = field(default_factory=list)


class FullPrefixKVObserver:
    def __init__(self, directory: Path, runner: SGLModelRunner, stage: str) -> None:
        self.directory = directory / stage
        self.directory.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.observations: dict[str, PrefixKVObservation] = {}
        self.prefill_inputs: dict[str, dict[str, torch.Tensor]] = {}
        self.live_rows: dict[str, tuple[int, int]] = {}
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
                    ".diagnostic_query_index",
                    ".diagnostic_attention_properties",
                )
            )
        }
        assert len(self.attention_tensors) == 6 * pool.layer_num, (
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

    def prepare_forward(self, forward_batch: ForwardBatch) -> None:
        query_index = (
            forward_batch.extend_seq_lens_cpu[0] - 1
            if forward_batch.forward_mode.is_extend()
            else 0
        )
        for name, tensor in self.attention_tensors.items():
            if name.endswith(".diagnostic_query_index"):
                tensor.fill_(query_index)
            else:
                pass

    def capture_prefill_inputs(
        self,
        forward_batch: ForwardBatch,
        request_ids: list[str],
        embeddings: torch.Tensor | None = None,
    ) -> None:
        if embeddings is None:
            sidecar = get_omni_prefill_inputs(forward_batch)
            embeddings = (
                sidecar.input_embeds
                if sidecar is not None
                else forward_batch.input_embeds
            )
        else:
            pass
        row_start = 0
        for request_id, extend_length in zip(
            request_ids, forward_batch.extend_seq_lens_cpu
        ):
            row_end = row_start + extend_length
            if (
                request_id in self.observations
                and len(self.observations[request_id].snapshots) >= 4
            ):
                row_start = row_end
                continue
            else:
                pass
            snapshot = {
                "prefill_input_ids": forward_batch.input_ids[row_start:row_end].clone(),
                "prefill_positions": forward_batch.positions[row_start:row_end].clone(),
            }
            if embeddings is not None:
                snapshot["prefill_input_embeddings"] = embeddings[
                    row_start:row_end
                ].clone()
            else:
                pass
            if forward_batch.mrope_positions is not None:
                snapshot["prefill_mrope_positions"] = forward_batch.mrope_positions[
                    :, row_start:row_end
                ].clone()
            else:
                pass
            self.prefill_inputs[request_id] = snapshot
            row_start = row_end

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
        for request, pool_index, sequence_length in zip(
            requests, request_pool_indices.cpu().tolist(), sequence_lengths
        ):
            self.live_rows[request.request_id] = (pool_index, sequence_length)
        live_request_ids = list(self.live_rows)
        live_pool_indices = torch.tensor(
            [self.live_rows[request_id][0] for request_id in live_request_ids],
            device=request_table.device,
        )
        maximum_live_length = max(length for _, length in self.live_rows.values())
        live_locations = (
            request_table.index_select(0, live_pool_indices)[:, :maximum_live_length]
            .cpu()
            .tolist()
        )
        slot_owners: dict[int, list[str]] = {}
        for request_id, locations in zip(live_request_ids, live_locations):
            for location in set(locations[: self.live_rows[request_id][1]]):
                slot_owners.setdefault(location, []).append(request_id)
        write_locations = forward_batch.out_cache_loc.cpu().tolist()
        write_lengths = (
            [1] * len(requests)
            if forward_batch.forward_mode.is_decode()
            else forward_batch.extend_seq_lens_cpu
        )
        write_start = 0
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
            if request.request_id in self.observations:
                observation = self.observations[request.request_id]
            else:
                assert not decode
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
                self.observations[request.request_id] = observation
            prefix_length = min(sequence_length, observation.prefix_locations.numel())
            write_end = write_start + write_lengths[row_index]
            for location in write_locations[write_start:write_end]:
                foreign_owners = [
                    owner
                    for owner in slot_owners[location]
                    if owner != request.request_id
                ]
                if foreign_owners:
                    observation.write_conflicts.append(
                        (sequence_length, location, foreign_owners)
                    )
                else:
                    pass
            checks = [
                (read_locations[row_index, :sequence_length] == locations).all(),
                metadata.cache_seqlens_int32[row_index] == sequence_length,
                (locations > 0).all(),
                (locations < self.pool.size + self.runner.page_size).all(),
                forward_batch.positions[write_end - 1] == sequence_length - 1,
                forward_batch.out_cache_loc[write_end - 1] == locations[-1],
            ]
            write_start = write_end
            for key_buffer, value_buffer, expected_keys, expected_values in zip(
                self.pool.k_buffer,
                self.pool.v_buffer,
                observation.prefix_keys,
                observation.prefix_values,
            ):
                checks.extend(
                    [
                        (
                            key_buffer.index_select(0, locations[:prefix_length])
                            == expected_keys[:prefix_length]
                        ).all(),
                        (
                            value_buffer.index_select(0, locations[:prefix_length])
                            == expected_values[:prefix_length]
                        ).all(),
                    ]
                )
            observation.checks.append(torch.stack(checks))
            observation.sequence_lengths.append(sequence_length)
            observation.prefix_locations = locations.clone()
            observation.prefix_keys = [
                buffer.index_select(0, locations) for buffer in self.pool.k_buffer
            ]
            observation.prefix_values = [
                buffer.index_select(0, locations) for buffer in self.pool.v_buffer
            ]
            if len(observation.snapshots) < 4:
                snapshot = {
                    "request_locations": locations.clone(),
                    "read_locations": read_locations[
                        row_index, :sequence_length
                    ].clone(),
                    "sequence_length": forward_batch.seq_lens[row_index].clone(),
                    "request_pool_index": request_pool_indices[row_index].clone(),
                }
                if not decode:
                    assert request.request_id in self.prefill_inputs, (
                        self.directory.name,
                        request.request_id,
                        forward_batch.rids,
                        list(self.prefill_inputs),
                    )
                    snapshot.update(self.prefill_inputs.pop(request.request_id))
                else:
                    pass
                if len(requests) == 1:
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
        self.live_rows.pop(request_id)
        checks = torch.stack(observation.checks).cpu()
        report = {
            "request_id": request_id,
            "history_tokens": observation.prefix_locations.numel(),
            "forward_count": len(observation.checks),
            "sequence_lengths": observation.sequence_lengths,
            "failed_checks": (~checks).nonzero().tolist(),
            "write_conflicts": observation.write_conflicts,
            "write_counts": {
                name: tensor.cpu().tolist()
                for name, tensor in self.attention_tensors.items()
                if name.endswith(".diagnostic_kv_write_counts")
            },
            "check_columns": [
                "read_locations",
                "sequence_length",
                "positive_locations",
                "bounded_locations",
                "last_write_position",
                "last_write_location",
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
