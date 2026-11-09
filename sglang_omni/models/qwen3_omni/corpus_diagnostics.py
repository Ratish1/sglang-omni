"""Retain every Talker frame and its actual operation boundaries for replay."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from pydantic import BaseModel
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch import nn

from sglang_omni.models.qwen3_omni.cache_history import CacheHistoryRecorder
from sglang_omni.models.qwen3_omni.conditioning_capture import (
    TalkerConditioningRecorder,
)
from sglang_omni.models.qwen3_omni.corpus_hooks import (
    AttentionCapture,
    OutputCapture,
    ProjectionCapture,
    TopKCapture,
)
from sglang_omni.models.qwen3_omni.expert_capture import FullExpertCapture
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


class StepMetadata(BaseModel):
    step: int
    phase: str
    captured_graph: bool
    request_ids: list[str]
    request_ordinals: list[int]
    frame_indices: list[int]
    live_rows: int
    predictor_batch: int
    projection_rows: dict[str, int]
    finite_including_padding: dict[str, bool]
    sequence_lengths: list[int]
    predictor_geometry: dict[str, int]
    predictor_stages: dict[str, int]


class RequestTrace(BaseModel):
    request_id: str
    ordinal: int
    steps: list[int]
    codes: list[list[int]]
    output_ids: list[int]
    main_codes_match: bool
    finish_reason: str


@dataclass(kw_only=True)
class CorpusProgress:
    ordinal: int
    codes: list[list[int]] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)


class TalkerCorpusRecorder:
    kv_pool: MHATokenToKVPool | None = None
    request_table: torch.Tensor | None = None
    predictor_counters: torch.Tensor | None = None
    full_experts: FullExpertCapture | None = None

    def __init__(
        self,
        directory: Path,
        layers: nn.ModuleList,
        codec_head: nn.Module,
        predictor_heads: nn.ModuleList,
        max_batch_size: int,
        hidden_size: int,
        predictor_layers: int,
        query_width: int,
        key_width: int,
        intermediate_size: int,
        device: torch.device,
    ) -> None:
        self.directory = directory
        self.cache_history = CacheHistoryRecorder(directory=directory)
        request_id_file = os.environ.get("SGLANG_OMNI_CORPUS_REQUEST_ID_FILE")
        self.conditioning = TalkerConditioningRecorder(
            directory=directory / "conditioning",
            request_ids=(
                frozenset(json.loads(Path(request_id_file).read_text()))
                if request_id_file is not None
                else None
            ),
        )
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "requests").mkdir()
        self.query_indices = torch.zeros(
            max_batch_size, device=device, dtype=torch.long
        )
        self.snapshots: dict[str, torch.Tensor] = {}
        self.counts: dict[str, torch.Tensor] = {}
        self.finite: dict[str, torch.Tensor] = {}
        self.progress: dict[str, CorpusProgress] = {}
        self.next_ordinal = 0
        self.step_index = 0
        self.predictor_stages: dict[tuple[str, int], int] = {}
        self.predictor_geometry: dict[str, int] = {}
        self.snapshots["backbone_input"] = torch.zeros(
            max_batch_size, hidden_size, device=device, dtype=torch.bfloat16
        )
        self.layer0_codes = torch.zeros(max_batch_size, device=device, dtype=torch.long)
        self.predictor_hidden = torch.zeros(
            max_batch_size, hidden_size, device=device, dtype=torch.bfloat16
        )
        self.predictor_batch = torch.zeros((), device=device, dtype=torch.int64)
        self.predictor_code_int32 = torch.zeros((), device=device, dtype=torch.bool)
        group_count = len(predictor_heads)
        self.dense_snapshots: dict[str, torch.Tensor] = {}
        for name, width in (
            ("input", hidden_size),
            ("query", query_width),
            ("attention", query_width),
            ("attention_residual", hidden_size),
            ("activation", intermediate_size),
            ("mlp_residual", hidden_size),
        ):
            self.dense_snapshots[name] = torch.zeros(
                group_count,
                predictor_layers,
                max_batch_size,
                width,
                device=device,
                dtype=torch.bfloat16,
            )
            self.dense_snapshots[f"opening_{name}"] = torch.zeros(
                predictor_layers,
                max_batch_size,
                width,
                device=device,
                dtype=torch.bfloat16,
            )
        for name in ("keys", "values"):
            self.dense_snapshots[name] = torch.zeros(
                group_count,
                predictor_layers,
                max_batch_size,
                2,
                key_width,
                device=device,
                dtype=torch.bfloat16,
            )
        self.dense_snapshots["seen"] = torch.zeros(
            group_count, predictor_layers, device=device, dtype=torch.bool
        )
        for layer_index, layer in enumerate(layers):
            prefix = f"backbone.{layer_index}"
            attention = layer.self_attn.attn
            for name, width in (
                ("query", attention.tp_q_head_num * attention.head_dim),
                ("key", attention.tp_k_head_num * attention.head_dim),
                ("value", attention.tp_v_head_num * attention.v_head_dim),
            ):
                self.snapshots[f"{prefix}.{name}"] = torch.zeros(
                    max_batch_size, width, device=device, dtype=torch.bfloat16
                )
            row_count = torch.zeros((), device=device, dtype=torch.int64)
            self.counts[f"{prefix}.attention"] = row_count
            attention.register_forward_hook(
                AttentionCapture(
                    self.query_indices,
                    self.snapshots[f"{prefix}.query"],
                    self.snapshots[f"{prefix}.key"],
                    self.snapshots[f"{prefix}.value"],
                    row_count,
                )
            )
            self.install_projection(
                f"{prefix}.qkv", layer.self_attn.qkv_proj, max_batch_size, True
            )
            self.install_projection(
                f"{prefix}.attention_projection",
                layer.self_attn.o_proj,
                max_batch_size,
                True,
            )
            self.install_projection(
                f"{prefix}.router", layer.mlp.gate, max_batch_size, True
            )
            self.install_projection(
                f"{prefix}.shared_gate",
                layer.mlp.shared_expert_gate,
                max_batch_size,
                True,
            )
            for name, module in (
                ("shared", layer.mlp.shared_expert),
                ("routed", layer.mlp.experts),
                ("combined", layer.mlp),
            ):
                snapshot = torch.zeros(
                    max_batch_size, hidden_size, device=device, dtype=torch.bfloat16
                )
                finite = torch.zeros((), device=device, dtype=torch.bool)
                self.snapshots[f"{prefix}.{name}"] = snapshot
                self.finite[f"{prefix}.{name}"] = finite
                row_count = torch.zeros((), device=device, dtype=torch.int64)
                self.counts[f"{prefix}.{name}"] = row_count
                module.register_forward_hook(
                    OutputCapture(snapshot, self.query_indices, finite, row_count)
                )
            expert_ids = torch.zeros(
                max_batch_size,
                layer.config.num_experts_per_tok,
                device=device,
                dtype=torch.int32,
            )
            weights = torch.zeros_like(expert_ids, dtype=torch.float32)
            self.snapshots[f"{prefix}.expert_ids"] = expert_ids
            self.snapshots[f"{prefix}.expert_weights"] = weights
            row_count = torch.zeros((), device=device, dtype=torch.int64)
            self.counts[f"{prefix}.topk"] = row_count
            layer.mlp.topk.register_forward_hook(
                TopKCapture(expert_ids, weights, self.query_indices, row_count)
            )
        self.install_projection("codec", codec_head, max_batch_size, False)
        for group_index, head in enumerate(predictor_heads):
            self.install_projection(
                f"predictor.{group_index}", head, max_batch_size, False
            )

    def install_projection(
        self, name: str, module: nn.Module, max_batch_size: int, gather_queries: bool
    ) -> None:
        weight = module.weight
        input_snapshot = torch.zeros(
            max_batch_size, weight.shape[1], device=weight.device, dtype=weight.dtype
        )
        output_snapshot = torch.zeros(
            max_batch_size, weight.shape[0], device=weight.device, dtype=weight.dtype
        )
        row_count = torch.zeros((), device=weight.device, dtype=torch.int64)
        finite = torch.zeros((), device=weight.device, dtype=torch.bool)
        self.snapshots[f"{name}.input"] = input_snapshot
        self.snapshots[f"{name}.output"] = output_snapshot
        self.counts[name] = row_count
        self.finite[name] = finite
        module.register_forward_hook(
            ProjectionCapture(
                input_snapshot,
                output_snapshot,
                row_count,
                finite,
                self.query_indices if gather_queries else None,
            )
        )

    def prepare(
        self, forward_batch: ForwardBatch, requests: list[SchedulerRequest]
    ) -> None:
        indices: list[int] = []
        if forward_batch.forward_mode.is_extend():
            length = 0
            for extend_length in forward_batch.extend_seq_lens_cpu:
                length += extend_length
                indices.append(length - 1)
        else:
            indices = list(range(len(requests)))
        assert len(indices) == len(requests)
        self.query_indices.zero_()
        self.query_indices[: len(indices)].copy_(
            torch.tensor(indices, device="cpu", dtype=torch.long)
        )
        for request in requests:
            if request.request_id not in self.progress:
                self.progress[request.request_id] = CorpusProgress(
                    ordinal=self.next_ordinal
                )
                self.next_ordinal += 1
            else:
                pass
        for count in self.counts.values():
            count.zero_()
        self.dense_snapshots["seen"].zero_()
        self.predictor_batch.zero_()
        if self.full_experts is not None:
            self.full_experts.prepare()
        else:
            pass

    def capture(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        codes: torch.Tensor | None,
        feedback: torch.Tensor | None,
        captured_graph: bool,
    ) -> None:
        if not any(
            self.conditioning.should_capture(request.request_id) for request in requests
        ):
            return
        else:
            pass
        predictor_batch = int(self.predictor_batch.cpu()) if codes is not None else 0
        live_rows = len(requests)
        snapshots = {
            name: tensor[
                : predictor_batch if name.startswith("predictor.") else live_rows
            ].cpu()
            for name, tensor in self.snapshots.items()
        }
        counts = {name: int(count.cpu()) for name, count in self.counts.items()}
        finite = {name: bool(value.cpu()) for name, value in self.finite.items()}
        assert all(
            count > 0
            for name, count in counts.items()
            if not name.startswith("predictor.")
        ), counts
        dense: dict[str, torch.Tensor] = {}
        for name, tensor in self.dense_snapshots.items():
            if name == "seen":
                dense[name] = tensor.cpu()
            elif name.startswith("opening_"):
                dense[name] = tensor[:, :predictor_batch].cpu()
            else:
                dense[name] = tensor[:, :, :predictor_batch].cpu()
        metadata = {
            "step": self.step_index,
            "phase": forward_batch.forward_mode.name,
            "captured_graph": captured_graph,
            "request_ids": [request.request_id for request in requests],
            "request_ordinals": [
                self.progress[request.request_id].ordinal for request in requests
            ],
            "frame_indices": [
                len(self.progress[request.request_id].codes) for request in requests
            ],
            "live_rows": live_rows,
            "predictor_batch": predictor_batch,
            "projection_rows": counts,
            "finite_including_padding": finite,
            "sequence_lengths": forward_batch.seq_lens[:live_rows].cpu().tolist(),
            "predictor_geometry": self.predictor_geometry,
            "predictor_stages": {
                f"{name}:{rows}": stages
                for (name, rows), stages in self.predictor_stages.items()
            },
        }
        assert self.kv_pool is not None and self.request_table is not None
        for row, request in enumerate(requests):
            if not self.conditioning.should_capture(request.request_id):
                continue
            else:
                pass
            locations = self.request_table[
                forward_batch.req_pool_indices[row],
                : metadata["sequence_lengths"][row],
            ].long()
            keys = [
                buffer.index_select(0, locations).cpu()
                for buffer in self.kv_pool.k_buffer
            ]
            values = [
                buffer.index_select(0, locations).cpu()
                for buffer in self.kv_pool.v_buffer
            ]
            if forward_batch.forward_mode.is_extend():
                self.cache_history.record_prefix(request.request_id, keys, values)
                torch.save(
                    {"keys": keys, "values": values},
                    self.directory
                    / f"prefix-{request.request_id}-{self.step_index:06d}.pt",
                )
            else:
                self.cache_history.verify_decode(
                    request.request_id,
                    metadata["frame_indices"][row],
                    keys,
                    values,
                    [
                        snapshots[f"backbone.{layer_index}.key"][row]
                        for layer_index in range(len(self.kv_pool.k_buffer))
                    ],
                    [
                        snapshots[f"backbone.{layer_index}.value"][row]
                        for layer_index in range(len(self.kv_pool.v_buffer))
                    ],
                )
        if self.full_experts is not None:
            real_rows = (
                sum(forward_batch.extend_seq_lens_cpu)
                if forward_batch.forward_mode.is_extend()
                else live_rows
            )
            self.full_experts.save(
                self.directory / f"experts-{self.step_index:06d}.pt",
                real_rows,
                self.query_indices[:live_rows],
                forward_batch,
            )
        else:
            pass
        if codes is None:
            cpu_codes = torch.empty(0, device="cpu", dtype=torch.long)
            cpu_feedback = torch.empty(0, device="cpu")
        else:
            assert feedback is not None and predictor_batch >= live_rows
            assert all(
                counts[f"predictor.{group_index}"] == predictor_batch
                for group_index in range(self.dense_snapshots["seen"].shape[0])
            )
            assert dense["seen"].all()
            cpu_codes = codes.cpu()
            cpu_feedback = feedback.cpu()
            for row, request in enumerate(requests):
                progress = self.progress[request.request_id]
                progress.codes.append(cpu_codes[row].tolist())
                progress.steps.append(self.step_index)
        torch.save(
            {
                "snapshots": snapshots,
                "dense": dense,
                "codes": cpu_codes,
                "feedback": cpu_feedback,
                "predictor_hidden": self.predictor_hidden[:predictor_batch].cpu(),
                "layer0_codes": self.layer0_codes[:predictor_batch].cpu(),
                "code_int32": self.predictor_code_int32.cpu(),
                "predictor_counters": self.predictor_counters.cpu(),
                "positions": forward_batch.positions.index_select(
                    0, self.query_indices[:live_rows]
                ).cpu(),
                "mrope_positions": (
                    forward_batch.mrope_positions.index_select(
                        1, self.query_indices[:live_rows]
                    ).cpu()
                    if forward_batch.mrope_positions is not None
                    else torch.empty(0, device="cpu")
                ),
            },
            self.directory / f"step-{self.step_index:06d}.pt",
        )
        with (self.directory / "steps.jsonl").open("a") as output:
            output.write(json.dumps(metadata) + "\n")
        self.step_index += 1

    def finish(self, request_id: str, request_data: SGLangARRequestData) -> None:
        if not self.conditioning.should_capture(request_id):
            self.progress.pop(request_id)
            return
        else:
            pass
        self.conditioning.finish(request_id, request_data)
        self.cache_history.finish(request_id)
        progress = self.progress.pop(request_id)
        assert request_data.req is not None
        report = {
            "request_id": request_id,
            "ordinal": progress.ordinal,
            "steps": progress.steps,
            "codes": progress.codes,
            "output_ids": list(request_data.req.output_ids),
            "finish_reason": request_data.finish_reason,
            "main_codes_match": [frame[0] for frame in progress.codes]
            == list(request_data.req.output_ids),
        }
        (self.directory / "requests" / f"{request_id}.json").write_text(
            json.dumps(report, indent=2)
        )
