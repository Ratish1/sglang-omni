"""Observe native Thinker head inputs and published greedy decisions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer
from torch import nn

from sglang_omni.scheduling.types import SchedulerRequest

TOP_TOKEN_COUNT = 8


class ThinkerBoundaryCapture:
    def __init__(
        self,
        query_indices: torch.Tensor,
        input_rows: torch.Tensor,
        output_rows: torch.Tensor,
        observed: torch.Tensor,
    ) -> None:
        self.query_indices = query_indices
        self.input_rows = input_rows
        self.output_rows = output_rows
        self.observed = observed

    def capture_input(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor] | tuple[torch.Tensor, ForwardBatch | None],
    ) -> None:
        if arguments[0].shape[0] == 0:
            return
        else:
            pass
        assert arguments[0].dtype == self.input_rows.dtype
        self.input_rows.copy_(arguments[0][self.query_indices])
        self.observed.fill_(1)

    def __call__(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor] | tuple[torch.Tensor, ForwardBatch | None],
        output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    ) -> None:
        tensor = output if isinstance(output, torch.Tensor) else output[0]
        if tensor.shape[0] == 0:
            return
        else:
            pass
        assert tensor.dtype == self.output_rows.dtype
        self.output_rows.copy_(tensor[self.query_indices])
        self.observed.add_(1)


class ThinkerRoutingCapture:
    def __init__(
        self,
        query_indices: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_weights: torch.Tensor,
        observed: torch.Tensor,
    ) -> None:
        self.query_indices = query_indices
        self.expert_ids = expert_ids
        self.expert_weights = expert_weights
        self.observed = observed

    def __call__(
        self,
        module: nn.Module,
        arguments: tuple[torch.Tensor, torch.Tensor],
        output: StandardTopKOutput,
    ) -> None:
        assert isinstance(output, StandardTopKOutput)
        if output.topk_ids.shape[0] == 0:
            return
        else:
            pass
        self.expert_ids.copy_(output.topk_ids[self.query_indices])
        self.expert_weights.copy_(output.topk_weights[self.query_indices])
        self.observed.fill_(2)


class ThinkerDecisionCapture:
    def __init__(
        self, directory: Path, hidden_states: torch.Tensor, layers: nn.ModuleList
    ) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        request_ids = json.loads((directory / "request_ids.json").read_text())
        assert isinstance(request_ids, list) and all(
            isinstance(request_id, str) for request_id in request_ids
        )
        self.request_ids = set(request_ids)
        self.hidden_states = hidden_states
        self.row_count = torch.zeros((), device=hidden_states.device, dtype=torch.int64)
        self.raw_logits_finite = torch.zeros(
            hidden_states.shape[0], device=hidden_states.device, dtype=torch.bool
        )
        self.layer_boundaries: dict[str, torch.Tensor] = {}
        self.boundaries_observed = torch.zeros(
            len(layers), 5, device=hidden_states.device, dtype=torch.int32
        )
        self.query_indices = torch.zeros(
            len(request_ids), device=hidden_states.device, dtype=torch.long
        )
        if os.environ.get("SGLANG_OMNI_THINKER_CAPTURE_LAYERS") == "1":
            for layer_index, layer in enumerate(layers):
                assert isinstance(layer, Qwen3MoeDecoderLayer)
                attention = layer.self_attn
                for boundary_index, (
                    name,
                    module,
                    input_width,
                    output_width,
                ) in enumerate(
                    (
                        (
                            "qkv",
                            attention.qkv_proj,
                            attention.hidden_size,
                            attention.q_size + 2 * attention.kv_size,
                        ),
                        (
                            "attention_projection",
                            attention.o_proj,
                            attention.q_size,
                            attention.hidden_size,
                        ),
                        (
                            "gate",
                            layer.mlp.gate,
                            attention.hidden_size,
                            layer.config.num_experts,
                        ),
                        (
                            "moe",
                            layer.mlp,
                            attention.hidden_size,
                            attention.hidden_size,
                        ),
                    )
                ):
                    input_rows = hidden_states.new_zeros(
                        (len(request_ids), input_width)
                    )
                    output_rows = hidden_states.new_zeros(
                        (len(request_ids), output_width)
                    )
                    self.layer_boundaries[f"layer.{layer_index}.{name}.input"] = (
                        input_rows
                    )
                    self.layer_boundaries[f"layer.{layer_index}.{name}.output"] = (
                        output_rows
                    )
                    boundary_capture = ThinkerBoundaryCapture(
                        self.query_indices,
                        input_rows,
                        output_rows,
                        self.boundaries_observed[layer_index, boundary_index],
                    )
                    module.register_forward_pre_hook(boundary_capture.capture_input)
                    module.register_forward_hook(boundary_capture)
                expert_ids = torch.zeros(
                    len(request_ids),
                    layer.config.num_experts_per_tok,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
                expert_weights = torch.zeros_like(expert_ids, dtype=torch.float32)
                self.layer_boundaries[f"layer.{layer_index}.expert_ids"] = expert_ids
                self.layer_boundaries[f"layer.{layer_index}.expert_weights"] = (
                    expert_weights
                )
                layer.mlp.topk.register_forward_hook(
                    ThinkerRoutingCapture(
                        self.query_indices,
                        expert_ids,
                        expert_weights,
                        self.boundaries_observed[layer_index, 4],
                    )
                )
        else:
            pass
        self.step = 0

    def prepare(self, forward_batch: ForwardBatch) -> None:
        if not self.layer_boundaries:
            return
        else:
            pass
        self.boundaries_observed.zero_()
        if forward_batch.forward_mode.is_extend():
            indices = forward_batch.extend_seq_lens.cumsum(0) - 1
        else:
            assert forward_batch.forward_mode.is_decode()
            indices = torch.arange(
                forward_batch.input_ids.shape[0], device=self.query_indices.device
            )
        self.query_indices.zero_()
        row_count = min(indices.numel(), self.query_indices.numel())
        self.query_indices[:row_count].copy_(indices[:row_count])

    def capture_hidden(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        if forward_batch.forward_mode.is_extend():
            query_indices = forward_batch.extend_seq_lens.cumsum(0) - 1
            selected_hidden_states = hidden_states[query_indices]
        else:
            assert forward_batch.forward_mode.is_decode()
            selected_hidden_states = hidden_states
        row_count = selected_hidden_states.shape[0]
        assert row_count <= self.hidden_states.shape[0]
        self.hidden_states[:row_count].copy_(selected_hidden_states)
        self.row_count.fill_(row_count)

    def record(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        request_ids = [request.request_id for request in requests]
        if not any(request_id in self.request_ids for request_id in request_ids):
            return
        else:
            pass
        assert set(request_ids) <= self.request_ids
        assert not schedule_batch.is_prefill_only
        assert forward_batch.sampling_info.is_all_greedy
        assert request_ids == [request.rid for request in schedule_batch.reqs]
        live_rows = len(requests)
        assert int(self.row_count.cpu()) >= live_rows
        assert bool(self.raw_logits_finite[:live_rows].all().cpu())
        logits = result.logits_output.next_token_logits[:live_rows]
        token_ids = result.next_token_ids[:live_rows].cpu()
        greedy_ids = logits.argmax(-1).cpu()
        top_logits, top_ids = logits.topk(TOP_TOKEN_COUNT, dim=-1)
        finite = bool((~logits.isnan() & ~logits.isposinf()).all().cpu())
        assert finite
        assert torch.equal(token_ids, greedy_ids), (token_ids, greedy_ids)
        head_inputs = self.hidden_states[:live_rows].cpu().clone()
        assert bool(head_inputs.isfinite().all())
        if self.layer_boundaries:
            assert bool((self.boundaries_observed == 2).all().cpu())
            assert live_rows <= self.query_indices.numel()
            layer_boundaries = {
                name: tensor[:live_rows].cpu().clone()
                for name, tensor in self.layer_boundaries.items()
            }
            assert all(
                bool(tensor.isfinite().all()) for tensor in layer_boundaries.values()
            )
        else:
            layer_boundaries = {}
        filename = f"step-{self.step:06d}.pt"
        torch.save(
            {
                "head_inputs": head_inputs,
                "top_logits": top_logits.cpu(),
                "top_ids": top_ids.cpu(),
                "token_ids": token_ids,
                "greedy_ids": greedy_ids,
                "raw_logits_finite": self.raw_logits_finite[:live_rows].cpu().clone(),
                "layer_boundaries": layer_boundaries,
            },
            self.directory / filename,
        )
        prefixes = [list(request.data.req.output_ids) for request in requests]
        report = {
            "step": self.step,
            "tensor_file": filename,
            "phase": forward_batch.forward_mode.name,
            "captured_graph": bool(result.can_run_cuda_graph),
            "request_ids": request_ids,
            "output_prefixes": prefixes,
            "remaining_prefill_chunks": [
                request.inflight_middle_chunks for request in schedule_batch.reqs
            ],
            "prompt_sha256": [
                hashlib.sha256(
                    json.dumps(list(request.data.req.origin_input_ids)).encode()
                ).hexdigest()
                for request in requests
            ],
            "sequence_lengths": forward_batch.seq_lens[:live_rows].cpu().tolist(),
            "token_ids": token_ids.tolist(),
            "top_ids": top_ids.cpu().tolist(),
            "top_logits": top_logits.cpu().tolist(),
            "hidden_sha256": [
                hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).hexdigest()
                for row in head_inputs
            ],
            "inputs_before_forward": True,
        }
        with (self.directory / "steps.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        self.step += 1
