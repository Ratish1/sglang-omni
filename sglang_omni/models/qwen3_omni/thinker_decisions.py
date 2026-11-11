"""Observe native Thinker head inputs and published greedy decisions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.scheduling.types import SchedulerRequest

TOP_TOKEN_COUNT = 8


class ThinkerDecisionCapture:
    def __init__(self, directory: Path, hidden_states: torch.Tensor) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        request_ids = json.loads((directory / "request_ids.json").read_text())
        assert isinstance(request_ids, list) and all(
            isinstance(request_id, str) for request_id in request_ids
        )
        self.request_ids = set(request_ids)
        self.hidden_states = hidden_states
        self.row_count = torch.zeros((), device=hidden_states.device, dtype=torch.int64)
        self.step = 0

    def capture_hidden(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ) -> None:
        if forward_batch.forward_mode.is_extend():
            query_indices = forward_batch.extend_seq_lens.cumsum(0) - 1
            selected_hidden_states = hidden_states.index_select(0, query_indices)
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
        logits = result.logits_output.next_token_logits[:live_rows]
        token_ids = result.next_token_ids[:live_rows].cpu()
        greedy_ids = logits.argmax(-1).cpu()
        top_logits, top_ids = logits.topk(TOP_TOKEN_COUNT, dim=-1)
        finite = bool((~logits.isnan() & ~logits.isposinf()).all().cpu())
        assert finite
        assert torch.equal(token_ids, greedy_ids), (token_ids, greedy_ids)
        head_inputs = self.hidden_states[:live_rows].cpu().clone()
        assert bool(head_inputs.isfinite().all())
        filename = f"step-{self.step:06d}.pt"
        torch.save(
            {
                "head_inputs": head_inputs,
                "top_logits": top_logits.cpu(),
                "top_ids": top_ids.cpu(),
                "token_ids": token_ids,
                "greedy_ids": greedy_ids,
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
        }
        with (self.directory / "steps.jsonl").open("a") as output:
            output.write(json.dumps(report) + "\n")
        self.step += 1
