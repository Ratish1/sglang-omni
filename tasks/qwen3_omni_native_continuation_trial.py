# SPDX-License-Identifier: Apache-2.0
"""Retain one request's alternative decode state over an observed split window."""

import json
from pathlib import Path

import torch
from qwen3_omni_native_attention_trial import AttentionTrial
from qwen3_omni_native_code_trial import assert_native_prefix
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker


class NativeContinuationTrial:
    def __init__(
        self,
        reference: Path,
        target: Path,
        start_sequence_tokens: int,
        stop_sequence_tokens: int,
        destination: Path,
    ) -> None:
        self.reference: list[dict[str, torch.Tensor]] = torch.load(
            reference / "trajectory.pt", map_location="cpu", weights_only=True
        )
        self.target: list[dict[str, torch.Tensor]] = torch.load(
            target / "trajectory.pt", map_location="cpu", weights_only=True
        )
        self.request_id: str = json.loads((reference / "trajectory.json").read_text())[
            "request_id"
        ]
        assert (
            self.request_id
            == json.loads((target / "trajectory.json").read_text())["request_id"]
        )
        self.start_sequence_tokens: int = start_sequence_tokens
        self.stop_sequence_tokens: int = stop_sequence_tokens
        assert start_sequence_tokens < stop_sequence_tokens
        self.destination: Path = destination
        self.sequence_tokens: list[int] = []
        self.probe: dict[str, torch.Tensor] = {}
        self.scheduler_metadata: torch.Tensor | None = None
        self.sequence_bound_tokens: int = 0
        self.first_frame_index: int = -1

    @torch.inference_mode()
    def publish(
        self,
        worker: ModelWorker,
        forward_batch: ForwardBatch,
        trial: AttentionTrial,
        selected_row: int,
        frames: list[dict[str, torch.Tensor]],
    ) -> None:
        sequence_tokens = int(forward_batch.seq_lens_cpu[selected_row])
        if not self.sequence_tokens:
            assert sequence_tokens == self.start_sequence_tokens
            assert_native_prefix(frames, self.reference)
            self.first_frame_index = len(frames) - 1
            target = self.target[self.first_frame_index]
            for output_name, target_name in (
                ("logits", "logits"),
                ("query", "predictor_query"),
                ("codes", "codec_ids"),
                ("feedback", "feedback"),
            ):
                assert torch.equal(
                    trial.selected_outputs[output_name].cpu(), target[target_name]
                )
        else:
            assert sequence_tokens == self.sequence_tokens[-1] + 1
        runner = worker.model_runner
        pool = runner.token_to_kv_pool
        assert isinstance(pool, MHATokenToKVPool)
        assert trial.result.next_token_ids is None
        buffers = (*pool.k_buffer, *pool.v_buffer)
        selected_slot = forward_batch.out_cache_loc[selected_row : selected_row + 1]
        slots = (
            torch.cat(
                [
                    runner.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices[row], : int(length)
                    ]
                    for row, length in enumerate(forward_batch.seq_lens_cpu)
                ]
            )
            .long()
            .unique()
        )
        preserved_slots = slots[slots != selected_slot[0]]
        preserved_cache = [
            buffer.index_select(0, preserved_slots) for buffer in buffers
        ]
        outputs = {
            "logits": trial.result.logits_output.next_token_logits,
            "query": trial.result.logits_output.hidden_states,
            "codes": runner.model.output_codes,
            "feedback": runner.model.output_embeds,
            "main_codes": runner.model.sampled_token_ids,
        }
        originals = {name: tensor.clone() for name, tensor in outputs.items()}
        for buffer, alternative in zip(buffers, trial.selected_cache, strict=True):
            buffer.index_copy_(0, selected_slot.long(), alternative)
            assert torch.equal(
                buffer.index_select(0, selected_slot.long()), alternative
            )
        for name, tensor in outputs.items():
            tensor[selected_row].copy_(trial.selected_outputs[name])
            assert torch.equal(tensor[selected_row], trial.selected_outputs[name])
            assert torch.equal(tensor[:selected_row], originals[name][:selected_row])
            assert torch.equal(
                tensor[selected_row + 1 :], originals[name][selected_row + 1 :]
            )
        assert int(outputs["codes"][selected_row, 0]) == int(
            outputs["main_codes"][selected_row]
        )
        for buffer, preserved in zip(buffers, preserved_cache, strict=True):
            assert torch.equal(buffer.index_select(0, preserved_slots), preserved)
        self.probe = trial.selected_probe
        self.scheduler_metadata = trial.scheduler_metadata
        self.sequence_bound_tokens = trial.sequence_bound_tokens
        self.sequence_tokens.append(sequence_tokens)

    def finish(self, frames: list[dict[str, torch.Tensor]]) -> None:
        final_sequence_tokens = int(frames[-1]["positions"].flatten()[0]) + 1
        assert self.sequence_tokens == list(
            range(
                self.start_sequence_tokens,
                min(self.stop_sequence_tokens, final_sequence_tokens + 1),
            )
        )
        first_differences: dict[str, int] = {}
        for frame_index, (actual, target) in enumerate(zip(frames, self.target)):
            for name in (
                "predictor_query",
                "logits",
                "codec_ids",
                "feedback",
                "queued_feedback",
                "text_embedding",
                "stored_keys",
                "stored_values",
            ):
                if name in actual and name not in first_differences:
                    if not torch.equal(actual[name].cpu(), target[name]):
                        first_differences[name] = frame_index
                    else:
                        pass
                else:
                    pass
        with self.destination.open("x") as receipt:
            json.dump(
                {
                    "request_id": self.request_id,
                    "frame_index": self.first_frame_index,
                    "sequence_tokens": self.sequence_tokens,
                    "requested_stop_sequence_tokens": self.stop_sequence_tokens,
                    "prefix_exact": True,
                    "old_prefixes_and_peers_unchanged": True,
                    "target_frames": len(self.target),
                    "actual_frames": len(frames),
                    "target_first_differences": first_differences,
                },
                receipt,
                indent=2,
            )
