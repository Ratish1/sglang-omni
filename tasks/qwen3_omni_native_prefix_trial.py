# SPDX-License-Identifier: Apache-2.0
"""Carry a captured prefix after repeated native decode controls."""

import json
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import validate
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.scheduling.types import SchedulerRequest


class NativePrefixTrial:
    def __init__(self, control: Path, donor: Path, destination: Path) -> None:
        self.control: list[dict[str, torch.Tensor]] = validate(control)
        self.donor: list[dict[str, torch.Tensor]] = validate(donor)
        self.destination: Path = destination
        self.request_id: str = json.loads((control / "trajectory.json").read_text())[
            "request_id"
        ]
        assert (
            json.loads((donor / "trajectory.json").read_text())["request_id"]
            == self.request_id
        )
        for field in (
            "input_ids",
            "positions",
            "rotary_positions",
            "codec_ids",
            "feedback",
        ):
            assert torch.equal(self.control[0][field], self.donor[0][field]), field
        for control_frame, donor_frame in zip(self.control[1:], self.donor[1:]):
            assert torch.equal(
                control_frame["text_embedding"], donor_frame["text_embedding"]
            )
        self.capture_hashes: list[str] = [
            sha256((directory / "trajectory.pt").read_bytes()).hexdigest()
            for directory in (control, donor)
        ]
        self.applied: bool = False

    @torch.inference_mode()
    def forward(
        self,
        worker: ModelWorker,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        selected_row: int,
        frames: list[dict[str, torch.Tensor]],
    ) -> GenerationBatchResult:
        assert not self.applied and len(frames) == 2
        assert (
            forward_batch.forward_mode.is_decode() and forward_batch.spec_info is None
        )
        for field in (
            "prefill_embeddings",
            "input_ids",
            "positions",
            "rotary_positions",
            "stored_keys",
            "stored_values",
            "codec_ids",
            "feedback",
        ):
            assert torch.equal(frames[0][field].cpu(), self.control[0][field]), field
        for field in (
            "positions",
            "rotary_positions",
            "queued_feedback",
            "text_embedding",
        ):
            assert torch.equal(frames[1][field].cpu(), self.control[1][field]), field
        runner = worker.model_runner
        model = runner.model
        pool = runner.token_to_kv_pool
        assert isinstance(pool, MHATokenToKVPool)
        buffers = (*pool.k_buffer, *pool.v_buffer)
        prefix_tokens = self.control[0]["stored_keys"].shape[1]
        assert int(forward_batch.seq_lens_cpu[selected_row]) == prefix_tokens + 1
        selected_slots = runner.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[selected_row], :prefix_tokens
        ].long()
        assert selected_slots.unique().numel() == prefix_tokens
        write_slots = forward_batch.out_cache_loc.long()
        assert write_slots.numel() == len(requests)
        assert write_slots.unique().numel() == len(requests) and (write_slots > 0).all()
        write_slots = torch.cat((write_slots, write_slots.new_zeros(1)))
        all_prefix_slots = (
            torch.cat(
                [
                    runner.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices[row], : int(length) - 1
                    ]
                    for row, length in enumerate(forward_batch.seq_lens_cpu)
                ]
            )
            .long()
            .unique()
        )
        assert not torch.isin(all_prefix_slots, write_slots).any()
        peer_prefix_slots = all_prefix_slots[
            ~torch.isin(all_prefix_slots, selected_slots)
        ]
        original_prefix = [buffer.index_select(0, selected_slots) for buffer in buffers]
        donor_prefix = [
            tensor.to(device=selected_slots.device)
            for field in ("stored_keys", "stored_values")
            for tensor in self.donor[0][field]
        ]
        expected_original = [
            tensor
            for field in ("stored_keys", "stored_values")
            for tensor in self.control[0][field]
        ]
        for original, expected, donor in zip(
            original_prefix, expected_original, donor_prefix, strict=True
        ):
            assert torch.equal(original.cpu(), expected)
            assert original.shape == donor.shape and original.dtype == donor.dtype
        previous_writes = [buffer.index_select(0, write_slots) for buffer in buffers]
        peer_prefix = [buffer.index_select(0, peer_prefix_slots) for buffer in buffers]
        feedback_mask = model.feedback_mask.clone()
        inputs = {
            name: getattr(model, name).clone()
            for name in (
                "feedback_buffer",
                "repetition_mask",
                "suppress_mask",
                "repetition_penalties",
                "sampling_seeds",
                "sampling_temperatures",
                "sampling_top_ks",
                "sampling_top_ps",
                "sampling_min_ps",
            )
        }
        outputs: list[dict[str, torch.Tensor]] = []
        writes: list[list[torch.Tensor]] = []
        for alternative in (False, True, False, True):
            prefix = donor_prefix if alternative else original_prefix
            for buffer, previous, prefix_values in zip(
                buffers, previous_writes, prefix, strict=True
            ):
                buffer.index_copy_(0, write_slots, previous)
                buffer.index_copy_(0, selected_slots, prefix_values)
            model.feedback_mask.copy_(feedback_mask)
            result = worker.forward_batch_generation(forward_batch)
            assert result.can_run_cuda_graph
            outputs.append(
                {
                    "logits": result.logits_output.next_token_logits[
                        : len(requests)
                    ].clone(),
                    "query": result.logits_output.hidden_states[
                        : len(requests)
                    ].clone(),
                    "codes": model.output_codes[: len(requests)].clone(),
                    "feedback": model.output_embeds[: len(requests)].clone(),
                    "main_codes": model.sampled_token_ids[: len(requests)].clone(),
                }
            )
            writes.append([buffer.index_select(0, write_slots) for buffer in buffers])
            for buffer, prefix_values, peers in zip(
                buffers, prefix, peer_prefix, strict=True
            ):
                assert torch.equal(
                    buffer.index_select(0, selected_slots), prefix_values
                )
                assert torch.equal(buffer.index_select(0, peer_prefix_slots), peers)
            for name, expected in inputs.items():
                assert torch.equal(getattr(model, name), expected), name
        torch.save(
            {
                "selected_row": selected_row,
                "request_ids": [request.request_id for request in requests],
                "outputs": [
                    {name: tensor.cpu() for name, tensor in output.items()}
                    for output in outputs
                ],
            },
            self.destination.with_suffix(".pt"),
        )
        peer_rows = (
            torch.arange(len(requests), device=selected_slots.device) != selected_row
        )
        for name in outputs[0]:
            assert torch.equal(outputs[0][name], outputs[2][name]), (
                "native_repeat",
                name,
            )
            assert torch.equal(outputs[1][name], outputs[3][name]), (
                "donor_repeat",
                name,
            )
            assert torch.equal(
                outputs[0][name][peer_rows], outputs[1][name][peer_rows]
            ), ("peers", name)
            assert all(torch.isfinite(output[name]).all() for output in outputs)
        for layer_index in range(len(buffers)):
            assert torch.equal(writes[0][layer_index], writes[2][layer_index])
            assert torch.equal(writes[1][layer_index], writes[3][layer_index])
            assert torch.equal(
                writes[0][layer_index][: len(requests)][peer_rows],
                writes[1][layer_index][: len(requests)][peer_rows],
            ), ("peer_cache", layer_index)
        for field, expected_field in (
            ("logits", "logits"),
            ("query", "predictor_query"),
            ("codes", "codec_ids"),
            ("feedback", "feedback"),
        ):
            assert torch.equal(
                outputs[0][field][selected_row].cpu(), self.control[1][expected_field]
            ), ("captured_native_control", field)
        for field, tensors in (
            ("stored_keys", writes[0][: len(pool.k_buffer)]),
            ("stored_values", writes[0][len(pool.k_buffer) :]),
        ):
            actual = torch.stack(
                [tensor[selected_row : selected_row + 1] for tensor in tensors]
            )
            assert torch.equal(actual.cpu(), self.control[1][field]), field
        for field in ("keys", "values"):
            frames[0][f"decode_prefix_{field}"] = self.donor[0][
                f"stored_{field}"
            ].clone()
        with self.destination.open("x") as receipt:
            json.dump(
                {
                    "request_id": self.request_id,
                    "capture_sha256": self.capture_hashes,
                    "selected_row": selected_row,
                    "request_rows": len(requests),
                    "graph_rows": runner.decode_cuda_graph_runner.bs,
                    "prefix_tokens": prefix_tokens,
                    "native_capture_control_exact": True,
                    "native_repeat_exact": True,
                    "donor_repeat_exact": True,
                    "peer_outputs_and_cache_unchanged": True,
                    "sampling_and_feedback_inputs_unchanged": True,
                    "carried_donor_prefix": True,
                    "changed_prefix_values": sum(
                        int((left != right).count_nonzero())
                        for left, right in zip(
                            original_prefix, donor_prefix, strict=True
                        )
                    ),
                    "native_codes": outputs[0]["codes"][selected_row].tolist(),
                    "carried_codes": outputs[-1]["codes"][selected_row].tolist(),
                },
                receipt,
                indent=2,
            )
        self.applied = True
        return result
