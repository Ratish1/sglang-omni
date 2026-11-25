# SPDX-License-Identifier: Apache-2.0
"""Compare two recorded audio inputs in one held native Thinker prefill."""

import json
from bisect import bisect_left
from hashlib import sha256
from pathlib import Path

import torch
from compare_qwen3_omni_native_trajectories import difference
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs


class ThinkerInputTrial:
    def __init__(self, first: Path, second: Path, destination: Path) -> None:
        self.destination: Path = destination
        self.applied: bool = False
        self.captures: list[dict[str, torch.Tensor]] = [
            torch.load(path, map_location="cpu", weights_only=True)
            for path in (first, second)
        ]
        metadata = [
            json.loads(path.with_suffix(".json").read_text())
            for path in (first, second)
        ]
        self.request_id: str = metadata[0]["request_id"]
        self.prefix_tokens: int = metadata[0]["prefix_tokens"]
        self.extend_tokens: int = metadata[0]["extend_tokens"]
        for field in ("request_id", "prefix_tokens", "extend_tokens"):
            assert metadata[0][field] == metadata[1][field], field
        for field in ("input_ids", "positions", "mrope_positions"):
            assert torch.equal(self.captures[0][field], self.captures[1][field]), field
        hidden_captures = [
            torch.load(
                path.with_name(path.name.replace("thinker-input", "thinker-hidden")),
                map_location="cpu",
                weights_only=True,
            )
            for path in (first, second)
        ]
        assert torch.equal(
            hidden_captures[0]["prompt_positions"],
            hidden_captures[1]["prompt_positions"],
        )
        self.audio_offsets: torch.Tensor = (
            hidden_captures[0]["prompt_positions"] - self.prefix_tokens
        )
        assert self.audio_offsets.numel() > 0
        assert (
            (self.audio_offsets >= 0) & (self.audio_offsets < self.extend_tokens)
        ).all()
        self.capture_hashes: dict[str, str] = {
            str(path): sha256(path.read_bytes()).hexdigest()
            for capture_path in (first, second)
            for path in (
                capture_path,
                capture_path.with_suffix(".json"),
                capture_path.with_name(
                    capture_path.name.replace("thinker-input", "thinker-hidden")
                ),
            )
        }

    @torch.inference_mode()
    def forward(
        self,
        worker: ModelWorker,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
    ) -> GenerationBatchResult:
        assert not self.applied
        assert (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_mixed()
        )
        assert (
            forward_batch.spec_info is None and forward_batch.spec_algorithm.is_none()
        )
        assert forward_batch.out_cache_loc_virtual is None
        assert get_parallel().tp_size == get_parallel().pp_size == 1
        request_ids = [request.rid for request in schedule_batch.reqs]
        selected_request = request_ids.index(self.request_id)
        prefix_lengths = forward_batch.extend_prefix_lens_cpu
        extend_lengths = forward_batch.extend_seq_lens_cpu
        assert prefix_lengths[selected_request] == self.prefix_tokens
        assert extend_lengths[selected_request] == self.extend_tokens
        start = sum(extend_lengths[:selected_request])
        end = start + self.extend_tokens
        prefill_inputs = get_omni_prefill_inputs(forward_batch)
        assert prefill_inputs is not None and forward_batch.input_embeds is None
        input_embeddings = prefill_inputs.input_embeds
        assert input_embeddings.dtype == torch.bfloat16
        assert sum(extend_lengths) == input_embeddings.shape[0]
        native_embeddings = input_embeddings.clone()
        for field in ("input_ids", "positions"):
            assert torch.equal(
                getattr(forward_batch, field)[start:end].cpu(), self.captures[0][field]
            ), field
        assert torch.equal(
            forward_batch.mrope_positions[:, start:end].cpu(),
            self.captures[0]["mrope_positions"],
        )
        assert torch.equal(
            schedule_batch.reqs[selected_request].prompt_hidden_offsets(
                self.prefix_tokens, self.extend_tokens
            ),
            self.audio_offsets,
        )
        audio_rows = torch.zeros(self.extend_tokens, dtype=torch.bool)
        audio_rows[self.audio_offsets] = True
        for capture in self.captures:
            assert capture["input_embeds"].shape == input_embeddings[start:end].shape
            assert torch.equal(
                capture["input_embeds"][~audio_rows],
                native_embeddings[start:end].cpu()[~audio_rows],
            )
        alternatives = [
            capture["input_embeds"].to(input_embeddings.device)
            for capture in self.captures
        ]
        runner = worker.model_runner
        assert runner.eplb_manager is None and not runner.enable_elastic_ep
        pool = runner.token_to_kv_pool
        assert isinstance(pool, MHATokenToKVPool)
        buffers = (*pool.k_buffer, *pool.v_buffer)
        write_slots = forward_batch.out_cache_loc.long()
        assert write_slots.numel() == sum(extend_lengths)
        assert (
            write_slots.unique().numel() == write_slots.numel()
            and (write_slots > 0).all()
        )
        prefix_slots = []
        row_start = 0
        for row, (prefix_tokens, extend_tokens) in enumerate(
            zip(prefix_lengths, extend_lengths, strict=True)
        ):
            slots = runner.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices[row]
            ]
            assert torch.equal(
                slots[prefix_tokens : prefix_tokens + extend_tokens].long(),
                write_slots[row_start : row_start + extend_tokens],
            )
            prefix_slots.append(slots[:prefix_tokens])
            row_start += extend_tokens
        prefix_slots = torch.cat(prefix_slots).long().unique()
        assert not torch.isin(prefix_slots, write_slots).any()
        original_writes = [
            buffer.index_select(0, write_slots).cpu() for buffer in buffers
        ]
        original_padding = [buffer[:1].clone() for buffer in buffers]
        original_prefix = [
            buffer.index_select(0, prefix_slots).cpu() for buffer in buffers
        ]
        selected_prefix_slots = runner.req_to_token_pool.req_to_token[
            forward_batch.req_pool_indices[selected_request], : self.prefix_tokens
        ].long()
        selected_prefix = [
            buffer.index_select(0, selected_prefix_slots).cpu() for buffer in buffers
        ]
        held_fields = {
            field: getattr(forward_batch, field).clone()
            for field in (
                "input_ids",
                "positions",
                "mrope_positions",
                "req_pool_indices",
                "seq_lens",
                "out_cache_loc",
            )
        }
        peer_tokens = torch.ones(write_slots.numel(), dtype=torch.bool)
        peer_tokens[start:end] = False
        peer_requests = torch.arange(len(request_ids)) != selected_request
        outputs: list[dict[str, torch.Tensor]] = []
        write_controls: dict[str, list[torch.Tensor]] = {}
        selected_writes: dict[str, list[torch.Tensor]] = {}
        peer_write_changes: dict[str, list[int]] = {}
        graph_modes: list[bool] = []
        graph_buckets: list[int | None] = []
        order = ("native", "first", "second", "first", "second", "native")
        for phase in order:
            input_embeddings.copy_(native_embeddings)
            if phase == "native":
                pass
            else:
                input_embeddings[start:end].copy_(
                    alternatives[0 if phase == "first" else 1]
                )
            expected_embeddings = input_embeddings.clone()
            for buffer, saved, padding in zip(
                buffers, original_writes, original_padding, strict=True
            ):
                buffer.index_copy_(0, write_slots, saved.to(buffer.device))
                buffer[:1].copy_(padding)
            result = worker.forward_batch_generation(forward_batch)
            assert result.next_token_ids is None
            graph_modes.append(bool(result.can_run_cuda_graph))
            if result.can_run_cuda_graph:
                buckets = runner.prefill_cuda_graph_runner.capture_num_tokens
                graph_buckets.append(buckets[bisect_left(buckets, write_slots.numel())])
            else:
                graph_buckets.append(None)
            assert graph_modes[-1] == graph_modes[0]
            assert graph_buckets[-1] == graph_buckets[0]
            assert get_omni_prefill_inputs(forward_batch) is prefill_inputs
            assert forward_batch.input_embeds is None
            assert torch.equal(input_embeddings, expected_embeddings)
            for field, saved in held_fields.items():
                assert torch.equal(getattr(forward_batch, field), saved), field
            output = {
                "hidden_states": result.logits_output.hidden_states.detach()
                .cpu()
                .clone(),
                "logits": result.logits_output.next_token_logits.detach().cpu().clone(),
            }
            assert output["hidden_states"].shape[0] == write_slots.numel()
            assert output["logits"].shape[0] == len(request_ids)
            assert all(torch.isfinite(tensor).all() for tensor in output.values())
            outputs.append(output)
            current_writes = []
            changed_peer_values = []
            for index, (buffer, saved_prefix) in enumerate(
                zip(buffers, original_prefix, strict=True)
            ):
                assert torch.equal(
                    buffer.index_select(0, prefix_slots).cpu(), saved_prefix
                ), (phase, "prefix", index)
                written = buffer.index_select(0, write_slots).cpu()
                if phase in write_controls:
                    assert torch.equal(written, write_controls[phase][index]), (
                        phase,
                        "repeat_writes",
                        index,
                    )
                else:
                    current_writes.append(written)
                if phase != "native":
                    changed_peer_values.append(
                        int(
                            (
                                written[peer_tokens]
                                != write_controls["native"][index][peer_tokens]
                            ).count_nonzero()
                        )
                    )
                else:
                    pass
            if phase not in write_controls:
                write_controls[phase] = current_writes
                selected_writes[phase] = [
                    written[start:end].clone() for written in current_writes
                ]
                peer_write_changes[phase] = changed_peer_values
            else:
                pass
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        selected_outputs = [
            {
                "hidden_states": output["hidden_states"][start:end].clone(),
                "logits": output["logits"][selected_request].clone(),
            }
            for output in outputs
        ]
        with self.destination.with_suffix(".pt").open("xb") as capture_file:
            torch.save(
                {
                    "outputs": selected_outputs,
                    "writes": selected_writes,
                    "audio_rows": audio_rows,
                    "selected_prefix": selected_prefix,
                    "native_input": native_embeddings[start:end].cpu(),
                },
                capture_file,
            )
        for left, right in ((0, 5), (1, 3), (2, 4)):
            assert all(
                torch.equal(outputs[left][field], outputs[right][field])
                for field in outputs[left]
            ), ("repeat_output", left, right)
        peer_output_changes = [
            {
                "hidden_states": int(
                    (
                        output["hidden_states"][peer_tokens]
                        != outputs[0]["hidden_states"][peer_tokens]
                    ).count_nonzero()
                ),
                "logits": int(
                    (
                        output["logits"][peer_requests]
                        != outputs[0]["logits"][peer_requests]
                    ).count_nonzero()
                ),
            }
            for output in outputs
        ]
        assert torch.equal(input_embeddings, native_embeddings)
        with self.destination.open("x") as receipt:
            json.dump(
                {
                    "request_id": self.request_id,
                    "request_ids": request_ids,
                    "prefix_lengths": list(prefix_lengths),
                    "extend_lengths": list(extend_lengths),
                    "selected_request": selected_request,
                    "capture_sha256": self.capture_hashes,
                    "order": order,
                    "native_repeat_exact": True,
                    "alternative_repeats_exact": True,
                    "peer_output_changes": peer_output_changes,
                    "peer_write_changes": peer_write_changes,
                    "prefix_and_input_metadata_unchanged": True,
                    "published_native_state": True,
                    "cuda_graph": graph_modes,
                    "graph_bucket_tokens": graph_buckets,
                    "input_transport": "omni_prefill_sidecar",
                    "input_difference": difference(
                        self.captures[0]["input_embeds"],
                        self.captures[1]["input_embeds"],
                    ),
                    "audio_hidden_difference": difference(
                        selected_outputs[1]["hidden_states"][audio_rows],
                        selected_outputs[2]["hidden_states"][audio_rows],
                    ),
                    "logit_difference": difference(
                        selected_outputs[1]["logits"], selected_outputs[2]["logits"]
                    ),
                    "key_value_differences": [
                        difference(first, second)
                        for first, second in zip(
                            selected_writes["first"],
                            selected_writes["second"],
                            strict=True,
                        )
                    ],
                },
                receipt,
                indent=2,
            )
        self.applied = True
        return result
