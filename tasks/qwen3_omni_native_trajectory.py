# SPDX-License-Identifier: Apache-2.0
"""Record selected native Talker trajectories without changing model operands."""

import json
from os import environ
from pathlib import Path

import torch
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


class NativeTalkerTrajectory:
    def __init__(self) -> None:
        self.directory = Path(environ["OMNI_NATIVE_TRAJECTORY_DIRECTORY"])
        self.request_ids = {
            f"feedback-trace-native-{sample_id}"
            for sample_id in environ["OMNI_NATIVE_TRAJECTORY_SAMPLES"].split(",")
        }
        self.tensors: dict[str, list[dict[str, torch.Tensor]]] = {}
        self.metadata: dict[str, list[dict[str, str | int | bool | list[str]]]] = {}
        assert self.directory.is_dir()

    def before_forward(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        prefill_embeddings: torch.Tensor | None,
    ) -> None:
        token_offset = 0
        is_prefill = forward_batch.forward_mode.is_extend()
        for request_row, request in enumerate(requests):
            token_count = (
                forward_batch.extend_seq_lens_cpu[request_row] if is_prefill else 1
            )
            if request.request_id in self.request_ids:
                tensors = self.tensors.setdefault(request.request_id, [])
                metadata = self.metadata.setdefault(request.request_id, [])
                assert len(tensors) == len(metadata)
                token_slice = slice(token_offset, token_offset + token_count)
                snapshot = {
                    "positions": forward_batch.positions[token_slice].detach().clone(),
                    "output_slots": forward_batch.out_cache_loc[token_slice]
                    .detach()
                    .clone(),
                    "input_ids": forward_batch.input_ids[token_slice].detach().clone(),
                }
                if forward_batch.mrope_positions is not None:
                    snapshot["rotary_positions"] = (
                        forward_batch.mrope_positions[:, token_slice].detach().clone()
                    )
                else:
                    pass
                request_metadata = {
                    "phase": "prefill" if is_prefill else "decode",
                    "request_row": request_row,
                    "request_ids": [peer.request_id for peer in requests],
                    "batch_tokens": forward_batch.input_ids.numel(),
                    "sequence_length": int(forward_batch.seq_lens_cpu[request_row]),
                    "generation_steps": request.data.generation_steps,
                    "thinker_done": request.data.thinker_chunks_done,
                }
                if is_prefill:
                    assert prefill_embeddings is not None
                    snapshot["prefill_embeddings"] = (
                        prefill_embeddings[token_slice].detach().clone()
                    )
                    request_metadata["prefix_length"] = (
                        forward_batch.extend_prefix_lens_cpu[request_row]
                    )
                else:
                    snapshot["queued_feedback"] = (
                        request.data.pending_feedback_queue[0].detach().clone()
                    )
                    uses_padding = not bool(request.data.pending_text_queue)
                    text_embedding = (
                        request.data.tts_pad_embed
                        if uses_padding
                        else request.data.pending_text_queue[0]
                    )
                    assert text_embedding is not None
                    snapshot["text_embedding"] = text_embedding.detach().clone()
                    request_metadata["uses_text_padding"] = uses_padding
                tensors.append(snapshot)
                metadata.append(request_metadata)
            else:
                pass
            token_offset += token_count
        assert token_offset == forward_batch.input_ids.numel()

    def after_forward(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        model_runner: ModelRunner,
        hidden_states: torch.Tensor,
        logits: torch.Tensor,
        can_run_cuda_graph: bool,
    ) -> None:
        pool = model_runner.token_to_kv_pool
        assert isinstance(pool, MHATokenToKVPool)
        is_prefill = forward_batch.forward_mode.is_extend()
        for request_row, request in enumerate(requests):
            if request.request_id in self.request_ids:
                snapshot = self.tensors[request.request_id][-1]
                metadata = self.metadata[request.request_id][-1]
                snapshot["predictor_query"] = (
                    hidden_states[request_row].detach().clone()
                )
                snapshot["logits"] = logits[request_row].detach().clone()
                snapshot["codec_ids"] = (
                    model_runner.model.output_codes[request_row].detach().clone()
                )
                snapshot["feedback"] = (
                    model_runner.model.output_embeds[request_row].detach().clone()
                )
                metadata["cuda_graph"] = can_run_cuda_graph
                if is_prefill:
                    token_slots = (
                        model_runner.req_to_token_pool.req_to_token.index_select(
                            0,
                            forward_batch.req_pool_indices[
                                request_row : request_row + 1
                            ].long(),
                        )[0, : int(forward_batch.seq_lens_cpu[request_row])].long()
                    )
                    snapshot["logical_slots"] = token_slots.detach().clone()
                else:
                    token_slots = snapshot["output_slots"].long()
                    snapshot["consumed_input"] = (
                        model_runner.model.feedback_buffer[request_row].detach().clone()
                    )
                    metadata["graph_batch_rows"] = (
                        model_runner.decode_cuda_graph_runner.bs
                        if can_run_cuda_graph
                        else len(requests)
                    )
                    snapshot.update(
                        model_runner.model.native_layer_probe.read(
                            request_row,
                            metadata["graph_batch_rows"],
                            can_run_cuda_graph,
                        )
                    )
                    backend = (
                        model_runner.decode_cuda_graph_runner.attn_backend
                        if can_run_cuda_graph
                        else forward_batch.attn_backend
                    )
                    assert isinstance(backend, FlashAttentionBackend)
                    assert backend.fa_impl_ver == 3 and backend.page_size == 1
                    assert not backend.use_mla and not backend.is_prefill_aware_swa
                    assert forward_batch.spec_info is None
                    attention = model_runner.model.model.layers[0].self_attn.attn
                    assert not attention.is_cross_attention
                    assert attention.sliding_window_size in (None, -1)
                    assert attention.k_scale is None and attention.v_scale is None
                    attention_metadata = backend.forward_metadata
                    assert attention_metadata.local_attn_metadata is None
                    assert attention_metadata.max_seq_len_q == 1
                    sequence_length = metadata["sequence_length"]
                    read_slots = attention_metadata.page_table[
                        request_row, :sequence_length
                    ].long()
                    snapshot["attention_read_slots"] = read_slots.detach().clone()
                    snapshot["attention_owned_slots"] = (
                        model_runner.req_to_token_pool.req_to_token.index_select(
                            0,
                            forward_batch.req_pool_indices[
                                request_row : request_row + 1
                            ].long(),
                        )[0, :sequence_length]
                        .detach()
                        .clone()
                    )
                    key_cache, value_cache = backend.get_paged_mha_kv_cache(attention)
                    snapshot["attention_read_keys"] = key_cache[:, 0].index_select(
                        0, read_slots
                    )
                    snapshot["attention_read_values"] = value_cache[:, 0].index_select(
                        0, read_slots
                    )
                    snapshot["attention_sequence_lengths"] = (
                        attention_metadata.cache_seqlens_int32.detach().clone()
                    )
                    snapshot["attention_query_offsets"] = (
                        attention_metadata.cu_seqlens_q.detach().clone()
                    )
                    if attention_metadata.scheduler_metadata is not None:
                        snapshot["attention_scheduler_metadata"] = (
                            attention_metadata.scheduler_metadata.detach().clone()
                        )
                    else:
                        pass
                    metadata["attention_num_splits"] = backend.decode_num_splits
                snapshot["stored_keys"] = torch.stack(
                    [layer.index_select(0, token_slots) for layer in pool.k_buffer]
                )
                snapshot["stored_values"] = torch.stack(
                    [layer.index_select(0, token_slots) for layer in pool.v_buffer]
                )
            else:
                pass

    def finish(self, request_id: str, request: SGLangARRequestData) -> None:
        if request_id not in self.request_ids:
            return
        else:
            pass
        snapshots = self.tensors.pop(request_id)
        metadata = self.metadata.pop(request_id)
        destination = self.directory / request_id.removeprefix("feedback-trace-native-")
        destination.mkdir()
        torch.save(snapshots, destination / "trajectory.pt")
        with (destination / "trajectory.json").open("x") as report:
            json.dump(
                {
                    "request_id": request_id,
                    "finish_reason": request.finish_reason,
                    "output_ids": list(request.req.output_ids),
                    "frames": metadata,
                },
                report,
            )
