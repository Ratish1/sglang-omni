# SPDX-License-Identifier: Apache-2.0
"""Compare native attention schedules with restored decode state."""

import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from compare_qwen3_omni_native_trajectories import difference
from replay_qwen3_omni_scheduler_bounds import physical_schedule
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.scheduling.types import SchedulerRequest


@torch.inference_mode()
def compare_attention_schedules(
    worker: ModelWorker,
    forward_batch: ForwardBatch,
    requests: list[SchedulerRequest],
    selected_row: int,
    destination: Path,
    attention_splits: int,
) -> GenerationBatchResult:
    runner = worker.model_runner
    model = runner.model
    pool = runner.token_to_kv_pool
    graph_runner = runner.decode_cuda_graph_runner
    backend = graph_runner.attn_backend
    assert isinstance(pool, MHATokenToKVPool)
    assert isinstance(backend, FlashAttentionBackend)
    assert graph_runner._metadata_glue is None
    assert forward_batch.spec_info is None
    assert backend.fa_impl_ver == 3 and backend.page_size == 1
    assert backend.num_splits == backend.decode_num_splits == 0
    assert not backend.needs_cpu_seq_lens
    assert len(pool.k_buffer) == len(model.model.layers)
    buffers = (*pool.k_buffer, *pool.v_buffer)
    write_slots = forward_batch.out_cache_loc.long()
    assert write_slots.numel() == len(requests)
    assert write_slots.unique().numel() == write_slots.numel()
    assert (write_slots > 0).all()
    write_slots = torch.cat((write_slots, write_slots.new_zeros(1)))
    previous_writes = [buffer.index_select(0, write_slots) for buffer in buffers]
    prefix_slots = (
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
    assert not torch.isin(prefix_slots, write_slots).any()
    prefix_values = [buffer.index_select(0, prefix_slots) for buffer in buffers]
    feedback_mask = model.feedback_mask.clone()
    input_buffers = {
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
    original_schedule = backend._compute_scheduler_metadata
    tight_bounds: list[int] = []

    def tight_schedule(
        batch_size: int,
        max_seq_len_k: int,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
    ) -> torch.Tensor | None:
        assert not torch.cuda.is_current_stream_capturing()
        sequence_bound = int(cache_seqlens.max())
        assert sequence_bound <= max_seq_len_k
        tight_bounds.append(sequence_bound)
        with patch.object(backend, "num_splits", attention_splits):
            return original_schedule(
                batch_size, sequence_bound, cache_seqlens, cu_seqlens_q
            )

    outputs: list[dict[str, torch.Tensor]] = []
    writes: list[list[torch.Tensor]] = []
    schedules: list[list[int]] = []
    for alternative in (False, True, False):
        for buffer, previous in zip(buffers, previous_writes, strict=True):
            buffer.index_copy_(0, write_slots, previous)
        model.feedback_mask.copy_(feedback_mask)
        context = (
            patch.object(backend, "_compute_scheduler_metadata", tight_schedule)
            if alternative
            else nullcontext()
        )
        with context:
            result = worker.forward_batch_generation(forward_batch)
        assert result.can_run_cuda_graph
        snapshot = model.native_layer_probe.snapshot_for_batch(graph_runner.bs, True)
        outputs.append(
            {
                "logits": result.logits_output.next_token_logits[
                    : len(requests)
                ].clone(),
                "query": result.logits_output.hidden_states[: len(requests)].clone(),
                "codes": model.output_codes[: len(requests)].clone(),
                "feedback": model.output_embeds[: len(requests)].clone(),
                "main_codes": model.sampled_token_ids[: len(requests)].clone(),
                "predictor_logits": torch.stack(
                    [
                        snapshot[f"predictor_logits_{group}"][: len(requests), -1]
                        for group in range(1, model.config.num_code_groups)
                    ],
                    dim=1,
                ),
            }
        )
        outputs[-1].update(
            {
                f"layer0_{name}": snapshot[name][: len(requests)].clone()
                for name in (
                    "attention_context",
                    "mlp_normalized",
                    "expert_ids",
                    "layer_output",
                )
            }
        )
        writes.append([buffer.index_select(0, write_slots) for buffer in buffers])
        for buffer, expected in zip(buffers, prefix_values, strict=True):
            assert torch.equal(buffer.index_select(0, prefix_slots), expected)
        for name, expected in input_buffers.items():
            assert torch.equal(getattr(model, name), expected), name
        metadata = backend.forward_metadata
        assert metadata.scheduler_metadata is not None
        schedules.append(
            physical_schedule(metadata.scheduler_metadata.cpu(), graph_runner.bs)[
                0
            ].tolist()
        )
    assert len(tight_bounds) == 1
    reference, alternative, repeated = outputs
    for name in reference:
        assert torch.equal(reference[name], repeated[name]), name
        assert torch.isfinite(reference[name]).all(), name
        assert torch.isfinite(alternative[name]).all(), name
    for output in outputs:
        assert torch.equal(
            output["predictor_logits"].argmax(dim=-1), output["codes"][:, 1:]
        )
    assert all(
        torch.equal(first, last)
        for first, last in zip(writes[0], writes[2], strict=True)
    )
    assert schedules[0] == schedules[2]
    if attention_splits:
        assert schedules[1][: len(requests)] == [attention_splits] * len(requests)
    else:
        pass
    assert torch.equal(
        forward_batch.seq_lens[: len(requests)].cpu(), forward_batch.seq_lens_cpu
    )
    report = {
        "request_ids": [request.request_id for request in requests],
        "selected_row": selected_row,
        "sequence_tokens": forward_batch.seq_lens_cpu.tolist(),
        "graph_rows": graph_runner.bs,
        "tight_bound_tokens": tight_bounds[0],
        "requested_attention_splits": attention_splits,
        "splits": schedules,
        "stock_repeat_exact": True,
        "prefix_unchanged": True,
        "selected": {
            name: difference(
                reference[name][selected_row], alternative[name][selected_row]
            )
            for name in reference
        },
        "stock_codes": reference["codes"][selected_row].tolist(),
        "alternative_codes": alternative["codes"][selected_row].tolist(),
        "raw_codec_top2": [
            {
                "values": output["logits"][selected_row].topk(2).values.tolist(),
                "ids": output["logits"][selected_row].topk(2).indices.tolist(),
            }
            for output in (reference, alternative)
        ],
        "predictor_top2": [
            {
                "values": output["predictor_logits"][selected_row]
                .topk(2)
                .values.tolist(),
                "ids": output["predictor_logits"][selected_row]
                .topk(2)
                .indices.tolist(),
            }
            for output in (reference, alternative)
        ],
        "changed_peer_code_rows": (reference["codes"] != alternative["codes"])
        .any(dim=-1)
        .nonzero()
        .flatten()
        .tolist(),
        "changed_key_layers": [
            index
            for index in range(len(pool.k_buffer))
            if not torch.equal(writes[0][index], writes[1][index])
        ],
        "changed_value_layers": [
            index
            for index in range(len(pool.v_buffer))
            if not torch.equal(
                writes[0][index + len(pool.k_buffer)],
                writes[1][index + len(pool.k_buffer)],
            )
        ],
        "changed_selected_key_layers": [
            index
            for index in range(len(pool.k_buffer))
            if not torch.equal(
                writes[0][index][selected_row], writes[1][index][selected_row]
            )
        ],
        "changed_selected_value_layers": [
            index
            for index in range(len(pool.v_buffer))
            if not torch.equal(
                writes[0][index + len(pool.k_buffer)][selected_row],
                writes[1][index + len(pool.k_buffer)][selected_row],
            )
        ],
    }
    torch.save(
        {
            "reference": {name: value.cpu() for name, value in reference.items()},
            "alternative": {name: value.cpu() for name, value in alternative.items()},
            "sampling": {
                name: value[: len(requests)].cpu()
                for name, value in input_buffers.items()
                if name != "feedback_buffer"
            },
            "positions": forward_batch.positions.cpu(),
        },
        destination.with_suffix(".pt"),
    )
    with destination.open("x") as receipt:
        json.dump(report, receipt, indent=2)
    return result
