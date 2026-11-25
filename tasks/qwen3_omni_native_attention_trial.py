# SPDX-License-Identifier: Apache-2.0
"""Compare native attention schedules with restored decode state."""

import json
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
from compare_qwen3_omni_native_trajectories import difference
from qwen3_omni_native_code_trial import assert_native_prefix
from replay_qwen3_omni_scheduler_bounds import physical_schedule
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass(kw_only=True)
class AttentionTrial:
    result: GenerationBatchResult
    selected_cache: list[torch.Tensor]
    selected_outputs_equal: bool
    selected_outputs: dict[str, torch.Tensor]
    selected_probe: dict[str, torch.Tensor]
    scheduler_metadata: torch.Tensor
    sequence_bound_tokens: int


class NativeCacheTrial:
    def __init__(
        self, reference: Path, destination: Path, target_reference: Path | None
    ) -> None:
        self.reference: list[dict[str, torch.Tensor]] = torch.load(
            reference / "trajectory.pt", map_location="cpu", weights_only=True
        )
        self.request_id: str = json.loads((reference / "trajectory.json").read_text())[
            "request_id"
        ]
        self.destination = destination
        self.applied = False
        self.target: list[dict[str, torch.Tensor]] | None = (
            torch.load(
                target_reference / "trajectory.pt",
                map_location="cpu",
                weights_only=True,
            )
            if target_reference is not None
            else None
        )
        if target_reference is not None:
            assert (
                json.loads((target_reference / "trajectory.json").read_text())[
                    "request_id"
                ]
                == self.request_id
            )
        else:
            pass

    @torch.inference_mode()
    def apply(
        self,
        worker: ModelWorker,
        forward_batch: ForwardBatch,
        trial: AttentionTrial,
        selected_row: int,
        frames: list[dict[str, torch.Tensor]],
    ) -> None:
        assert not self.applied
        assert trial.selected_outputs_equal
        assert_native_prefix(frames, self.reference)
        runner = worker.model_runner
        pool = runner.token_to_kv_pool
        assert isinstance(pool, MHATokenToKVPool)
        buffers = (*pool.k_buffer, *pool.v_buffer)
        if self.target is not None:
            target = self.target[len(frames) - 1]
            for field, tensors in (
                ("stored_keys", trial.selected_cache[: len(pool.k_buffer)]),
                ("stored_values", trial.selected_cache[len(pool.k_buffer) :]),
            ):
                assert torch.equal(torch.stack(tensors).cpu(), target[field]), field
            for field in ("positions", "rotary_positions", "codec_ids", "feedback"):
                assert torch.equal(frames[-1][field].cpu(), target[field]), field
        else:
            pass
        selected_slot = forward_batch.out_cache_loc[
            selected_row : selected_row + 1
        ].long()
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
        original_cache = [buffer.index_select(0, selected_slot) for buffer in buffers]
        original_codes = runner.model.output_codes.clone()
        original_feedback = runner.model.output_embeds.clone()
        changed_buffers = [
            index
            for index, (original, alternative) in enumerate(
                zip(original_cache, trial.selected_cache, strict=True)
            )
            if not torch.equal(original, alternative)
        ]
        assert changed_buffers
        assert 0 not in changed_buffers and len(pool.k_buffer) not in changed_buffers
        for buffer, alternative in zip(buffers, trial.selected_cache, strict=True):
            buffer.index_copy_(0, selected_slot, alternative)
            assert torch.equal(buffer.index_select(0, selected_slot), alternative)
        for buffer, preserved in zip(buffers, preserved_cache, strict=True):
            assert torch.equal(buffer.index_select(0, preserved_slots), preserved)
        assert torch.equal(runner.model.output_codes, original_codes)
        assert torch.equal(runner.model.output_embeds, original_feedback)
        torch.save(
            {
                "original_cache": [tensor.cpu() for tensor in original_cache],
                "alternative_cache": [tensor.cpu() for tensor in trial.selected_cache],
                "codes": original_codes[selected_row].cpu(),
                "feedback": original_feedback[selected_row].cpu(),
            },
            self.destination.with_suffix(".pt"),
        )
        with self.destination.open("x") as receipt:
            json.dump(
                {
                    "request_id": self.request_id,
                    "sequence_tokens": int(forward_batch.seq_lens_cpu[selected_row]),
                    "frame_index": len(frames) - 1,
                    "prefix_exact": True,
                    "selected_outputs_equal": True,
                    "target_state_exact": self.target is not None,
                    "old_prefixes_and_peers_unchanged": True,
                    "changed_key_layers": [
                        index for index in changed_buffers if index < len(pool.k_buffer)
                    ],
                    "changed_value_layers": [
                        index - len(pool.k_buffer)
                        for index in changed_buffers
                        if index >= len(pool.k_buffer)
                    ],
                },
                receipt,
                indent=2,
            )
        self.applied = True


@torch.inference_mode()
def compare_attention_schedules(
    worker: ModelWorker,
    forward_batch: ForwardBatch,
    requests: list[SchedulerRequest],
    selected_row: int,
    destination: Path,
    attention_splits: int,
    save_tensors: bool = True,
) -> AttentionTrial:
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
    layer_operands: list[list[dict[str, torch.Tensor]]] = []
    writes: list[list[torch.Tensor]] = []
    schedules: list[list[int]] = []
    selected_probe: dict[str, torch.Tensor] = {}
    alternative_scheduler_metadata: torch.Tensor | None = None
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
        if save_tensors:
            probes = (model.native_layer_probe, *model.native_deeper_layer_probes)
            layer_operands.append(
                [
                    {
                        name: tensor.detach().clone()
                        for name, tensor in probe.snapshot_for_batch(
                            graph_runner.bs, True
                        ).items()
                    }
                    for probe in probes
                ]
            )
        else:
            pass
        if alternative:
            selected_probe = model.native_layer_probe.read(
                selected_row, graph_runner.bs, True
            )
        else:
            pass
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
        if alternative:
            alternative_scheduler_metadata = metadata.scheduler_metadata.clone()
        else:
            pass
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
    if save_tensors:
        assert len(layer_operands) == 3
        for first_layer, last_layer in zip(
            layer_operands[0], layer_operands[2], strict=True
        ):
            assert first_layer.keys() == last_layer.keys()
            for name in first_layer:
                assert torch.equal(first_layer[name], last_layer[name]), name
    else:
        pass
    if attention_splits:
        # note (ratish): FA3 treats explicit split counts as an upper bound.
        assert all(
            1 <= split_count <= attention_splits
            for split_count in schedules[1][: len(requests)]
        )
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
        "captured_layer_count": len(layer_operands[0]) if save_tensors else 0,
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
    if save_tensors:
        torch.save(
            {
                "reference": {name: value.cpu() for name, value in reference.items()},
                "alternative": {
                    name: value.cpu() for name, value in alternative.items()
                },
                "reference_cache": [buffer[selected_row].cpu() for buffer in writes[0]],
                "alternative_cache": [
                    buffer[selected_row].cpu() for buffer in writes[1]
                ],
                "sampling": {
                    name: value[: len(requests)].cpu()
                    for name, value in input_buffers.items()
                    if name != "feedback_buffer"
                },
                "positions": forward_batch.positions.cpu(),
                "layer_operands": [
                    [
                        {name: tensor.cpu() for name, tensor in layer.items()}
                        for layer in forward_layers
                    ]
                    for forward_layers in layer_operands
                ],
            },
            destination.with_suffix(".pt"),
        )
    else:
        pass
    with destination.open("x") as receipt:
        json.dump(report, receipt, indent=2)
    assert alternative_scheduler_metadata is not None
    return AttentionTrial(
        result=result,
        selected_cache=[
            buffer[selected_row : selected_row + 1].clone() for buffer in writes[1]
        ],
        selected_outputs_equal=all(
            torch.equal(reference[name][selected_row], alternative[name][selected_row])
            for name in ("codes", "feedback", "main_codes")
        ),
        selected_outputs={
            name: alternative[name][selected_row].clone()
            for name in ("logits", "query", "codes", "feedback", "main_codes")
        },
        selected_probe=selected_probe,
        scheduler_metadata=alternative_scheduler_metadata,
        sequence_bound_tokens=tight_bounds[0],
    )
