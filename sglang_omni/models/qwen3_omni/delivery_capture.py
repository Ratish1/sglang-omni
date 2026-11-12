"""Observe actual multimodal hidden producers and request-local delivery."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateAccumulator
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.qwen3_omni.layer_operand_capture import LayerOperandRecorder
from sglang_omni.scheduling.sglang_backend.prompt_hidden import (
    PromptHiddenRequest,
    PromptHiddenStore,
)


class HiddenCapture(Protocol):
    def __call__(self, hidden_states: torch.Tensor, *, owned: bool = False) -> None: ...


@dataclass(kw_only=True)
class ThinkerDeliveryRecorder:
    directory: Path
    request_ids: frozenset[str]
    multimodal_token_ids: tuple[int, ...]
    operands: LayerOperandRecorder
    write_count: int = 0
    slot_writers: dict[int, tuple[int, int]] = field(default_factory=dict)

    def install(self, layer: torch.nn.Module) -> None:
        original_forward = layer.forward

        def forward(
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            captured_last_layer_outputs: AuxHiddenStateAccumulator | None = None,
            capture_output: HiddenCapture | None = None,
            post_residual_addition: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if forward_batch.forward_mode.is_extend():
                assert capture_output is not None

                def capture(hidden_rows: torch.Tensor, *, owned: bool = False) -> None:
                    self.operands.begin(positions, is_prefill=True)
                    self.operands.record("accepted_hidden", hidden_rows)
                    self.operands.record("input_ids", forward_batch.input_ids)
                    capture_output(hidden_rows, owned=owned)

                selected_capture = capture
            else:
                selected_capture = capture_output

            return original_forward(
                positions,
                hidden_states,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                capture_output=selected_capture,
                post_residual_addition=post_residual_addition,
            )

        layer.forward = forward

    def record_inputs(
        self,
        input_embeds: torch.Tensor,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
    ) -> None:
        offset = 0
        for request, prefix, length in zip(
            schedule_batch.reqs,
            schedule_batch.prefix_lens,
            schedule_batch.extend_lens,
            strict=True,
        ):
            if request.rid in self.request_ids:
                destination = self.directory / "inputs"
                destination.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "request_id": request.rid,
                        "forward_step": int(self.operands.step_counter),
                        "prefix_length": prefix,
                        "extend_length": length,
                        "batch_offset": offset,
                        "embedded_input": input_embeds[offset : offset + length]
                        .detach()
                        .cpu()
                        .clone(),
                        "actual_input_ids": forward_batch.input_ids[
                            offset : offset + length
                        ]
                        .cpu()
                        .clone(),
                        "mrope_positions": forward_batch.mrope_positions[
                            :, offset : offset + length
                        ]
                        .cpu()
                        .clone(),
                        "causal_token_ids": torch.tensor(request.origin_input_ids),
                    },
                    destination
                    / f"{request.rid}-{int(self.operands.step_counter):06d}.pt",
                )
            else:
                pass
            offset += length

    def record_writes(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
    ) -> None:
        if not forward_batch.capture_hidden_mode.is_full():
            return
        else:
            pass
        assert result.logits_output.hidden_states is not None
        step = int(self.operands.step_counter)
        virtual_slots = (
            forward_batch.out_cache_loc
            if forward_batch.out_cache_loc_virtual is None
            else forward_batch.out_cache_loc_virtual
        )
        offset = 0
        for request, prefix, length in zip(
            schedule_batch.reqs,
            schedule_batch.prefix_lens,
            schedule_batch.extend_lens,
            strict=True,
        ):
            if isinstance(request, PromptHiddenRequest):
                relative_rows = request.prompt_hidden_offsets(prefix, length)
            else:
                relative_rows = torch.empty(0, dtype=torch.long)
            if relative_rows.numel():
                source = self.operands.read(
                    step,
                    offset + length - 1,
                    prefill_rows=slice(offset, offset + length),
                )
                batch_rows = (relative_rows + offset).to(virtual_slots.device)
                accepted = source["prefix_accepted_hidden"][relative_rows]
                auxiliary = result.logits_output.hidden_states[batch_rows].cpu().clone()
                assert torch.equal(accepted, auxiliary)
                slots = virtual_slots[batch_rows].cpu().clone()
                self.write_count += 1
                destination = self.directory / "writes"
                destination.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "request_id": request.rid,
                        "write_id": self.write_count,
                        "forward_step": step,
                        "prefix_length": prefix,
                        "extend_length": length,
                        "batch_offset": offset,
                        "prompt_positions": relative_rows + prefix,
                        "packed_rows": relative_rows + offset,
                        "virtual_slots": slots,
                        "physical_slots": forward_batch.out_cache_loc[batch_rows]
                        .cpu()
                        .clone(),
                        "producer_hidden": accepted,
                        "accepted_full_hidden": source["prefix_accepted_hidden"],
                        "auxiliary_hidden": auxiliary,
                        "positions": source["prefix_positions"][relative_rows],
                        "actual_input_ids": source["prefix_input_ids"],
                        "causal_token_ids": torch.tensor(request.origin_input_ids),
                        "scheduled_token_ids": torch.tensor(request.get_fill_ids()),
                    },
                    destination / f"{self.write_count:06d}.pt",
                )
                for row, slot in enumerate(slots.tolist()):
                    self.slot_writers[slot] = (self.write_count, row)
            else:
                pass
            offset += length

    def record_handoff(
        self,
        request: PromptHiddenRequest,
        store: PromptHiddenStore,
        outgoing: torch.Tensor,
    ) -> None:
        if request.rid not in self.request_ids:
            return
        else:
            pass
        positions = request.prompt_hidden_positions
        slots = store.req_to_token_pool.req_to_token[
            request.kv.req_pool_idx,
            positions.to(store.req_to_token_pool.req_to_token.device),
        ].cpu()
        assert len(slots) == len(outgoing) == len(positions)
        destination = self.directory / request.rid
        destination.mkdir(parents=True, exist_ok=False)
        torch.save(
            {
                "write_count": self.write_count,
                "request_pool_index": request.kv.req_pool_idx,
                "prompt_positions": positions.clone(),
                "virtual_slots": slots.clone(),
                "writers": torch.tensor(
                    [self.slot_writers[int(slot)] for slot in slots]
                ),
                "causal_token_ids": torch.tensor(request.origin_input_ids),
                "multimodal_token_ids": torch.tensor(self.multimodal_token_ids),
                "mrope_positions": (
                    request.multimodal_inputs.mrope_positions[:, positions]
                    .cpu()
                    .clone()
                    if request.multimodal_inputs is not None
                    and request.multimodal_inputs.mrope_positions is not None
                    else torch.empty(0)
                ),
                "outgoing_hidden": outgoing.clone(),
            },
            destination / "handoff.pt",
        )
        (destination / "identity.json").write_text(
            json.dumps({"request_id": request.rid})
        )
