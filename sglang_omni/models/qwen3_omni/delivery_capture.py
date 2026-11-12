"""Observe actual multimodal hidden producers and request-local delivery."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch
from sglang.srt.layers.aux_hidden_states import AuxHiddenStateAccumulator
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock

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
    is_prefill_active: bool = False
    write_count: int = 0
    slot_writers: dict[int, tuple[int, int]] = field(default_factory=dict)

    def install(self, layer: torch.nn.Module, layer_index: int) -> None:
        original_forward = layer.forward

        def forward(
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            captured_last_layer_outputs: AuxHiddenStateAccumulator | None = None,
            capture_output: HiddenCapture | None = None,
            post_residual_addition: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.is_prefill_active = forward_batch.forward_mode.is_extend()
            if forward_batch.forward_mode.is_extend():

                def capture(hidden_rows: torch.Tensor, *, owned: bool = False) -> None:
                    if layer_index == 0:
                        self.operands.begin(positions, is_prefill=True)
                        self.operands.record("input_ids", forward_batch.input_ids)
                    else:
                        pass
                    self.operands.record(f"layer_{layer_index:02d}_hidden", hidden_rows)
                    if capture_output is not None:
                        self.operands.record("accepted_hidden", hidden_rows)
                        capture_output(hidden_rows, owned=owned)
                    else:
                        pass

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

    def install_operators(self, layer: Qwen3MoeDecoderLayer, layer_index: int) -> None:
        """Observe real module calls without replacing their computations."""
        prefix = f"operator_{layer_index:02d}"
        linear_names = {
            layer.self_attn.qkv_proj: "qkv",
            layer.self_attn.o_proj: "attention_projection",
            layer.mlp.gate: "router",
        }

        def capture_linear(
            module: LinearBase,
            input_tensors: tuple[torch.Tensor],
            output_tensors: tuple[torch.Tensor, torch.Tensor | None],
        ) -> None:
            if self.is_prefill_active:
                name = f"{prefix}_{linear_names[module]}"
                self.operands.record(f"{name}_input", input_tensors[0])
                self.operands.record(f"{name}_output", output_tensors[0])
            else:
                pass

        def capture_topk(
            module: torch.nn.Module,
            input_tensors: tuple[torch.Tensor, torch.Tensor],
            output: TopKOutput,
        ) -> None:
            if self.is_prefill_active:
                assert isinstance(output, StandardTopKOutput)
                self.operands.record(f"{prefix}_expert_indices", output.topk_ids)
                self.operands.record(f"{prefix}_expert_weights", output.topk_weights)
            else:
                pass

        def capture_moe(
            module: Qwen3MoeSparseMoeBlock,
            input_tensors: tuple[torch.Tensor, ForwardBatch],
            output: torch.Tensor,
        ) -> None:
            if self.is_prefill_active:
                self.operands.record(f"{prefix}_moe_output", output)
            else:
                pass

        original_apply = layer.self_attn.apply_qk_norm_rope

        def apply_qk_norm_rope(
            qkv: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            query, key, value = original_apply(qkv, positions, forward_batch)
            if self.is_prefill_active:
                self.operands.record(f"{prefix}_query_rotated", query)
                self.operands.record(f"{prefix}_key_rotated", key)
                self.operands.record(f"{prefix}_value", value)
            else:
                pass
            return query, key, value

        for module in linear_names:
            module.register_forward_hook(capture_linear)
        layer.mlp.topk.register_forward_hook(capture_topk)
        layer.mlp.register_forward_hook(capture_moe)
        layer.self_attn.apply_qk_norm_rope = apply_qk_norm_rope

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
        key_buffers: list[torch.Tensor],
        value_buffers: list[torch.Tensor],
        request_slots: torch.Tensor,
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
                    retain_operator_batches=True,
                )
                batch_rows = (relative_rows + offset).to(virtual_slots.device)
                accepted = source["prefix_accepted_hidden"][relative_rows]
                auxiliary = result.logits_output.hidden_states[batch_rows].cpu().clone()
                assert torch.equal(accepted, auxiliary)
                slots = virtual_slots[batch_rows].cpu().clone()
                cache_locations = request_slots[
                    request.kv.req_pool_idx, : prefix + length
                ].long()
                assert torch.equal(
                    cache_locations[prefix:],
                    forward_batch.out_cache_loc[offset : offset + length].long(),
                )
                layer_count = sum(name.startswith("prefix_layer_") for name in source)
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
                        "layer_hidden": {
                            name.removeprefix("prefix_"): tensor
                            for name, tensor in source.items()
                            if name.startswith("prefix_layer_")
                        },
                        "operator_batches": {
                            name.removeprefix("batch_"): tensor
                            for name, tensor in source.items()
                            if name.startswith("batch_operator_")
                        },
                        "operator_layouts": {
                            name.removeprefix("layout_"): tensor
                            for name, tensor in source.items()
                            if name.startswith("layout_operator_")
                        },
                        "cache_locations": cache_locations.cpu().clone(),
                        "cache_keys": [
                            buffer.index_select(0, cache_locations).cpu()
                            for buffer in key_buffers[:layer_count]
                        ],
                        "cache_values": [
                            buffer.index_select(0, cache_locations).cpu()
                            for buffer in value_buffers[:layer_count]
                        ],
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
