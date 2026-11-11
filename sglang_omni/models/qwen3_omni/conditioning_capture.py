"""Retain actual Talker conditioning for closed-loop checkpoint comparisons."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from sglang_omni.models.qwen3_omni.layer_operand_capture import (
    LayerOperandRecorder,
    MoeTacticChoice,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass(kw_only=True)
class TalkerConditioningRecorder:
    directory: Path
    key_buffers: list[torch.Tensor]
    value_buffers: list[torch.Tensor]
    request_slots: torch.Tensor
    request_ids: frozenset[str] | None = None
    assistant_tokens: dict[str, list[int]] = field(default_factory=dict)
    future_rows: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    previous_keys: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    previous_values: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    operator_recorders: list[LayerOperandRecorder] = field(default_factory=list)
    retain_moe_batches: bool = False
    moe_tactic_choices: list[MoeTacticChoice] = field(default_factory=list)

    def should_capture(self, request_id: str) -> bool:
        return self.request_ids is None or request_id in self.request_ids

    def record_prompt(
        self,
        request_id: str,
        prompt_input_ids: torch.Tensor,
        assistant_token_ids: torch.Tensor,
        prompt_hidden: torch.Tensor,
        speaker_id: int,
        model_inputs: dict[str, object],
        has_prompt_hidden: bool,
        future_text_rows: torch.Tensor | None,
    ) -> None:
        if not self.should_capture(request_id):
            return
        else:
            pass
        destination = self.directory / request_id
        destination.mkdir(parents=True, exist_ok=True)
        self.assistant_tokens[request_id] = assistant_token_ids.tolist()
        self.future_rows[request_id] = (
            [row.detach().cpu().clone() for row in future_text_rows]
            if future_text_rows is not None
            else []
        )
        torch.save(
            {
                "prompt_input_ids": prompt_input_ids.cpu().clone(),
                "initial_assistant_token_ids": assistant_token_ids.cpu().clone(),
                "prompt_hidden": prompt_hidden.cpu().clone(),
                "speaker_id": torch.tensor(speaker_id),
                "has_prompt_hidden": torch.tensor(has_prompt_hidden),
                **{
                    name: tensor.detach().cpu().clone()
                    for name in (
                        "audio_embeds",
                        "audio_feature_lengths",
                        "feature_attention_mask",
                    )
                    for tensor in [model_inputs.get(name)]
                    if isinstance(tensor, torch.Tensor)
                },
            },
            destination / "prompt.pt",
        )

    def record_inputs(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        input_embeds: torch.Tensor | None,
        sampling_seeds: torch.Tensor | None,
        repetition_mask: torch.Tensor | None = None,
    ) -> None:
        if self.operator_recorders:
            self.operator_recorders[0].step_counter.add_(1)
            if forward_batch.forward_mode.is_extend():
                self.operator_recorders[0].selected_prefill_token_row.zero_()
            else:
                pass
        else:
            pass
        offset = 0
        for row, request in enumerate(requests):
            if not self.should_capture(request.request_id):
                if forward_batch.forward_mode.is_extend():
                    offset += forward_batch.extend_seq_lens_cpu[row]
                else:
                    pass
                continue
            else:
                pass
            request_data: SGLangARRequestData = request.data
            native_request = request_data.req
            assert native_request is not None
            destination = self.directory / request.request_id
            destination.mkdir(parents=True, exist_ok=True)
            frame_index = len(native_request.output_ids)
            if forward_batch.forward_mode.is_extend():
                assert input_embeds is not None
                assert request_data.input_embeds_are_projected
                rows = forward_batch.extend_seq_lens_cpu[row]
                selected = slice(offset, offset + rows)
                if self.operator_recorders and self.retain_moe_batches:
                    assert self.request_ids is not None and len(self.request_ids) == 1
                    assert rows > 0
                    assert frame_index == 0
                    assert forward_batch.forward_mode == ForwardMode.EXTEND
                    self.operator_recorders[0].selected_prefill_token_row.fill_(
                        offset + rows - 1
                    )
                else:
                    pass
                captured = {
                    "prefill_embeds": input_embeds[selected],
                    "input_ids": forward_batch.input_ids[selected],
                    "positions": forward_batch.positions[selected],
                    "extend_prefix_length": torch.tensor(
                        forward_batch.extend_prefix_lens_cpu[row]
                    ),
                    "mrope_positions": (
                        forward_batch.mrope_positions[:, selected]
                        if forward_batch.mrope_positions is not None
                        else torch.empty(0)
                    ),
                }
                parameters = native_request.sampling_params
                (destination / "sampling.json").write_text(
                    json.dumps(
                        {
                            "temperature": parameters.temperature,
                            "top_k": parameters.top_k,
                            "top_p": parameters.top_p,
                            "min_p": parameters.min_p,
                            "repetition_penalty": parameters.repetition_penalty,
                            "sampling_seed": parameters.sampling_seed,
                            "max_new_tokens": parameters.max_new_tokens,
                            "stop_token_ids": sorted(parameters.stop_token_ids or []),
                            "suppress_tokens": request_data.suppress_tokens,
                        },
                        indent=2,
                    )
                )
                offset += rows
                phase = "before_prefill"
            else:
                assert sampling_seeds is not None
                queued_text = bool(request_data.pending_text_queue)
                next_text = (
                    request_data.pending_text_queue[0]
                    if queued_text
                    else request_data.tts_pad_embed
                )
                assert next_text is not None
                captured = {
                    "input_ids": forward_batch.input_ids[row : row + 1],
                    "positions": forward_batch.positions[row : row + 1],
                    "sampling_seed": sampling_seeds[row : row + 1],
                    "text_input": next_text,
                    "feedback_input": request_data.pending_feedback_queue[0],
                    "queued_text": torch.tensor(queued_text),
                }
                assert repetition_mask is not None
                captured["repetition_mask"] = repetition_mask[row]
                phase = "before_decode"
            captured["output_ids"] = torch.tensor(
                native_request.output_ids, dtype=torch.long
            )
            captured["history_length"] = torch.tensor(
                len(request_data.decode_input_embeds or [])
            )
            torch.save(
                {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in captured.items()
                },
                destination / f"{frame_index:04d}-{phase}.pt",
            )

    def record_logits(
        self, requests: list[SchedulerRequest], logits: torch.Tensor
    ) -> None:
        for row, request in enumerate(requests):
            if self.should_capture(request.request_id):
                destination = self.directory / request.request_id
                frame_index = len(request.data.req.output_ids)
                torch.save(
                    logits[row].detach().cpu().clone(),
                    destination / f"{frame_index:04d}-prefill-logits.pt",
                )
            else:
                pass

    def record_outputs(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        codes: torch.Tensor,
        feedback: torch.Tensor,
        logits: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        for row, request in enumerate(requests):
            if not self.should_capture(request.request_id):
                continue
            else:
                pass
            destination = self.directory / request.request_id
            frame_index = len(request.data.req.output_ids)
            sequence_length = int(forward_batch.seq_lens[row])
            locations = self.request_slots[
                forward_batch.req_pool_indices[row], :sequence_length
            ].long()
            keys = [
                buffer.index_select(0, locations).cpu() for buffer in self.key_buffers
            ]
            values = [
                buffer.index_select(0, locations).cpu() for buffer in self.value_buffers
            ]
            is_extend = forward_batch.forward_mode.is_extend()
            if request.request_id in self.previous_keys:
                prior_keys = self.previous_keys[request.request_id]
                prior_values = self.previous_values[request.request_id]
                past_exact = all(
                    torch.equal(current_keys[: len(previous_keys)], previous_keys)
                    and torch.equal(
                        current_values[: len(previous_values)], previous_values
                    )
                    for current_keys, current_values, previous_keys, previous_values in zip(
                        keys, values, prior_keys, prior_values, strict=True
                    )
                )
            else:
                past_exact = True
            self.previous_keys[request.request_id] = keys
            self.previous_values[request.request_id] = values
            torch.save(
                {
                    "codec_ids": codes[row].detach().cpu().clone(),
                    "codec_feedback": feedback[row].detach().cpu().clone(),
                    "logits": logits[row].detach().cpu().clone(),
                    "hidden_states": hidden_states[row].detach().cpu().clone(),
                    "keys": (
                        keys if is_extend else [tensor[-1:].clone() for tensor in keys]
                    ),
                    "values": (
                        values
                        if is_extend
                        else [tensor[-1:].clone() for tensor in values]
                    ),
                    "locations": locations.cpu(),
                    "past_exact": torch.tensor(past_exact),
                    "sequence_length": torch.tensor(sequence_length),
                    "request_pool_index": forward_batch.req_pool_indices[row]
                    .detach()
                    .cpu(),
                    "is_extend": torch.tensor(is_extend),
                },
                destination / f"{frame_index:04d}-output.pt",
            )
            if self.operator_recorders and (not is_extend or self.retain_moe_batches):
                step = int(self.operator_recorders[0].step_counter)
                if is_extend:
                    assert forward_batch.forward_mode == ForwardMode.EXTEND
                    assert frame_index == 0
                    assert forward_batch.extend_seq_lens_cpu[row] > 0
                else:
                    pass
                operand_row = (
                    sum(forward_batch.extend_seq_lens_cpu[: row + 1]) - 1
                    if is_extend
                    else row
                )
                torch.save(
                    [
                        recorder.read(
                            step,
                            operand_row,
                            retain_moe_batches=self.retain_moe_batches,
                        )
                        for recorder in self.operator_recorders
                    ],
                    destination / f"{frame_index:04d}-operators.pt",
                )
            else:
                pass

    def record_text_chunk(
        self, request_data: SGLangARRequestData, token_id: int
    ) -> None:
        request_id = request_data.req.rid
        if self.should_capture(request_id):
            self.assistant_tokens[request_id].append(token_id)
            with (self.directory / request_id / "text-events.jsonl").open(
                "a"
            ) as output:
                output.write(
                    json.dumps(
                        {
                            "token_id": token_id,
                            "thinker_done": request_data.thinker_chunks_done,
                            "output_frames": len(request_data.req.output_ids),
                            "queued_text_rows": len(request_data.pending_text_queue),
                        }
                    )
                    + "\n"
                )
        else:
            pass

    def finish(self, request_id: str, request_data: SGLangARRequestData) -> None:
        if not self.should_capture(request_id):
            return
        else:
            pass
        pending_rows = list(request_data.pending_text_queue)
        assert request_data.tts_pad_embed is not None
        assert request_data.tts_eos_embed is not None
        torch.save(
            {
                "pending_text": (
                    torch.stack(pending_rows).cpu()
                    if pending_rows
                    else request_data.tts_pad_embed.new_empty(
                        (0, request_data.tts_pad_embed.numel())
                    ).cpu()
                ),
                "tts_pad_embed": request_data.tts_pad_embed.cpu(),
                "tts_eos_embed": request_data.tts_eos_embed.cpu(),
                "thinker_done": torch.tensor(request_data.thinker_chunks_done),
                "assistant_token_ids": torch.tensor(
                    self.assistant_tokens.pop(request_id), dtype=torch.long
                ),
                "enqueued_text_rows": torch.stack(self.future_rows.pop(request_id)),
                "decode_input_history": torch.stack(
                    request_data.decode_input_embeds
                ).cpu(),
                "output_ids": torch.tensor(
                    request_data.req.output_ids, dtype=torch.long
                ),
                "finished_reason": request_data.req.finished_reason.to_json(),
            },
            self.directory / request_id / "finished.pt",
        )
        self.previous_keys.pop(request_id)
        self.previous_values.pop(request_id)
        if self.retain_moe_batches:
            (self.directory / "moe-choices-final.json").write_text(
                json.dumps(
                    [asdict(choice) for choice in self.moe_tactic_choices], indent=2
                )
            )
        else:
            pass

    def record_enqueue(self, request_id: str, row: torch.Tensor) -> None:
        if self.should_capture(request_id):
            self.future_rows[request_id].append(row.detach().cpu().clone())
        else:
            pass
