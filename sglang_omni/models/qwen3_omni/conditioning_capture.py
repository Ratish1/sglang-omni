"""Retain actual Talker conditioning for closed-loop checkpoint comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass(kw_only=True)
class TalkerConditioningRecorder:
    directory: Path
    request_ids: frozenset[str] | None = None
    assistant_tokens: dict[str, list[int]] = field(default_factory=dict)

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
    ) -> None:
        if not self.should_capture(request_id):
            return
        else:
            pass
        destination = self.directory / request_id
        destination.mkdir(parents=True, exist_ok=True)
        self.assistant_tokens[request_id] = assistant_token_ids.tolist()
        torch.save(
            {
                "prompt_input_ids": prompt_input_ids.cpu().clone(),
                "initial_assistant_token_ids": assistant_token_ids.cpu().clone(),
                "prompt_hidden": prompt_hidden.cpu().clone(),
                "speaker_id": torch.tensor(speaker_id),
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
    ) -> None:
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
                assert forward_batch.extend_prefix_lens_cpu[row] == 0
                rows = forward_batch.extend_seq_lens_cpu[row]
                selected = slice(offset, offset + rows)
                captured = {
                    "prefill_embeds": input_embeds[selected],
                    "input_ids": forward_batch.input_ids[selected],
                    "positions": forward_batch.positions[selected],
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
                phase = "before_decode"
            torch.save(
                {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in captured.items()
                },
                destination / f"{frame_index:04d}-{phase}.pt",
            )

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
            },
            self.directory / request_id / "finished.pt",
        )
