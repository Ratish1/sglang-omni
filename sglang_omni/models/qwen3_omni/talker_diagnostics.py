"""Check real Talker sampling buffers and feedback/text FIFO handoffs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


@dataclass(kw_only=True)
class TalkerStateObservation:
    text_rows: tuple[torch.Tensor, ...]
    feedback: torch.Tensor
    text_index: int = 0
    checks: list[tuple[str, torch.Tensor]] = field(default_factory=list)
    codes: list[torch.Tensor] = field(default_factory=list)
    decode_input: torch.Tensor | None = None


class TalkerStateObserver:
    def __init__(self, directory: Path) -> None:
        self.directory = directory / "talker_state"
        self.directory.mkdir()
        self.observations: dict[str, TalkerStateObservation] = {}

    def observe_output(
        self,
        model: Qwen3OmniTalker,
        requests: list[SchedulerRequest],
        codes: torch.Tensor,
        feedback: torch.Tensor,
    ) -> None:
        expected_feedback = model.get_input_embeddings()(codes[:, 0]).clone()
        for group, embedding in enumerate(model.code_predictor.model.codec_embedding):
            expected_feedback.add_(embedding(codes[:, group + 1]))
        for row, request in enumerate(requests):
            if request.request_id not in self.observations:
                assert request.data.thinker_chunks_done
                self.observations[request.request_id] = TalkerStateObservation(
                    text_rows=tuple(
                        tensor.clone() for tensor in request.data.pending_text_queue
                    ),
                    feedback=feedback[row].clone(),
                )
            else:
                pass
            observation = self.observations[request.request_id]
            observation.checks.append(
                ("codec_feedback", (feedback[row] == expected_feedback[row]).all())
            )
            observation.feedback = feedback[row].clone()
            observation.codes.append(codes[row].clone())

    def before_decode(
        self, model: Qwen3OmniTalker, requests: list[SchedulerRequest]
    ) -> None:
        for row, request in enumerate(requests):
            observation = self.observations[request.request_id]
            request_data = request.data
            parameters = request_data.req.sampling_params
            device = model.feedback_buffer.device
            repetition_mask = torch.zeros_like(model.repetition_mask[row])
            if parameters.repetition_penalty != 1.0:
                token_ids = [
                    token_id
                    for token_id in request_data.req.output_ids
                    if 0 <= token_id < repetition_mask.numel()
                ]
                repetition_mask[
                    torch.tensor(token_ids, device=device, dtype=torch.long)
                ] = True
            else:
                pass
            suppress_mask = torch.zeros_like(model.suppress_mask[row])
            suppress_tokens = (
                request_data.suppress_tokens or request_data.req._codec_suppress_tokens
            )  # noqa: leading-underscore  # SGLang request field.
            if suppress_tokens:
                token_ids = [
                    token_id
                    for token_id in suppress_tokens
                    if 0 <= token_id < suppress_mask.numel()
                ]
                suppress_mask[
                    torch.tensor(token_ids, device=device, dtype=torch.long)
                ] = True
            else:
                pass
            text = (
                observation.text_rows[observation.text_index]
                if observation.text_index < len(observation.text_rows)
                else request_data.tts_pad_embed.reshape(-1)
            )
            queued_text = (
                request_data.pending_text_queue[0].reshape(-1)
                if request_data.pending_text_queue
                else request_data.tts_pad_embed.reshape(-1)
            )
            seed = parameters.sampling_seed
            if seed is None:
                seed = derive_sampling_seed(
                    "sglang-omni-unseeded-row", request.request_id
                )
            else:
                pass
            checks = [
                (
                    "feedback_fifo",
                    (
                        request_data.pending_feedback_queue[0].reshape(-1)
                        == observation.feedback
                    ).all(),
                ),
                ("text_fifo", (queued_text == text.reshape(-1)).all()),
                (
                    "repetition_mask",
                    (model.repetition_mask[row] == repetition_mask).all(),
                ),
                ("suppress_mask", (model.suppress_mask[row] == suppress_mask).all()),
            ]
            for name, actual, expected in (
                (
                    "repetition_penalty",
                    model.repetition_penalties[row, 0],
                    parameters.repetition_penalty,
                ),
                (
                    "temperature",
                    model.sampling_temperatures[row, 0],
                    parameters.temperature,
                ),
                ("top_p", model.sampling_top_ps[row], parameters.top_p),
                ("top_k", model.sampling_top_ks[row], parameters.top_k),
                ("min_p", model.sampling_min_ps[row], parameters.min_p),
                ("sampling_seed", model.sampling_seeds[row], seed),
            ):
                checks.append((name, actual == expected))
            observation.checks.extend(checks)
            observation.decode_input = observation.feedback + text.reshape(-1)
            observation.text_index += 1

    def after_decode_inputs(
        self, model: Qwen3OmniTalker, requests: list[SchedulerRequest]
    ) -> None:
        for row, request in enumerate(requests):
            observation = self.observations[request.request_id]
            observation.checks.extend(
                [
                    ("feedback_mask", model.feedback_mask[row].clone()),
                    (
                        "feedback_buffer",
                        (model.feedback_buffer[row] == observation.decode_input).all(),
                    ),
                    (
                        "decode_history",
                        (
                            request.data.decode_input_embeds[-1]
                            == observation.decode_input
                        ).all(),
                    ),
                ]
            )

    def finish(self, request_id: str, request_data: SGLangARRequestData) -> None:
        observation = self.observations.pop(request_id)
        checks = torch.stack([value for _, value in observation.checks]).cpu().tolist()
        codes = torch.stack(observation.codes).cpu()
        output_ids = list(request_data.req.output_ids)
        report = {
            "request_id": request_id,
            "text_rows": len(observation.text_rows),
            "decode_steps": observation.text_index,
            "codec_frames": codes.shape[0],
            "checks": len(checks),
            "failed_checks": [
                {"index": index, "name": name}
                for index, ((name, _), passed) in enumerate(
                    zip(observation.checks, checks)
                )
                if not passed
            ],
            "main_codes_match_request_output": codes[:, 0].tolist() == output_ids,
            "request_output_ids": list(output_ids),
            "codec_main_ids": codes[:, 0].tolist(),
            "finish_reason": request_data.finish_reason,
        }
        (self.directory / f"{request_id}.json").write_text(json.dumps(report, indent=2))
        torch.save(codes, self.directory / f"{request_id}.pt")
