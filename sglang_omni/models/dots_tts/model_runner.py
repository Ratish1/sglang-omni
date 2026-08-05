# SPDX-License-Identifier: Apache-2.0
"""Omni ModelRunner hooks for dots.tts continuous latent feedback."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner


class DotsTTSModelRunner(ModelRunner):
    """Use the shared SGLang forward path and own only latent recurrence."""

    def _require_single_request(self, requests: list) -> None:
        # ponytail: per-request DiT/patch KV is single-row for the base PR;
        # add a row pool when continuous batching is implemented.
        if len(requests) > 1:
            raise RuntimeError("dots.tts base support allows one running request")

    def before_prefill(
        self, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del schedule_batch
        self._require_single_request(requests)
        if not requests:
            return
        data = requests[0].data
        schedule = data.generation_schedule
        if schedule is None or data.span_positions is None:
            raise RuntimeError("dots.tts request is missing its generation schedule")
        flow_state, prompt_embeddings = self.model.flow.new_request(
            max_audio_patch_count=int(data.span_positions.numel()),
            prompt_latents=data.state.prompt_latents,
            speaker_embedding=data.state.speaker_embedding,
            speaker_scale=data.state.speaker_scale,
        )
        data.flow_state = flow_state
        device = forward_batch.input_ids.device
        prefill_ids = schedule[:, : data.prefill_end].to(device=device)
        inputs_embeds = self.model.get_input_embeddings()(prefill_ids).clone()
        prompt_positions = data.prompt_span_positions
        if prompt_positions is not None and prompt_positions.numel():
            if prompt_embeddings is None:
                raise RuntimeError("dots.tts prompt spans require prompt embeddings")
            inputs_embeds[:, prompt_positions.to(device=device), :] = (
                prompt_embeddings.to(device=device, dtype=inputs_embeds.dtype)
            )
        prefix_len = len(data.req.prefix_indices)
        req_len = int(data.req.extend_range.length)
        forward_batch.input_embeds = inputs_embeds[0, prefix_len : prefix_len + req_len]

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ) -> None:
        del schedule_batch, is_lookahead
        self._require_single_request(requests)
        if not requests:
            return
        queue = requests[0].data.pending_feedback_queue
        if not queue:
            raise RuntimeError("dots.tts decode is missing its latent feedback")
        feedback = queue.popleft()
        if feedback.ndim == 3:
            feedback = feedback[:, -1]
        forward_batch.input_embeds = feedback.to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.LAST

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        if not requests:
            return
        data = requests[0].data
        hidden = self._hidden_states(result)
        if hidden.ndim == 2:
            full_hidden = hidden.unsqueeze(0)
        elif hidden.ndim == 3:
            full_hidden = hidden
        else:
            raise RuntimeError(
                f"dots.tts expected rank-2/3 prefill hidden, got {hidden.ndim}"
            )
        self.model.flow.initialize_history(
            data.flow_state,
            hidden_states=full_hidden,
            prompt_span_positions=data.prompt_span_positions,
            audio_span_token_ids=set(data.state.audio_span_token_ids),
            generation_schedule=data.generation_schedule.to(full_hidden.device),
            prefill_end=data.prefill_end,
        )
        self._run_flow_step(result, data, full_hidden[:, -1:])

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del forward_batch, schedule_batch
        if not requests:
            return
        data = requests[0].data
        hidden = self._hidden_states(result)
        if hidden.ndim == 2:
            hidden = hidden.unsqueeze(0)
        elif hidden.ndim != 3:
            raise RuntimeError(
                f"dots.tts expected rank-2/3 decode hidden, got {hidden.ndim}"
            )
        hidden = hidden[:, -1:]
        self.model.flow.append_hidden(data.flow_state, hidden)
        self._run_flow_step(result, data, hidden)

    def _run_flow_step(self, result: Any, data: Any, hidden: torch.Tensor) -> None:
        step = self.model.flow.decode_next(
            data.flow_state,
            hidden_states=hidden,
            num_steps=data.state.num_steps,
            ode_method=data.state.ode_method,
            guidance_scale=data.state.guidance_scale,
            eos_threshold=data.state.eos_threshold,
        )
        data.pending_feedback_queue.append(step.feedback_embedding.detach())
        if step.emit:
            latent = step.latent_patch.detach()
            data.latest_latent_patch = latent
            data.latent_patches.append(latent)
        result.next_token_ids = torch.tensor(
            [data.control_token_id],
            dtype=torch.long,
            device=hidden.device,
        )
        if step.finished:
            data.req.finished_reason = FINISH_MATCHED_TOKEN(data.control_token_id)

    @staticmethod
    def _hidden_states(result: Any) -> torch.Tensor:
        logits_output = getattr(result, "logits_output", None)
        hidden = getattr(logits_output, "hidden_states", None)
        if hidden is None:
            hidden = getattr(result, "hidden_states", None)
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("dots.tts SGLang forward did not return hidden states")
        return hidden

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        del request_id
        req_data.pending_feedback_queue.clear()
        req_data.flow_state = None


__all__ = ["DotsTTSModelRunner"]
