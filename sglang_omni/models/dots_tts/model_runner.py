# SPDX-License-Identifier: Apache-2.0
"""Model runner hooks for native dots TTS latent generation."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_execution import attn_forward_context
from sglang_omni.models.dots_tts.sglang_model import DotsTTSLatentBatch


class DotsTTSModelRunner(ModelRunner):
    """SGLang model runner scaffold for continuous latent generation."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    def before_prefill(self, forward_batch: Any, schedule_batch: Any, requests: list):
        del forward_batch, schedule_batch, requests

    def before_decode(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
        *,
        is_lookahead: bool = False,
    ):
        del schedule_batch, is_lookahead
        reference = None
        for sched_req in requests:
            queue = sched_req.data.decode_input_embeds
            if queue:
                reference = queue[0]
                break
        if reference is None:
            return
        if reference.ndim == 3:
            reference = reference[:, -1, :]
        reference = reference.reshape(-1)
        pieces = []
        for sched_req in requests:
            queue = sched_req.data.decode_input_embeds
            if queue:
                embed = queue.pop(0)
                if embed.ndim == 3:
                    embed = embed[:, -1, :]
                pieces.append(embed.reshape(-1))
            else:
                pieces.append(torch.zeros_like(reference))
        forward_batch.input_embeds = torch.stack(pieces, dim=0).to(
            device=forward_batch.input_ids.device
        )

    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        if not any(
            sched_req.data.prefill_input_embeds is not None for sched_req in requests
        ):
            return None
        input_embeds = self._build_prefill_input_embeds(forward_batch, requests)
        return self._forward_with_input_embeds(forward_batch, input_embeds)

    def custom_decode_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list,
    ) -> GenerationBatchResult | None:
        del schedule_batch
        input_embeds = getattr(forward_batch, "input_embeds", None)
        if input_embeds is None:
            return None
        model_runner = self.tp_worker.model_runner
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        with attn_forward_context(model_runner.attn_backend):
            latent_output = self.model.forward_latent_decode_step(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                requests=requests,
                input_embeds=input_embeds.to(
                    device=forward_batch.input_ids.device,
                    dtype=next(self.model.parameters()).dtype,
                ),
            )
        return latent_output.batch_result

    def _forward_with_input_embeds(
        self,
        forward_batch: Any,
        input_embeds: torch.Tensor,
    ) -> GenerationBatchResult:
        model_runner = self.tp_worker.model_runner
        model_dtype = next(self.model.parameters()).dtype
        model_runner.attn_backend.init_forward_metadata(forward_batch)
        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions
        with attn_forward_context(model_runner.attn_backend):
            logits_output = self.model(
                input_ids=forward_batch.input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds.to(
                    device=forward_batch.input_ids.device,
                    dtype=model_dtype,
                ),
            )
        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )

    def _build_prefill_input_embeds(
        self,
        forward_batch: Any,
        requests: list,
    ) -> torch.Tensor:
        pieces = []
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            embeds = data.prefill_input_embeds
            if embeds is None:
                raise RuntimeError(
                    "dots TTS mixed prefill batches with and without prompt "
                    "input embeddings are not supported in v1"
                )
            if embeds.ndim == 3:
                embeds = embeds.squeeze(0)
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
            pieces.append(embeds[prefix_len : prefix_len + req_len])
        return torch.cat(pieces, dim=0).to(device=forward_batch.input_ids.device)

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        self._capture_hidden_states(result, requests, packed_prefill=True)
        self._run_model_latent_batch(result, requests)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        latent_output = getattr(result, "dots_tts_latent_output", None)
        if latent_output is not None:
            self._apply_latent_step_output(latent_output, result, requests)
            return
        self._capture_hidden_states(result, requests, packed_prefill=False)
        self._run_model_latent_batch(result, requests)

    def _capture_hidden_states(
        self,
        result: Any,
        requests: list,
        *,
        packed_prefill: bool,
    ) -> None:
        hidden_states = getattr(result, "hidden_states", None)
        if hidden_states is None:
            logits_output = getattr(result, "logits_output", None)
            hidden_states = getattr(logits_output, "hidden_states", None)
        if hidden_states is None:
            return
        if hidden_states.ndim == 3:
            for index, sched_req in enumerate(requests):
                sched_req.data.latest_hidden_state = hidden_states[
                    index : index + 1, -1:, :
                ]
            return
        if hidden_states.ndim != 2:
            raise RuntimeError(
                f"dots TTS expected 2D or 3D hidden states, got {hidden_states.ndim}D"
            )
        if not packed_prefill or hidden_states.size(0) == len(requests):
            for index, sched_req in enumerate(requests):
                sched_req.data.latest_hidden_state = hidden_states[
                    index : index + 1
                ].unsqueeze(0)
            return
        offset = 0
        for sched_req in requests:
            data = sched_req.data
            req = data.req
            req_len = int(req.extend_range.length)
            if req_len <= 0:
                raise RuntimeError(
                    "dots TTS hidden capture requires extend range length > 0"
                )
            offset += req_len
            data.latest_hidden_state = hidden_states[offset - 1 : offset].unsqueeze(0)

    def _run_model_latent_batch(self, result: Any, requests: list) -> None:
        active_indices = [
            index
            for index, sched_req in enumerate(requests)
            if self._can_run_latent_step(sched_req.data)
        ]
        latent_output = self.model.decode_audio_batch(
            DotsTTSLatentBatch(
                requests=requests,
                active_indices=active_indices,
                hidden_states=[
                    sched_req.data.latest_hidden_state for sched_req in requests
                ],
            )
        )
        self._apply_latent_step_output(latent_output, result, requests)

    def _apply_latent_step_output(
        self,
        latent_output: Any,
        result: Any,
        requests: list,
    ) -> None:
        for index, sched_req in enumerate(requests):
            data = sched_req.data
            hidden_state = latent_output.hidden_states[index]
            latent_patch = latent_output.latent_patches[index]
            feedback_embedding = latent_output.feedback_embeddings[index]
            eos_score = latent_output.eos_scores[index]
            if hidden_state is not None:
                data.latest_hidden_state = hidden_state
            if latent_patch is not None:
                data.latest_latent_patch = latent_patch
                data.latent_patches.append(latent_patch.detach())
                data.position += 1
            if feedback_embedding is not None:
                data.decode_input_embeds.append(feedback_embedding)
            data.eos_score = eos_score
            if latent_output.finished[index]:
                self._mark_finished(data)
        result.next_token_ids = latent_output.next_token_ids

    @staticmethod
    def _mark_finished(data: Any) -> None:
        data.finish_reason = "stop"
        req = data.req
        if req is not None and req.finished_reason is None:
            req.finished_reason = FINISH_MATCHED_TOKEN(data.control_token_id)

    @staticmethod
    def _can_run_latent_step(data: Any) -> bool:
        return (
            data.finish_reason is None
            and data.fm_state is not None
            and data.latest_hidden_state is not None
        )


__all__ = ["DotsTTSModelRunner"]
