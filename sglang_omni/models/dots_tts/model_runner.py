# SPDX-License-Identifier: Apache-2.0
"""Model runner hooks for native dots TTS latent generation."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner


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
            queue = getattr(sched_req.data, "decode_input_embeds", None)
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
            queue = getattr(sched_req.data, "decode_input_embeds", None)
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
            getattr(sched_req.data, "prefill_input_embeds", None) is not None
            for sched_req in requests
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
        del schedule_batch, requests
        input_embeds = getattr(forward_batch, "input_embeds", None)
        if input_embeds is None:
            return None
        return self._forward_with_input_embeds(forward_batch, input_embeds)

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
            req_len = int(req.extend_input_len)
            prefix_len = len(req.prefix_indices)
            pieces.append(embeds[prefix_len : prefix_len + req_len])
        return torch.cat(pieces, dim=0).to(device=forward_batch.input_ids.device)

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        self._capture_hidden_states(result, requests, packed_prefill=True)
        self._run_audio_step(result, requests)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        self._capture_hidden_states(result, requests, packed_prefill=False)
        self._run_audio_step(result, requests)

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
            req_len = int(req.extend_input_len)
            if req_len <= 0:
                raise RuntimeError(
                    "dots TTS hidden capture requires extend_input_len > 0"
                )
            offset += req_len
            data.latest_hidden_state = hidden_states[
                offset - 1 : offset
            ].unsqueeze(0)

    def _run_audio_step(self, result: Any, requests: list) -> None:
        control_ids: list[int] = []
        for sched_req in requests:
            data = sched_req.data
            if getattr(data, "finish_reason", None) is not None:
                control_ids.append(int(getattr(data, "control_token_id", 0)))
                continue
            hidden_state = getattr(data, "latest_hidden_state", None)
            fm_state = getattr(data, "fm_state", None)
            model = getattr(self, "model", None)
            step_audio_latent = getattr(model, "step_audio_latent", None)
            if (
                fm_state is not None
                and hidden_state is not None
                and step_audio_latent is not None
            ):
                audio_step = step_audio_latent(data, hidden_state)
                data.latest_latent_patch = audio_step.latent_patch
                data.latent_patches.append(audio_step.latent_patch.detach())
                data.decode_input_embeds.append(audio_step.feedback_embedding)
                data.eos_score = audio_step.eos_score
                data.position = int(getattr(data, "position", 0)) + 1
                if self._should_finish(data):
                    self._mark_finished(data)
            control_ids.append(int(getattr(data, "control_token_id", 0)))

        result.next_token_ids = torch.tensor(
            control_ids,
            dtype=torch.long,
            device=self.device,
        )

    @staticmethod
    def _mark_finished(data: Any) -> None:
        data.finish_reason = "stop"
        req = getattr(data, "req", None)
        if req is not None and getattr(req, "finished_reason", None) is None:
            req.finished_reason = FINISH_MATCHED_TOKEN(
                int(getattr(data, "control_token_id", 0))
            )

    @staticmethod
    def _should_finish(data: Any) -> bool:
        eos_score = getattr(data, "eos_score", None)
        if eos_score is not None and bool((eos_score > 0.5).any()):
            return True
        max_generate_length = int(getattr(data, "max_generate_length", 500))
        return len(getattr(data, "latent_patches", [])) >= max_generate_length


__all__ = ["DotsTTSModelRunner"]
