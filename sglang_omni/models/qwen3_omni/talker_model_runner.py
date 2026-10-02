# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker runner with FIFO text/feedback decode handoff."""

from __future__ import annotations

from collections import deque
from collections.abc import Set
from queue import Queue
from typing import TYPE_CHECKING, TypeAlias

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.scheduler import TEST_RETRACT

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.qwen3_omni.components.talker import sampled_length
from sglang_omni.platforms import current_platform
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue
from sglang_omni.scheduling.types import (
    ModelRunnerOutput,
    SchedulerOutput,
    SchedulerRequest,
)

if TYPE_CHECKING:

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
else:
    pass


TalkerInputQueue: TypeAlias = (
    PendingTextTensorQueue | deque[torch.Tensor] | list[torch.Tensor] | None
)


class QwenTalkerModelRunner(ModelRunner["SGLangARRequestData"]):
    model: Qwen3OmniTalker

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        outbox: Queue[OutgoingMessage],
        *,
        code2wav_target: str = "code2wav",
        code2wav_in_process: bool = False,
        feedback_enabled: bool = True,
        codec_coalesce_frames: int = 0,
        codec_coalesce_first_frames: int = 0,
        codec_coalesce_early_frames: int = 0,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.outbox = outbox
        self.code2wav_target = code2wav_target
        self.code2wav_in_process = code2wav_in_process
        self.feedback_enabled = bool(feedback_enabled)
        self.codec_coalesce_frames = max(int(codec_coalesce_frames), 0)
        self.codec_coalesce_first_frames = max(int(codec_coalesce_first_frames), 0)
        self.codec_coalesce_early_frames = max(int(codec_coalesce_early_frames), 0)
        # Launched, unresolved decode steps per request id (one-step lookahead).
        self.inflight_steps: dict[str, int] = {}

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        return super().execute(scheduler_output)

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        composed = self.compose_prefill_embeds(forward_batch, requests)
        if composed is None:
            return
        else:
            pass
        input_embeds, input_embeds_are_projected = composed
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=input_embeds,
                input_embeds_are_projected=input_embeds_are_projected,
            ),
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del is_lookahead
        del forward_batch
        del schedule_batch
        if not self.feedback_enabled:
            return
        else:
            pass

        if not self.requests_ready_for_decode(requests):
            raise RuntimeError(
                "Talker decode reached model runner without ready feedback/text input"
            )
        else:
            pass

        self.model.prepare_decode_buffers(requests, self.inflight_steps)
        self.write_feedback_buffers(requests)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        # Note (Xuesong): Do not clear data.prefill_input_embeds: decode retract may requeue
        # the Req for another prefill pass and Req.input_embeds is None.
        if not self.feedback_enabled:
            return
        else:
            pass

        if result.next_token_ids is None:
            return
        else:
            pass
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        talker_hidden = result.logits_output.hidden_states
        if isinstance(talker_hidden, torch.Tensor) and talker_hidden.ndim == 2:
            talker_hidden = talker_hidden.unsqueeze(1)
        else:
            pass
        self.model.code_predictor_forward(layer0_codes, talker_hidden)
        self.stage_token_ids(result, result.next_token_ids)
        self.emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        if not self.feedback_enabled:
            return
        else:
            pass

        batch_size = len(requests)
        result.next_token_ids = self.model.sampled_token_ids[:batch_size].clone()
        self.stage_token_ids(result, result.next_token_ids)
        self.emit_code_chunks_and_feedback(
            schedule_batch=schedule_batch,
            requests=requests,
        )

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        """A lookahead launch reuses the last prepare's buffers.

        The rebuild reads the repetition history from output_ids, which lack the
        tokens of launched, unresolved steps, so only a step on the same rows, each
        one sampled token further, launches ahead. SGLang's forced test retraction
        can drop an in-flight row that the talker has already fed forward, so it
        runs synchronously.
        """
        if (
            not self.feedback_enabled
            or TEST_RETRACT
            or not super().lookahead_eligible(batch)
        ):
            return False
        else:
            pass
        return self.model.can_reuse_decode_buffers(
            [request.rid for request in batch.reqs],
            [sampled_length(request, self.inflight_steps) for request in batch.reqs],
        )

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        """Publish the sampled codes and queue each row's feedback for the next
        launch; the code rows are emitted at resolve, once the rows still alive
        are known."""
        batch_size = len(requests)
        result.next_token_ids = self.model.sampled_token_ids[:batch_size].clone()
        self.stage_token_ids(result, result.next_token_ids)
        codes_snap, embeds_snap = self.snapshot_step_outputs(batch_size)
        self.queue_feedback_rows(requests, embeds_snap)
        for sched_req in requests:
            request_id = sched_req.data.req.rid
            self.inflight_steps[request_id] = self.inflight_steps.get(request_id, 0) + 1
        return codes_snap

    def post_decode_resolve(
        self,
        launch_buf: torch.Tensor,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        for sched_req in requests:
            request_id = sched_req.data.req.rid
            remaining = self.inflight_steps[request_id] - 1
            if remaining:
                self.inflight_steps[request_id] = remaining
            else:
                self.inflight_steps.pop(request_id)
        # note (ratish): a row that finished in the previous step ran one step past
        # its end; its codes are never sent.
        overrun_request_ids = {
            request.rid
            for request in schedule_batch.reqs
            if request.finished() or self.req_is_retracted(request)
        }
        self.emit_code_chunks(
            schedule_batch=schedule_batch,
            requests=requests,
            codes_snap=launch_buf,
            skip_request_ids=overrun_request_ids,
        )

    def emit_code_chunks_and_feedback(
        self,
        *,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        codes_snap, embeds_snap = self.snapshot_step_outputs(len(requests))
        self.queue_feedback_rows(requests, embeds_snap)
        self.emit_code_chunks(
            schedule_batch=schedule_batch, requests=requests, codes_snap=codes_snap
        )

    def snapshot_step_outputs(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Note (wenyao): one batched clone per buffer, not one per row: the
        # snapshot must be a fresh allocation so its rows survive the next
        # in-graph write to the fixed-address _output_codes/_output_embeds.
        codes_snap = self.model.output_codes[:batch_size].detach().clone()
        embeds_snap = self.model.output_embeds[:batch_size].detach().clone()
        return codes_snap, embeds_snap

    @staticmethod
    def queue_feedback_rows(
        requests: list[SchedulerRequest], embeds_snap: torch.Tensor
    ) -> None:
        for idx, sched_req in enumerate(requests):
            sched_req.data.pending_feedback_queue.append(embeds_snap[idx])

    def emit_code_chunks(
        self,
        *,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        codes_snap: torch.Tensor,
        skip_request_ids: Set[str] = frozenset(),
    ) -> None:
        coalesce = self.codec_coalesce_frames
        code_messages: list[OutgoingMessage] = []
        for idx, sched_req in enumerate(requests):
            req = schedule_batch.reqs[idx]
            if req.rid in skip_request_ids:
                continue
            else:
                pass
            code_chunk = codes_snap[idx]
            if coalesce > 1:
                data = sched_req.data
                pending = data.pending_codec_rows
                data.codec_frames_seen += 1
                if data.codec_frames_seen <= self.codec_coalesce_early_frames:
                    code_messages.append(
                        OutgoingMessage(
                            request_id=req.rid,
                            type="stream",
                            data=code_chunk,
                            target=self.code2wav_target,
                            metadata={"stream": self.is_streaming(data)},
                        )
                    )
                else:
                    # Note (wenyao): Only the newest row can be EOS; holding it for the finish
                    # hook keeps threshold flushes EOS-free without a sender-side sync.
                    if self.codec_coalesce_early_frames > 0:
                        # Note (wenyao): Align full batches to the emitted early prefix.
                        flush_ready = len(pending) >= coalesce
                    else:
                        flush_ready = len(pending) >= self.coalesce_threshold(data)
                    if flush_ready:
                        self.flush_codec_rows(req.rid, data, code_messages)
                    else:
                        pass
                    pending.append(code_chunk)
            else:
                code_messages.append(
                    OutgoingMessage(
                        request_id=req.rid,
                        type="stream",
                        data=code_chunk,
                        target=self.code2wav_target,
                        metadata={"stream": self.is_streaming(sched_req.data)},
                    )
                )
        self.put_code_messages(code_messages)

    def put_code_messages(self, code_messages: list[OutgoingMessage]) -> None:
        """Send the messages with one ready event recorded after all their codes.

        The event is recorded once every snapshot and stack the messages carry
        is enqueued, and before any message is visible to the consumer. Only a
        code2wav in this process can wait on it; another process orders its
        reads on the receiving stream instead.
        """
        if (
            self.code2wav_in_process
            and code_messages
            and code_messages[0].data.device.type == "cuda"
        ):
            device = code_messages[0].data.device
            device_module = torch.get_device_module(device)
            codes_ready_event = device_module.Event()
            codes_ready_event.record(device_module.current_stream(device))
            for message in code_messages:
                message.metadata["codes_ready_event"] = codes_ready_event
        else:
            pass
        for message in code_messages:
            self.outbox.put(message)

    @staticmethod
    def is_streaming(data: SGLangARRequestData) -> bool:
        stage_payload = data.stage_payload
        return bool(
            stage_payload is not None
            and (stage_payload.request.params or {}).get("stream", False)
        )

    def coalesce_threshold(self, data: SGLangARRequestData) -> int:
        first = self.codec_coalesce_first_frames
        if first > 0 and not data.codec_first_flush_done:
            return first
        else:
            pass
        return self.codec_coalesce_frames

    def flush_codec_rows(
        self,
        request_id: str,
        data: SGLangARRequestData,
        code_messages: list[OutgoingMessage],
    ) -> None:
        pending = data.pending_codec_rows
        if not pending:
            return
        else:
            pass
        data.codec_first_flush_done = True
        rows = pending[0] if len(pending) == 1 else torch.stack(pending, dim=0)
        pending.clear()
        code_messages.append(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=rows,
                target=self.code2wav_target,
                metadata={"stream": self.is_streaming(data)},
            )
        )

    def on_request_finished(
        self, request_id: str, req_data: "SGLangARRequestData"
    ) -> None:
        pending = req_data.pending_codec_rows
        if not pending:
            return
        else:
            pass
        # Only preceding rows are known to be non-EOS. Send the uncertain last
        # row through Code2Wav's 1-D EOS scan without synchronizing on the sender.
        code_messages: list[OutgoingMessage] = []
        last_row = pending.pop()
        self.flush_codec_rows(request_id, req_data, code_messages)
        pending.append(last_row)
        self.flush_codec_rows(request_id, req_data, code_messages)
        self.put_code_messages(code_messages)

    def sample_before_post_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return True

    def sample_before_post_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> bool:
        del forward_batch, schedule_batch, requests
        return False

    def process_sampling_logits(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        # note (ratish): a retract re-prefill resumes with output history, which the
        # talker penalizes here in float32 as SGLang's repetition penalizer would.
        logits = logits_output.next_token_logits
        vocab_size = logits.shape[1]
        history_rows: list[int] = []
        history_token_ids: list[int] = []
        history_penalties: list[float] = []
        for row_idx, sched_req in enumerate(requests):
            request_data = sched_req.data
            if request_data.repetition_penalty != 1.0:
                token_ids = {
                    token_id
                    for token_id in request_data.req.output_ids
                    if 0 <= token_id < vocab_size
                }
                history_rows.extend([row_idx] * len(token_ids))
                history_token_ids.extend(token_ids)
                history_penalties.extend(
                    [request_data.repetition_penalty] * len(token_ids)
                )
            else:
                pass
        if not history_rows:
            return
        else:
            pass
        device = logits.device
        pin_memory = current_platform.is_pin_memory_available(device)
        history_index = torch.tensor(
            [history_rows, history_token_ids], dtype=torch.int64, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        penalties = torch.tensor(
            history_penalties, dtype=torch.float32, pin_memory=pin_memory
        ).to(device, non_blocking=True)
        history_logits = logits[history_index[0], history_index[1]]
        logits.index_put_(
            (history_index[0], history_index[1]),
            torch.where(
                history_logits > 0,
                history_logits / penalties,
                history_logits * penalties,
            ).to(logits.dtype),
        )

    def is_decode_batch_ready(self, schedule_batch: ScheduleBatch) -> bool:
        if not self.feedback_enabled or not schedule_batch.forward_mode.is_decode():
            return True
        else:
            pass
        return all(
            self.data_has_next_decode_input(
                getattr(req, "omni_data", None)
            )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            for req in schedule_batch.reqs
        )

    def compose_prefill_embeds(
        self,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> tuple[torch.Tensor, bool] | None:
        """Assemble prefill rows and preserve whether they are projected."""
        projected_flags = [
            bool(req.data.input_embeds_are_projected) for req in requests
        ]
        has_tensor_requests = any(
            req.data.prefill_input_embeds is not None for req in requests
        )
        if not any(projected_flags) and not has_tensor_requests:
            return None
        else:
            pass

        has_projected_requests = any(projected_flags)
        if has_projected_requests and not all(projected_flags):
            raise RuntimeError(
                "Talker projected and unprojected prefill requests cannot be "
                "batched together"
            )
        else:
            pass

        parts: list[torch.Tensor] = []
        for sched_req in requests:
            req = sched_req.data.req
            prefix_len = len(req.prefix_indices)
            extend_len = int(req.extend_range.length)
            part = self.projected_prefill_slice(
                sched_req=sched_req,
                prefix_len=prefix_len,
                extend_len=extend_len,
                device=forward_batch.input_ids.device,
            )
            if part is not None and part.shape[0] > 0:
                parts.append(part)
            else:
                pass
        if not parts:
            return None
        else:
            pass
        input_embeds = torch.cat(parts, dim=0)

        expected_rows = int(forward_batch.input_ids.shape[0])
        if input_embeds.shape[0] != expected_rows:
            raise RuntimeError(
                "Talker prefill embeds must align with forward input_ids: "
                f"got {input_embeds.shape[0]} rows for {expected_rows} input ids"
            )
        else:
            pass
        return (
            input_embeds.to(
                device=forward_batch.input_ids.device,
                dtype=self.model.activation_dtype,
            ),
            has_projected_requests,
        )

    @staticmethod
    def projected_prefill_slice(
        *,
        sched_req: SchedulerRequest,
        prefix_len: int,
        extend_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if extend_len <= 0:
            return None
        else:
            pass

        data = sched_req.data
        req = data.req
        end = prefix_len + extend_len
        tensor = data.prefill_input_embeds
        if tensor is not None:
            prompt_len = int(tensor.shape[0])
            dtype = tensor.dtype
            embed_device = tensor.device
            parts = QwenTalkerModelRunner.prefill_prompt_parts_from_tensor(
                tensor=tensor,
                prefix_len=prefix_len,
                end=end,
            )
        else:
            embeds = req.input_embeds
            if not embeds:
                return None
            else:
                pass
            prompt_len = len(embeds)
            dtype = torch.float32
            embed_device = device
            parts = QwenTalkerModelRunner.prefill_prompt_parts_from_list(
                embeds=embeds,
                prefix_len=prefix_len,
                end=end,
                device=device,
            )

        if end > prompt_len:
            generated = QwenTalkerModelRunner.generated_prefill_slice(
                sched_req=sched_req,
                gen_start=max(prefix_len, prompt_len) - prompt_len,
                gen_end=end - prompt_len,
                device=embed_device,
                dtype=dtype,
            )
            if generated is not None:
                parts.append(generated)
            else:
                pass
        else:
            pass

        if not parts:
            return None
        else:
            pass
        return torch.cat(parts, dim=0)

    @staticmethod
    def prefill_prompt_parts_from_tensor(
        *,
        tensor: torch.Tensor,
        prefix_len: int,
        end: int,
    ) -> list[torch.Tensor]:
        prompt_len = int(tensor.shape[0])
        start = min(prefix_len, prompt_len)
        stop = min(end, prompt_len)
        return [tensor[start:stop]] if stop > start else []

    @staticmethod
    def prefill_prompt_parts_from_list(
        *,
        embeds: list,
        prefix_len: int,
        end: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        prompt_len = len(embeds)
        start = min(prefix_len, prompt_len)
        stop = min(end, prompt_len)
        if stop <= start:
            return []
        else:
            pass
        return [
            torch.as_tensor(
                embeds[start:stop],
                device=device,
                dtype=torch.float32,
            )
        ]

    @staticmethod
    def generated_prefill_slice(
        *,
        sched_req: SchedulerRequest,
        gen_start: int,
        gen_end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if gen_end <= gen_start:
            return None
        else:
            pass

        data = sched_req.data
        history = QwenTalkerModelRunner.decode_input_history(data)
        while len(history) < gen_end:
            combined = QwenTalkerModelRunner.take_next_decode_input_embed(
                sched_req=sched_req,
                device=device,
                dtype=dtype,
            )
            if combined is None:
                raise RuntimeError(
                    "Cannot replay retracted talker decode tokens: missing "
                    "feedback/text input embeds for generated-token prefill"
                )
            else:
                pass
            QwenTalkerModelRunner.append_decode_input_history(data, combined)

        rows = [
            QwenTalkerModelRunner.decode_row(row, device=device, dtype=dtype)
            for row in history[gen_start:gen_end]
        ]
        if not rows:
            return None
        else:
            pass
        return torch.stack(rows, dim=0)

    def write_feedback_buffers(self, requests: list[SchedulerRequest]) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        else:
            pass

        feedback_buffer = self.model.feedback_buffer
        feedback_mask = self.model.feedback_mask
        feedback_mask[:batch_size] = False

        rows: list[int] = []
        embeds: list[torch.Tensor] = []
        for row_idx, sched_req in enumerate(requests):
            combined = self.take_next_decode_input_embed(
                sched_req=sched_req,
                device=feedback_buffer.device,
                dtype=feedback_buffer.dtype,
            )
            if combined is None:
                continue
            else:
                pass
            self.append_decode_input_history(sched_req.data, combined)
            rows.append(row_idx)
            embeds.append(combined)
        if not rows:
            return
        else:
            pass
        embeds_stacked = torch.stack(embeds, dim=0)
        if len(rows) == batch_size:
            # Note (wenyao): dense steady state: rows is exactly range(batch_size),
            # so slice-assign and skip the per-frame pageable index H2D
            feedback_buffer[:batch_size] = embeds_stacked
            feedback_mask[:batch_size] = True
            return
        else:
            pass
        rows_t = torch.tensor(rows, dtype=torch.long, device=feedback_buffer.device)
        feedback_buffer[rows_t] = embeds_stacked
        feedback_mask[rows_t] = True

    @staticmethod
    def data_has_next_decode_input(data: SGLangARRequestData | None) -> bool:
        if data is None:
            return False
        else:
            pass
        if not data.pending_feedback_queue:
            return False
        else:
            pass
        if data.pending_text_queue:
            return True
        else:
            pass
        return bool(data.thinker_chunks_done and data.tts_pad_embed is not None)

    def requests_ready_for_decode(self, requests: list[SchedulerRequest]) -> bool:
        return all(
            self.data_has_next_decode_input(sched_req.data) for sched_req in requests
        )

    @staticmethod
    def pop_left(queue: TalkerInputQueue) -> torch.Tensor | None:
        if not queue:
            return None
        else:
            pass
        if hasattr(queue, "popleft"):
            return queue.popleft()
        else:
            pass
        if isinstance(queue, list):
            return queue.pop(0)
        else:
            pass
        return None

    @staticmethod
    def peek_left(queue: TalkerInputQueue) -> torch.Tensor | None:
        if not queue:
            return None
        else:
            pass
        if isinstance(queue, list):
            return queue[0]
        else:
            pass
        if hasattr(queue, "__getitem__"):
            return queue[0]
        else:
            pass
        return None

    @staticmethod
    def decode_input_history(data: SGLangARRequestData) -> list[torch.Tensor]:
        history = data.decode_input_embeds
        if history is None:
            history = []
            data.decode_input_embeds = history
        else:
            pass
        return history

    @staticmethod
    def append_decode_input_history(
        data: SGLangARRequestData, row: torch.Tensor
    ) -> None:
        QwenTalkerModelRunner.decode_input_history(data).append(row.detach())

    @staticmethod
    def decode_row(
        row: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        row = row.reshape(-1)
        if row.device != device or row.dtype != dtype:
            raise RuntimeError(
                "Talker decode rows must already match the feedback buffer "
                f"device/dtype, got {row.device}/{row.dtype}, "
                f"expected {device}/{dtype}"
            )
        else:
            pass
        return row

    @staticmethod
    def peek_next_decode_inputs(
        data: SGLangARRequestData,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """The feedback row and the text row of the next decode input. None
        while the feedback row is missing, or while the text row is missing
        and the text stream is still open. After the stream has closed, the
        pad embedding stands in for the text row."""
        feedback = QwenTalkerModelRunner.peek_left(data.pending_feedback_queue)
        if feedback is None:
            return None
        else:
            pass
        next_text = QwenTalkerModelRunner.peek_left(data.pending_text_queue)
        if next_text is None:
            if not data.thinker_chunks_done:
                return None
            else:
                pass
            next_text = data.tts_pad_embed
        else:
            pass
        return feedback, next_text

    @staticmethod
    def pop_next_decode_inputs(data: SGLangARRequestData) -> None:
        QwenTalkerModelRunner.pop_left(data.pending_feedback_queue)
        QwenTalkerModelRunner.pop_left(data.pending_text_queue)

    @staticmethod
    def combine_feedback_with_next_text(
        *,
        data: SGLangARRequestData,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        inputs = QwenTalkerModelRunner.peek_next_decode_inputs(data)
        if inputs is None:
            return None
        else:
            pass
        feedback, next_text = inputs
        return QwenTalkerModelRunner.decode_row(
            feedback,
            device=device,
            dtype=dtype,
        ) + QwenTalkerModelRunner.decode_row(
            next_text,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def take_next_decode_input_embed(
        *,
        sched_req: SchedulerRequest,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        data = sched_req.data
        combined = QwenTalkerModelRunner.combine_feedback_with_next_text(
            data=data,
            device=device,
            dtype=dtype,
        )
        if combined is None:
            return None
        else:
            pass
        QwenTalkerModelRunner.pop_next_decode_inputs(data)
        return combined
