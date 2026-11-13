# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from collections.abc import Collection
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict

import torch
from sglang.srt.utils.common import async_h2d

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.qwen3_tts.code_table import CodecFrameStep, Qwen3TTSCodeTable
from sglang_omni.models.qwen3_tts.request_builders import (
    Qwen3TTSSGLangRequestData,
    first_stream_metadata,
)
from sglang_omni.scheduling.message import OutgoingMessage, StreamStep
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from sglang_omni.utils.device import has_device_streams

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from sglang_omni.models.qwen3_tts.sglang_model import (
        PredictorGraphStats,
        Qwen3TTSTalker,
    )
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
else:
    pass

# note (ratish): official qwen-tts generates the talker with min_new_tokens=2; an EOS on the first frame ended a request with no audio.
QWEN3_TTS_MIN_NEW_FRAMES = 2


class Qwen3TTSModelInfo(TypedDict):
    predictor_cuda_graph: PredictorGraphStats


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    model: Qwen3TTSTalker
    tp_worker: ModelWorker
    # Resolved in __init__: post_process_outputs records a readiness event per
    # decode step, and the device cannot change under the runner.
    codes_event_module: ModuleType | None = None

    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        leading_silence_mask_frames: int,
        silence_codec_ids: torch.Tensor,
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.has_pending_code_step = False
        self.row_ids_cache: torch.Tensor | None = None
        self.leading_silence_mask_frames = leading_silence_mask_frames
        self.silence_codec_ids = silence_codec_ids
        self.codes_event_module = (
            torch.get_device_module(self.device)
            if has_device_streams(self.device)
            else None
        )
        request_slot_count, context_length = (
            tp_worker.model_runner.req_to_token_pool.req_to_token.shape
        )
        # note (ratish): the predictor's feedback row is the next decode's input; kept
        # per request slot on the device, a step reads its batch in one gather.
        self.slot_feedback_rows = torch.zeros(
            (request_slot_count, self.model.output_embeds.shape[-1]),
            dtype=self.model.output_embeds.dtype,
            device=self.device,
        )
        # note (ratish): a request's frames stay under its positions, which the context
        # length bounds. The engine's requests hold at most one row a slot, and as many
        # streams the vocoder is still finishing; past that a new request waits.
        self.code_table = Qwen3TTSCodeTable(
            row_count=2 * request_slot_count,
            row_frames=context_length,
            codebook_count=self.model.config.num_code_groups,
            device=self.device,
        )
        self.frame_index_bases: torch.Tensor | None = None
        self.code_step = 0
        self.frames_written_events: list[torch.Event] = []
        self.step_frames_written: torch.Event | None = None
        self.stream_step_target: str | None = None
        self.step_request_ids: list[str] = []
        self.step_rows: list[int] = []
        self.step_frame_counts: list[int] = []
        self.step_opened_streams: dict[str, dict[str, object]] = {}

    def model_info(self) -> Qwen3TTSModelInfo:
        return {"predictor_cuda_graph": self.model.predictor_graph_stats()}

    def admission_capacity(self) -> int:
        return self.code_table.free_row_count()

    def enable_stream_steps(self, target: str) -> bool:
        self.stream_step_target = target
        return True

    def take_stream_step(
        self,
        aborted_request_ids: Collection[str],
        skipped_request_ids: Collection[str],
    ) -> OutgoingMessage | None:
        request_ids = self.step_request_ids
        rows = self.step_rows
        frame_counts = self.step_frame_counts
        opened = self.step_opened_streams
        self.step_request_ids = []
        self.step_rows = []
        self.step_frame_counts = []
        self.step_opened_streams = {}
        kept = [
            index
            for index, request_id in enumerate(request_ids)
            if request_id not in aborted_request_ids
            and request_id not in skipped_request_ids
        ]
        if not kept:
            return None
        elif len(kept) < len(request_ids):
            request_ids = [request_ids[index] for index in kept]
            rows = [rows[index] for index in kept]
            frame_counts = [frame_counts[index] for index in kept]
            opened = {
                request_id: metadata
                for request_id, metadata in opened.items()
                if request_id not in aborted_request_ids
                and request_id not in skipped_request_ids
            }
        else:
            pass
        # note (ratish): the vocoder holds an opened stream's row from here until the stream ends.
        for request_id, row in zip(request_ids, rows, strict=True):
            if request_id in opened:
                self.code_table.hold(row)
            else:
                pass
        return OutgoingMessage(
            request_id="",
            type="stream_step",
            target=self.stream_step_target,
            data=StreamStep(
                request_ids=request_ids,
                opened=opened,
                data=CodecFrameStep(
                    table=self.code_table,
                    rows=rows,
                    frame_counts=frame_counts,
                    written=self.step_frames_written,
                    step=self.code_step,
                ),
            ),
        )

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        for sched_req in requests:
            if sched_req.data.code_row is None:
                self.open_code_row(sched_req.data)
            else:
                pass
        self.stage_frame_index_bases(requests)
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=self.build_prefill_input_embeds(
                    forward_batch,
                    requests,
                ),
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
        del schedule_batch
        self.stage_frame_index_bases(requests)
        self.write_feedback_buffers(forward_batch, requests)

    def open_code_row(self, data: Qwen3TTSSGLangRequestData) -> None:
        """Give a request its table row, the reference frames first; a retracted request
        keeps its row."""
        # note (ratish): frame k of a request lands at ref_code_len + k, and k stays under
        # the context length less the prompt, which holds the reference frames.
        assert data.ref_code_len < len(data.req.origin_input_ids), (
            f"Qwen3-TTS request {data.req.rid} has {data.ref_code_len} reference "
            f"frames in a {len(data.req.origin_input_ids)}-position prompt"
        )
        data.code_table = self.code_table
        data.code_row = self.code_table.acquire()
        if data.ref_code_len:
            self.code_table.codes[data.code_row, : data.ref_code_len].copy_(
                data.ref_code
            )
        else:
            pass

    def stage_frame_index_bases(self, requests: list[SchedulerRequest]) -> None:
        """Stage the batch's per-request decode values, and with them what each request
        adds to its sampled position for its frame's flat index in the table."""
        if self.model.prepare_decode_buffers(requests):
            row_frames = self.code_table.row_frames
            self.frame_index_bases = async_h2d(
                [
                    sched_req.data.code_row * row_frames
                    + sched_req.data.ref_code_len
                    - (len(sched_req.data.req.origin_input_ids) - 1)
                    for sched_req in requests
                ],
                dtype=torch.long,
                device=self.device,
            )
        else:
            pass

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_codes(result, forward_batch, schedule_batch, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_codes(result, forward_batch, schedule_batch, requests)

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
        return True

    def lookahead_eligible(self, batch: ScheduleBatch | None) -> bool:
        # note(ratish): the lookahead's launch and resolve hooks do not run the
        # codec collect, they would feed token embeddings back.
        del batch
        return False

    # ------------------------------------------------------------------
    # Qwen3-TTS logit shaping
    # ------------------------------------------------------------------

    def apply_codec_suppress_tokens(
        self, logits_output: LogitsProcessorOutput, requests: list[SchedulerRequest]
    ) -> None:
        logits = logits_output.next_token_logits
        if logits is None or logits.ndim != 2 or not requests:
            return
        else:
            pass

        # Qwen3-TTS reserves the final 1024 configured token IDs for codec
        # control and suppresses that range except for codec EOS.
        configured_vocab = int(self.model.config.vocab_size)
        suppress_start = max(0, configured_vocab - 1024)
        suppress_stop = min(configured_vocab, logits.shape[1])
        if suppress_start >= suppress_stop:
            return
        else:
            pass

        active_logits = logits[: len(requests)]
        codec_eos = int(self.model.config.codec_eos_token_id)
        if suppress_start <= codec_eos < suppress_stop:
            active_logits[:, suppress_start:codec_eos] = float("-inf")
            active_logits[:, codec_eos + 1 : suppress_stop] = float("-inf")
        else:
            active_logits[:, suppress_start:suppress_stop] = float("-inf")
        for row_index, scheduled_request in enumerate(requests):
            if scheduled_request.data.frame_count < QWEN3_TTS_MIN_NEW_FRAMES:
                # note (ratish): a slice fills on the device; assigning one element copies a host scalar and waits for the stream.
                active_logits[row_index, codec_eos : codec_eos + 1] = float("-inf")
            else:
                pass

        # note (luojiaxuan): a cold-start clone that samples a silence id first
        # tends to stay silent for several frames, so silence ids are excluded
        # from its opening frames.
        if self.leading_silence_mask_frames > 0:
            for row_index, scheduled_request in enumerate(requests):
                if (
                    scheduled_request.data.mask_leading_silence
                    and scheduled_request.data.frame_count
                    < self.leading_silence_mask_frames
                ):
                    active_logits[row_index].index_fill_(
                        0, self.silence_codec_ids, float("-inf")
                    )
                else:
                    pass
        else:
            pass

    def install_sampling_seeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Install the staged semantic seeds when a request in the batch carries a
        seed, or deterministic inference has already seeded every row."""
        sampling_info = forward_batch.sampling_info
        if sampling_info.sampling_seed is None and all(
            scheduled_request.data.req.sampling_params.sampling_seed is None
            for scheduled_request in requests
        ):
            return
        else:
            pass
        self.validate_seeded_sampling_supported(sampling_info)
        sampling_info.sampling_seed = self.model.semantic_sampling_seed_tensor[
            : len(requests)
        ]

    def collect_codes(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.has_pending_code_step = False
        if result.next_token_ids is None:
            return
        else:
            pass
        layer0_codes = result.next_token_ids
        if layer0_codes.ndim == 1:
            layer0_codes = layer0_codes.unsqueeze(1)
        else:
            pass
        # note(ratish): the layer 0 id is the step's only host read, staged ahead
        # of the predictor so the finalize wait covers the sample, not the predictor.
        self.stage_token_ids(result, result.next_token_ids)

        hidden = result.logits_output.hidden_states
        if isinstance(hidden, torch.Tensor) and hidden.ndim == 2:
            hidden = hidden.unsqueeze(1)
        else:
            pass
        semantic_positions = self.sample_positions(forward_batch, layer0_codes.device)
        self.model.code_predictor_forward(
            layer0_codes,
            hidden,
            semantic_positions=semantic_positions,
        )
        self.slot_feedback_rows.index_copy_(
            0,
            forward_batch.req_pool_indices,
            self.model.output_embeds[: len(requests)],
        )
        self.write_step_frames(forward_batch, requests, semantic_positions)
        self.has_pending_code_step = True

    def write_step_frames(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
        semantic_positions: torch.Tensor,
    ) -> None:
        """Write the step's frames into the table in one launch; an EOS row's frame lands
        past its request's counted frames, where nothing reads it."""
        output_codes = self.model.output_codes[: len(requests)]
        frame_indices = semantic_positions + self.frame_index_bases
        if forward_batch.forward_mode.is_decode():
            self.code_table.frames.index_copy_(0, frame_indices, output_codes)
        else:
            # note (ratish): a middle prefill chunk samples at a prompt row, so it has no frame.
            recorded = async_h2d(
                [
                    row_index
                    for row_index, sched_req in enumerate(requests)
                    if sched_req.data.req.inflight_middle_chunks <= 0
                ],
                dtype=torch.long,
                device=self.device,
            )
            self.code_table.frames.index_copy_(
                0, frame_indices[recorded], output_codes[recorded]
            )
        self.code_step += 1
        if self.codes_event_module is not None:
            if not self.frames_written_events:
                # note (ratish): an event's first record waits while another thread is in
                # the device runtime, so the two are made once.
                self.frames_written_events = [
                    self.codes_event_module.Event() for _ in range(2)
                ]
            else:
                pass
            self.step_frames_written = self.frames_written_events[self.code_step % 2]
            self.step_frames_written.record()
        else:
            pass

    def post_process_outputs(
        self,
        result: GenerationBatchResult | None,
        scheduler_output: SchedulerOutput | None,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        if not self.has_pending_code_step:
            return
        else:
            pass
        self.has_pending_code_step = False
        # note (ratish): the step's sample has resolved, and the stream ran the previous
        # step's frame write before it.
        self.code_table.settled_step = self.code_step - 1
        eos_id = int(self.model.config.codec_eos_token_id)
        if self.stream_step_target is None:
            # Note: (Jiaxin Deng) per-row clones were a c32 decode-loop hot spot;
            # rows must stay views of a snapshot, never of the reused graph buffers.
            batch_size = len(scheduler_output.requests)
            codes_snap = self.model.output_codes[:batch_size].detach().clone()
            codes_ready = None
            if self.codes_event_module is not None:
                codes_ready = self.codes_event_module.Event()
                codes_ready.record()
            else:
                pass
        else:
            codes_snap = None
            codes_ready = None
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            data: Qwen3TTSSGLangRequestData = sched_req.data
            req_output = outputs[sched_req.request_id]
            # note (ratish): a middle prefill chunk samples at a prompt row; SGLang drops
            # that token, so it gives no frame or stream chunk either.
            if data.req.inflight_middle_chunks > 0 or req_output.data is None:
                continue
            else:
                pass
            data.speech_output_tokens += 1
            if int(req_output.data) == eos_id:
                continue
            else:
                pass
            data.frame_count += 1
            if codes_snap is not None:
                data.latest_stream_code_chunk = codes_snap[row_idx]
                data.codes_ready_event = codes_ready
            elif data.should_stream_codes:
                self.step_request_ids.append(sched_req.request_id)
                self.step_rows.append(data.code_row)
                self.step_frame_counts.append(data.frame_count)
                if not data.stream_ref_sent:
                    data.stream_ref_sent = True
                    self.step_opened_streams[sched_req.request_id] = (
                        first_stream_metadata(data, self.code_table.codes.shape[-1])
                    )
                else:
                    pass
            else:
                pass

    def finalize_skip_rids(self, scheduler_output: SchedulerOutput) -> set[str]:
        return {
            sched_req.request_id
            for sched_req in scheduler_output.requests
            if sched_req.data.req.inflight_middle_chunks > 0
        }

    def sample_positions(
        self, forward_batch: ForwardBatch, device: torch.device
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_decode():
            return forward_batch.positions.to(device=device, dtype=torch.long)
        else:
            seq_lens = forward_batch.seq_lens.to(device=device, dtype=torch.long)
            return (seq_lens - 1).clamp_min(0)

    def write_feedback_buffers(
        self, forward_batch: ForwardBatch, requests: list[SchedulerRequest]
    ) -> None:
        batch_size = len(requests)
        if batch_size == 0:
            return
        else:
            pass
        decode_feedback_embedding = self.model.decode_feedback_embedding
        input_ids = forward_batch.input_ids
        if input_ids.numel() < batch_size:
            raise RuntimeError(
                "Qwen3-TTS decode input_ids must contain one row id per request"
            )
        else:
            pass
        if batch_size > decode_feedback_embedding.num_embeddings:
            raise RuntimeError(
                "Qwen3-TTS decode batch exceeds staged feedback embedding rows"
            )
        else:
            pass
        row_ids = self.decode_row_ids(batch_size, input_ids)
        # note (ratish): a decode feeds back its request's newest committed frame.
        text_rows = torch.stack(
            [
                sched_req.data.decode_text_row(len(sched_req.data.req.output_ids) - 1)
                for sched_req in requests
            ]
        )
        with torch.no_grad():
            target = decode_feedback_embedding.weight[:batch_size]
            torch.index_select(
                self.slot_feedback_rows, 0, forward_batch.req_pool_indices, out=target
            )
            target.add_(text_rows)
        # During graph decode, input_ids carries staged embedding row ids.
        input_ids[:batch_size].copy_(row_ids)

    def decode_row_ids(self, batch_size: int, input_ids: torch.Tensor) -> torch.Tensor:
        cached = self.row_ids_cache
        if (
            cached is None
            or cached.numel() < batch_size
            or cached.dtype != input_ids.dtype
            or cached.device != input_ids.device
        ):
            cached = torch.arange(
                max(batch_size, 64),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            self.row_ids_cache = cached
        else:
            pass
        return cached[:batch_size]

    def build_prefill_input_embeds(
        self,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for sched_req in requests:
            data: Qwen3TTSSGLangRequestData = sched_req.data
            req = data.req
            req_len = req.extend_len
            prefix_len = req.prefix_len
            if data.prefill_input_embeds is None:
                data.prefill_input_embeds = data.prompt_input_embeds
            else:
                pass
            if data.prefill_input_embeds is None:
                raise RuntimeError("Qwen3-TTS prefill requires prompt_input_embeds")
            else:
                pass
            prompt_length = int(data.prefill_input_embeds.shape[0])
            extend_end = prefix_len + req_len
            if prefix_len < prompt_length:
                pieces.append(
                    data.prefill_input_embeds[
                        prefix_len : min(extend_end, prompt_length)
                    ]
                )
            else:
                pass
            if extend_end > prompt_length:
                pieces.append(
                    self.replayed_decode_inputs(
                        data,
                        max(prefix_len, prompt_length) - prompt_length,
                        extend_end - prompt_length,
                    )
                )
            else:
                pass
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def replayed_decode_inputs(
        self, data: Qwen3TTSSGLangRequestData, first_frame: int, stop_frame: int
    ) -> torch.Tensor:
        """The decode inputs of generated frames [first_frame, stop_frame) that a retracted
        request replays: each frame's feedback row plus the text row its decode read."""
        feedback_rows = self.model.feedback_rows_from_codes(
            data.committed_codes(first_frame, stop_frame)
        )
        text_rows = torch.stack(
            [
                data.decode_text_row(frame_index)
                for frame_index in range(first_frame, stop_frame)
            ]
        )
        return feedback_rows + text_rows
