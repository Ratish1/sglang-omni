# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS model runner for the OmniScheduler AR stage."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict

import torch
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.utils.common import is_pin_memory_available

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.scheduling.types import (
    RequestOutput,
    SchedulerOutput,
    SchedulerRequest,
)
from sglang_omni.utils.device import supports_device_streams

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSSGLangRequestData
    from sglang_omni.models.qwen3_tts.sglang_model import (
        PredictorGraphStats,
        Qwen3TTSTalker,
    )
    from sglang_omni.scheduling.sglang_backend.output_processor import (
        SGLangOutputProcessor,
    )
else:
    pass


class Qwen3TTSModelInfo(TypedDict):
    predictor_cuda_graph: PredictorGraphStats


@dataclass(frozen=True, kw_only=True)
class Qwen3TTSCodeStep:
    """One step's frames, cloned from the predictor's buffers when the step launched, and
    the event that marks them complete."""

    codes: torch.Tensor
    codes_ready: torch.Event | None


class Qwen3TTSModelRunner(ModelRunner):
    """Runs Qwen3-TTS AR steps and stores generated codec frames per request."""

    model: Qwen3TTSTalker
    tp_worker: ModelWorker
    # Resolved in __init__: each launched step records a readiness event, and the
    # device cannot change under the runner.
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
        self.resolving_code_step: Qwen3TTSCodeStep | None = None
        self.row_ids_cache: torch.Tensor | None = None
        self.leading_silence_mask_frames = leading_silence_mask_frames
        self.silence_codec_ids = silence_codec_ids
        self.codes_event_module = (
            torch.get_device_module(self.device)
            if supports_device_streams(self.device)
            else None
        )
        request_slot_count = int(
            tp_worker.model_runner.req_to_token_pool.req_to_token.shape[0]
        )
        # note (ratish): each request slot's repetition state lives on the device and takes a
        # step's ids when they are sampled, so no step waits for the host to commit the last one.
        self.repetition_scales = torch.ones(
            (request_slot_count, int(self.model.config.vocab_size)),
            dtype=torch.float32,
            device=self.device,
        )
        self.repetition_penalties = torch.ones(
            request_slot_count, dtype=torch.float32, device=self.device
        )
        # note (ratish): the predictor writes each slot's next feedback row here, so the next
        # decode reads its input from the device rather than from host state.
        self.feedback_pool = torch.zeros(
            (request_slot_count, self.model.decode_feedback_embedding.embedding_dim),
            dtype=self.model.decode_feedback_embedding.weight.dtype,
            device=self.device,
        )

    def model_info(self) -> Qwen3TTSModelInfo:
        return {"predictor_cuda_graph": self.model.predictor_graph_stats()}

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        self.model.prepare_decode_buffers(requests)
        # note (ratish): a prefilled slot restarts from its committed ids, which a retracted
        # request carries into its re-prefill.
        request_slots = forward_batch.req_pool_indices[: len(requests)]
        self.repetition_scales.index_fill_(0, request_slots, 1.0)
        self.repetition_penalties.index_copy_(
            0,
            request_slots,
            torch.tensor(
                [sched_req.data.repetition_penalty for sched_req in requests],
                dtype=torch.float32,
                pin_memory=is_pin_memory_available(self.device),
            ).to(self.device, non_blocking=True),
        )
        committed_row_indices: list[int] = []
        committed_token_ids: list[int] = []
        for row_index, sched_req in enumerate(requests):
            output_ids = sched_req.data.req.output_ids
            committed_row_indices.extend([row_index] * len(output_ids))
            committed_token_ids.extend(output_ids)
        if committed_row_indices:
            committed_slots = request_slots[
                torch.tensor(committed_row_indices, device=request_slots.device)
            ]
            self.repetition_scales.index_put_(
                (
                    committed_slots,
                    torch.tensor(committed_token_ids, device=request_slots.device),
                ),
                self.repetition_penalties[committed_slots],
            )
        else:
            pass
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
        self.model.prepare_decode_buffers(requests)
        self.write_feedback_buffers(forward_batch, requests)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        self.resolving_code_step = self.launch_code_step(
            result, forward_batch, requests
        )

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        self.resolving_code_step = self.launch_code_step(
            result, forward_batch, requests
        )

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> Qwen3TTSCodeStep | None:
        return self.launch_code_step(result, forward_batch, requests)

    def post_decode_resolve(
        self,
        launch_buf: Qwen3TTSCodeStep | None,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del result, forward_batch, schedule_batch, requests
        self.resolving_code_step = launch_buf

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

    def mask_leading_silence(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """The frame a row samples is its position past the prompt, which holds whether
        or not the frames before it are committed yet."""
        logits = logits_output.next_token_logits
        sequence_lengths = forward_batch.seq_lens_cpu[: len(requests)].tolist()
        # note (luojiaxuan): a cold-start clone that samples a silence id first
        # tends to stay silent for several frames, so silence ids are excluded
        # from its opening frames.
        for row_index, (scheduled_request, sequence_length) in enumerate(
            zip(requests, sequence_lengths, strict=True)
        ):
            frame_index = sequence_length - len(
                scheduled_request.data.req.origin_input_ids
            )
            if (
                scheduled_request.data.mask_leading_silence
                and frame_index < self.leading_silence_mask_frames
            ):
                logits[row_index].index_fill_(0, self.silence_codec_ids, float("-inf"))
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

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        request_slots = forward_batch.req_pool_indices[: len(requests)]
        # note (ratish): the scaling commutes with the codec suppression that follows, which
        # leaves a masked logit at -inf either way.
        apply_scaling_penalties(
            logits_output.next_token_logits[: len(requests)],
            self.repetition_scales.index_select(0, request_slots),
        )
        if self.leading_silence_mask_frames > 0:
            self.mask_leading_silence(logits_output, forward_batch, requests)
        else:
            pass
        next_token_ids = super().sample_next_token_ids(
            logits_output, forward_batch, schedule_batch, requests
        )
        if forward_batch.forward_mode.is_extend():
            # note (ratish): a middle prefill chunk's token is dropped, so it never counts as
            # generated.
            recorded_row_indices = torch.tensor(
                [
                    row_index
                    for row_index, sched_req in enumerate(requests)
                    if sched_req.data.req.inflight_middle_chunks <= 0
                ],
                dtype=torch.long,
                device=request_slots.device,
            )
            recorded_slots = request_slots[recorded_row_indices]
            recorded_token_ids = next_token_ids[recorded_row_indices]
        else:
            recorded_slots = request_slots
            recorded_token_ids = next_token_ids
        self.repetition_scales.index_put_(
            (recorded_slots, recorded_token_ids),
            self.repetition_penalties[recorded_slots],
        )
        return next_token_ids

    def launch_code_step(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch,
        requests: list[SchedulerRequest],
    ) -> Qwen3TTSCodeStep | None:
        """Enqueue the step's device work after its sample: the predictor, the slots' next
        feedback rows, and a snapshot of the frames that its resolve reads."""
        if result.next_token_ids is None:
            return None
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
        self.feedback_pool.index_copy_(
            0,
            forward_batch.req_pool_indices[: len(requests)],
            self.model.output_embeds[: len(requests)],
        )
        # Note: (Jiaxin Deng) per-row clones were a c32 decode-loop hot spot;
        # rows must stay views of a snapshot, never of the reused graph buffers.
        codes = self.model.output_codes[: len(requests)].detach().clone()
        codes_ready = None
        if self.codes_event_module is not None:
            codes_ready = self.codes_event_module.Event()
            codes_ready.record()
        else:
            pass
        return Qwen3TTSCodeStep(codes=codes, codes_ready=codes_ready)

    def post_process_outputs(
        self,
        result: GenerationBatchResult | None,
        scheduler_output: SchedulerOutput | None,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        code_step = self.resolving_code_step
        self.resolving_code_step = None
        if code_step is None:
            return
        else:
            pass
        eos_id = int(self.model.config.codec_eos_token_id)
        for row_idx, sched_req in enumerate(scheduler_output.requests):
            sglang_request = sched_req.data.req
            req_output = outputs[sched_req.request_id]
            # note (ratish): a middle prefill chunk samples at a prompt row; SGLang drops
            # that token, so it gives no frame or stream chunk either. A request that
            # finished or was retracted after this step launched gets nothing from it.
            if (
                sglang_request.inflight_middle_chunks > 0
                or sglang_request.finished()
                or sglang_request.is_retracted
                or req_output.data is None
                or int(req_output.data) == eos_id
            ):
                continue
            else:
                pass
            code_chunk = code_step.codes[row_idx]
            sched_req.data.output_codes.append(code_chunk)
            sched_req.data.latest_stream_code_chunk = code_chunk
            sched_req.data.codes_ready_event = code_step.codes_ready

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
        # note (ratish): the text row follows from the position, so the first decode after
        # the prompt reads the first trailing text row and every decode past them the pad.
        text_rows: list[torch.Tensor] = []
        for sched_req, sequence_length in zip(
            requests, forward_batch.seq_lens_cpu[:batch_size].tolist(), strict=True
        ):
            data: Qwen3TTSSGLangRequestData = sched_req.data
            text_index = sequence_length - len(data.req.origin_input_ids) - 1
            text_rows.append(
                data.text_rows[min(text_index, data.text_rows.shape[0] - 1)]
            )
        with torch.no_grad():
            target = decode_feedback_embedding.weight[:batch_size]
            torch.index_select(
                self.feedback_pool,
                0,
                forward_batch.req_pool_indices[:batch_size],
                out=target,
            )
            target.add_(torch.stack(text_rows, dim=0))
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
            req_len = int(req.extend_range.length)
            prefix_len = len(req.prefix_indices)
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
            request_pieces: list[torch.Tensor] = []
            if prefix_len < prompt_length:
                request_pieces.append(
                    data.prefill_input_embeds[
                        prefix_len : min(extend_end, prompt_length)
                    ]
                )
            else:
                pass
            if extend_end > prompt_length:
                # note (ratish): a retracted request replays its generated positions from its
                # committed frames, each row the feedback and text row its decode consumed.
                generated_start = max(prefix_len, prompt_length) - prompt_length
                generated_stop = extend_end - prompt_length
                frame_codes = torch.stack(
                    data.output_codes[generated_start:generated_stop]
                )
                text_indices = torch.arange(
                    generated_start,
                    generated_start + frame_codes.shape[0],
                    device=frame_codes.device,
                ).clamp_max(data.text_rows.shape[0] - 1)
                request_pieces.append(
                    self.model.feedback_rows_from_codes(frame_codes)
                    + data.text_rows[text_indices]
                )
            else:
                pass
            piece_rows = sum(int(piece.shape[0]) for piece in request_pieces)
            if piece_rows != req_len:
                raise RuntimeError(
                    f"Qwen3-TTS prefill embed mismatch for {req.rid}: "
                    f"have {piece_rows} rows, need {req_len}"
                )
            else:
                pass
            pieces.extend(request_pieces)
        return torch.cat(pieces, dim=0).to(
            device=forward_batch.input_ids.device,
            dtype=next(self.model.parameters()).dtype,
        )
