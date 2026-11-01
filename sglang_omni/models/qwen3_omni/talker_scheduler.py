"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import copy
import logging
import time
from collections import deque
from dataclasses import replace

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.sampling.penaltylib import BatchedPenalizerOrchestrator
from sglang.srt.server_args import ServerArgs
from typing_extensions import Unpack

from sglang_omni.models.qwen3_omni.config import (
    ENABLE_TALKER_START_TOPOLOGY,
    MIN_PARTIAL_START_CHUNKS,
    TALKER_START_MIN_CHUNKS,
)
from sglang_omni.models.qwen3_omni.payload_types import NO_SPEECH
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler, OmniSchedulerArguments
from sglang_omni.scheduling.sglang_backend.ar_session import SessionUnit
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)

_CHUNK_WAIT_LOG_INTERVAL_S = 10.0

# note (ratish): a reply with no spoken token has nothing to speak; a reply of one
# token terminates.
MIN_SPOKEN_TOKENS = 1


def configure_talker_server_args(
    server_args: ServerArgs, *, feedback_enabled: bool = True
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were requested so the caller can capture them
    after the model worker is constructed.
    """
    from sglang.srt.arg_groups.model_override_base import resolved_view

    cfg = resolved_view(server_args)
    want_cuda_graph = not bool(cfg.disable_cuda_graph)
    overrides = {"disable_radix_cache": True, "chunked_prefill_size": 0}
    if feedback_enabled:
        overrides["disable_overlap_schedule"] = True
    else:
        pass
    override_server_args(server_args, "qwen3_omni.talker", **overrides)
    return want_cuda_graph


class QwenTalkerScheduler(OmniScheduler[SGLangARRequestData]):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    talker_start_topology: bool = ENABLE_TALKER_START_TOPOLOGY
    chunk_wait_steps: int = 0
    chunk_wait_last_log_s: float = 0.0
    held_decode_batch: ScheduleBatch | None = None

    def __init__(
        self,
        *args: object,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
        im_end_token_id: int | None = None,
        enable_talker_start_topology: bool | None = None,
        **kwargs: Unpack[OmniSchedulerArguments[SGLangARRequestData]],
    ) -> None:
        super().__init__(*args, **kwargs)
        if partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS:
            raise ValueError(
                f"partial_start_min_chunks must be >= {MIN_PARTIAL_START_CHUNKS}, got {partial_start_min_chunks}"
            )
        else:
            pass
        self.enable_partial_start = bool(enable_partial_start)
        self.partial_start_min_chunks = int(partial_start_min_chunks)
        self.im_end_token_id = im_end_token_id
        self.talker_start_topology = self.enable_partial_start and (
            ENABLE_TALKER_START_TOPOLOGY
            if enable_talker_start_topology is None
            else bool(enable_talker_start_topology)
        )
        self.chunk_wait_steps = 0
        self.chunk_wait_last_log_s = 0.0
        if self.talker_start_topology:
            logger.info(
                "talker-start topology on: building at %d thinker chunk(s); later chunks gate decode per step (partial_start_min_chunks=%d applies to the legacy path only)",
                TALKER_START_MIN_CHUNKS,
                self.partial_start_min_chunks,
            )
        else:
            pass

    def count_usable_prefetched_chunks(self, prefetched: list[StreamItem]) -> int:
        im_end = self.im_end_token_id
        if im_end is None or not prefetched:
            return len(prefetched)
        else:
            pass
        metadata = getattr(prefetched[-1], "metadata", None) or {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == int(im_end):
            return len(prefetched) - 1
        else:
            pass
        return len(prefetched)

    def is_request_build_ready(
        self, payload: StagePayload, *, pending_stream_done: bool
    ) -> bool:
        if pending_stream_done:
            return True
        else:
            pass
        if not self.enable_partial_start:
            return False
        else:
            pass
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        usable = self.count_usable_prefetched_chunks(prefetched)
        if self.talker_start_topology:
            return usable >= TALKER_START_MIN_CHUNKS
        else:
            pass
        return usable >= self.partial_start_min_chunks

    def complete_without_generation(
        self, payload: StagePayload, session_unit: SessionUnit | None
    ) -> StagePayload | None:
        """A finished reply too short to speak passes on without audio."""
        if not payload.prefetched_stream_done or (
            self.count_usable_prefetched_chunks(payload.prefetched_chunks)
            >= MIN_SPOKEN_TOKENS
        ):
            return None
        elif session_unit is None:
            return replace(payload, data={**payload.data, NO_SPEECH: True})
        else:
            return self.session_bridge.adapter.complete_without_generation(
                session_unit.session_identity, session_unit.chunk, payload
            )

    def initialize_request_stream_state(
        self, req_data: SGLangARRequestData, payload: StagePayload
    ) -> None:
        del req_data, payload
        return None

    def should_recheck_deferred_request_on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> bool:
        del request_id, chunk
        return self.enable_partial_start

    def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch | None:
        """Decode the rows whose next input arrived; hold the others with their KV until the step's result is processed."""
        unready_rows = self.model_runner.unready_decode_rows(batch)
        if not unready_rows:
            return Scheduler.update_running_batch(self, batch)
        elif len(unready_rows) == len(batch.reqs):
            self.note_chunk_wait(batch)
            return None
        else:
            self.hold_decode_rows(batch, unready_rows)
            return Scheduler.update_running_batch(self, batch)

    def hold_decode_rows(self, batch: ScheduleBatch, unready_rows: list[int]) -> None:
        held_batch = copy.copy(batch)
        held_batch.sampling_info = copy.copy(batch.sampling_info)
        # note (ratish): the talker's decode samples through its own repetition mask, so held
        # rows may restart the batch penalizers; merging back needs only matching rows.
        held_batch.sampling_info.penalizer_orchestrator = BatchedPenalizerOrchestrator(
            vocab_size=batch.sampling_info.vocab_size,
            batch=held_batch,
            penalizers=set(batch.sampling_info.penalizer_orchestrator.penalizers),
        )
        held_batch.filter_batch(keep_indices=unready_rows)
        held_rows = set(unready_rows)
        batch.filter_batch(
            keep_indices=[row for row in range(len(batch.reqs)) if row not in held_rows]
        )
        self.held_decode_batch = held_batch

    def merge_held_decode_rows(self) -> None:
        held_batch = self.held_decode_batch
        if held_batch is None:
            return
        elif self.running_batch.is_empty():
            self.running_batch = held_batch
        else:
            self.running_batch.merge_batch(held_batch)
        self.held_decode_batch = None

    def process_batch_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        super().process_batch_result(batch, result)
        self.merge_held_decode_rows()

    def handle_batch_failure(self, batch: ScheduleBatch, error: Exception) -> None:
        super().handle_batch_failure(batch, error)
        self.merge_held_decode_rows()

    def note_chunk_wait(self, batch: ScheduleBatch) -> None:
        self.chunk_wait_steps += 1
        logger.debug("Deferring decode batch until talker feedback/text input is ready")
        now = time.monotonic()
        if now - self.chunk_wait_last_log_s < _CHUNK_WAIT_LOG_INTERVAL_S:
            return
        else:
            pass
        self.chunk_wait_last_log_s = now
        logger.info(
            "talker chunk gate: %d decode steps deferred so far (current batch rows=%d)",
            self.chunk_wait_steps,
            len(getattr(batch, "reqs", ()) or ()),
        )

    def get_next_batch_to_run(self) -> ScheduleBatch | None:
        batch = super().get_next_batch_to_run()
        if batch is None:
            self.merge_held_decode_rows()
        else:
            pass
        return batch

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and (not self.running_batch.is_empty()):
            return
        else:
            pass
        if self.waiting_queue:
            return
        else:
            pass
        super().self_check_during_idle()

    @staticmethod
    def append_stream_chunk_default(
        req_data: SGLangARRequestData, chunk: StreamItem
    ) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        else:
            pass
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def mark_stream_done(self, req_data: SGLangARRequestData) -> None:
        if self.stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        else:
            pass
        self.stream_done_handler(req_data)
