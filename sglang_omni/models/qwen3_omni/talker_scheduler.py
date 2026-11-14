"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import replace

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import async_h2d
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

WAITING_ROW_LOG_INTERVAL_S = 10.0

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
    waiting_row_steps: int = 0
    waiting_row_last_log_s: float = 0.0
    waiting_rows: ScheduleBatch | None = None

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
        self.waiting_row_steps = 0
        self.waiting_row_last_log_s = 0.0
        self.waiting_rows = None
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
        """Decode the rows whose next input has arrived; the others sit out the step with their KV."""
        waiting_row_indices = self.model_runner.rows_without_next_decode_input(batch)
        if not waiting_row_indices:
            return super().update_running_batch(batch)
        else:
            pass
        self.note_waiting_rows(len(waiting_row_indices), batch.batch_size())
        if len(waiting_row_indices) == batch.batch_size():
            return None
        else:
            pass
        self.waiting_rows = self.take_rows(batch, waiting_row_indices)
        decode_batch = super().update_running_batch(batch)
        if decode_batch is None or decode_batch.is_empty():
            self.return_waiting_rows(batch)
        else:
            pass
        return decode_batch

    def take_rows(self, batch: ScheduleBatch, row_indices: list[int]) -> ScheduleBatch:
        """Move rows out of the batch with their KV slot, length and next input id; no KV is freed."""
        rows = ScheduleBatch.init_new(
            reqs=[batch.reqs[row_index] for row_index in row_indices],
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
            model_config=self.model_config,
            enable_overlap=self.enable_overlap,
            spec_algorithm=self.spec_algorithm,
        )
        host_indices = torch.tensor(row_indices, dtype=torch.int64)
        device_indices = async_h2d(row_indices, dtype=torch.int64, device=batch.device)
        rows.forward_mode = ForwardMode.DECODE
        rows.req_pool_indices = batch.req_pool_indices[device_indices]
        rows.req_pool_indices_cpu = batch.req_pool_indices_cpu[host_indices]
        rows.seq_lens = batch.seq_lens[device_indices]
        rows.seq_lens_cpu = batch.seq_lens_cpu[host_indices]
        rows.orig_seq_lens = batch.orig_seq_lens[device_indices]
        rows.input_ids = (
            None if batch.input_ids is None else batch.input_ids[device_indices]
        )
        rows.multimodal_inputs = (
            None
            if batch.multimodal_inputs is None
            else [batch.multimodal_inputs[row_index] for row_index in row_indices]
        )
        # note (ratish): the talker samples from its own buffers, so these rows get fresh sampling state, as a batch built for rows already holding KV does.
        rows.sampling_info = SamplingBatchInfo.from_schedule_batch(
            rows, self.model_config.vocab_size
        )
        taken = set(row_indices)
        batch.filter_batch(
            keep_indices=[
                row_index
                for row_index in range(batch.batch_size())
                if row_index not in taken
            ]
        )
        return rows

    def return_waiting_rows(self, batch: ScheduleBatch) -> None:
        rows = self.waiting_rows
        if rows is None:
            return
        else:
            pass
        self.waiting_rows = None
        batch.merge_batch(rows)

    def process_batch_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> None:
        super().process_batch_result(batch, result)
        self.return_waiting_rows(batch)

    def handle_batch_failure(self, batch: ScheduleBatch, error: Exception) -> None:
        super().handle_batch_failure(batch, error)
        self.return_waiting_rows(batch)

    def note_waiting_rows(self, waiting_rows: int, batch_rows: int) -> None:
        self.waiting_row_steps += waiting_rows
        now = time.monotonic()
        if now - self.waiting_row_last_log_s < WAITING_ROW_LOG_INTERVAL_S:
            return
        else:
            pass
        self.waiting_row_last_log_s = now
        logger.info(
            "talker rows waiting for thinker text: %d row steps so far (now %d of %d rows)",
            self.waiting_row_steps,
            waiting_rows,
            batch_rows,
        )

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
