# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni talker scheduler policy on top of the generic OmniScheduler."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from sglang_omni.models.qwen3_omni.config import MIN_PARTIAL_START_CHUNKS
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.vendor.sglang.server_args import override_server_args

logger = logging.getLogger(__name__)


# SGLang admits a request only while the KV pool holds every running
# request's reservation, min(max_new_tokens, 4096) times new_token_ratio,
# and new_token_ratio starts at 0.7 times schedule_conservativeness and decays
# to 0.14 of that over 600 decode steps. The talker keeps the official
# max_new_tokens of 4096 as its stop, but it emits 12.5 codec frames per audio
# second (42 frames median, 93 at most on the voice clone set), so at 1.0 each
# running request reserves 2867 tokens it never uses and a 21373 token pool
# runs six to nine talker requests. 0.1 reserves 287 tokens per running
# request at the start of a burst, three times the median output, and the
# pool fills to max_running_requests. Outputs that outrun the reservation are
# handled the way SGLang handles them for every model: the scheduler retracts
# the youngest rows and the talker replays them from its decode input history
# (QwenTalkerModelRunner._decode_input_history).
TALKER_SCHEDULE_CONSERVATIVENESS = 0.1


def configure_talker_server_args(
    server_args: Any,
    *,
    feedback_enabled: bool = True,
) -> bool:
    """Apply talker-specific scheduler/runtime defaults.

    Returns whether CUDA graphs were requested so the caller can capture them
    after the model worker is constructed.
    """

    want_cuda_graph = not bool(server_args.disable_cuda_graph)
    overrides = {
        "disable_radix_cache": True,
        "chunked_prefill_size": 0,
        "schedule_conservativeness": TALKER_SCHEDULE_CONSERVATIVENESS,
    }
    if feedback_enabled:
        overrides["disable_overlap_schedule"] = True
    override_server_args(server_args, "qwen3_omni.talker", **overrides)
    return want_cuda_graph


class QwenTalkerScheduler(OmniScheduler):
    """Talker scheduler with Qwen-specific request and decode readiness."""

    def __init__(
        self,
        *args: Any,
        enable_partial_start: bool = False,
        partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
        im_end_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if partial_start_min_chunks < MIN_PARTIAL_START_CHUNKS:
            raise ValueError(
                f"partial_start_min_chunks must be >= {MIN_PARTIAL_START_CHUNKS}, "
                f"got {partial_start_min_chunks}"
            )
        self._enable_partial_start = bool(enable_partial_start)
        self._partial_start_min_chunks = int(partial_start_min_chunks)
        self._im_end_token_id = im_end_token_id

    def _count_usable_prefetched_chunks(self, prefetched: list[Any]) -> int:
        im_end = self._im_end_token_id
        if im_end is None or not prefetched:
            return len(prefetched)
        metadata = getattr(prefetched[-1], "metadata", None) or {}
        token_id = metadata.get("token_id")
        if token_id is not None and int(token_id) == int(im_end):
            return len(prefetched) - 1
        return len(prefetched)

    def _is_request_build_ready(
        self,
        payload: Any,
        *,
        pending_stream_done: bool,
    ) -> bool:
        if pending_stream_done:
            return True
        if not self._enable_partial_start:
            return False
        prefetched = getattr(payload, "prefetched_chunks", None) or []
        return (
            self._count_usable_prefetched_chunks(prefetched)
            >= self._partial_start_min_chunks
        )

    def _initialize_request_stream_state(self, req_data: Any, payload: Any) -> None:
        del req_data, payload
        return None

    def _should_recheck_deferred_request_on_stream_chunk(
        self, request_id: str, chunk: Any
    ) -> bool:
        del request_id, chunk
        return self._enable_partial_start

    def _is_batch_ready_to_run(self, batch: Any) -> bool:
        if (
            batch is not None
            and batch.forward_mode.is_decode()
            and self._model_runner is not None
            and hasattr(self._model_runner, "is_decode_batch_ready")
            and not self._model_runner.is_decode_batch_ready(batch)
        ):
            logger.debug(
                "Deferring decode batch until talker feedback/text input is ready"
            )
            return False
        return True

    def get_next_batch_to_run(self) -> Any | None:
        batch = super().get_next_batch_to_run()
        if batch is not None and not self._is_batch_ready_to_run(batch):
            self._rollback_decode_prep_after_skip(batch)
            return None
        return batch

    def _rollback_decode_prep_after_skip(self, batch: Any) -> None:
        # Note(Chenchen Hong, Xuesong): This is talker-only. It does not fully
        # invert prepare_for_decode; talker disables overlap/spec/Mamba/hisparse,
        # and the penalizer's cumulate scatter_ is idempotent under the talker's
        # own SamplingBatchInfo. Zero the req_to_token_pool cell that
        # alloc_for_decode wrote at (req_pool_indices, pre-increment seq_lens);
        # seq_lens_sum stays untouched (always None after prepare_for_decode,
        # recomputed at the next forward).
        if not batch.forward_mode.is_decode():
            return
        if batch.out_cache_loc is not None:
            self.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            batch.out_cache_loc = None
        for req in batch.reqs:
            req.decode_batch_idx -= 1
            req.kv_committed_len -= 1
            req.kv.kv_allocated_len -= 1
        batch.seq_lens.sub_(1)
        batch.seq_lens_cpu.sub_(1)
        batch.orig_seq_lens.sub_(1)
        batch.req_to_token_pool.req_to_token[batch.req_pool_indices, batch.seq_lens] = 0

    def self_check_during_idle(self) -> None:
        if self.running_batch is not None and not self.running_batch.is_empty():
            return
        if self.waiting_queue:
            return
        super().self_check_during_idle()

    @staticmethod
    def _append_stream_chunk_default(req_data: Any, chunk: Any) -> None:
        pending_text_queue = getattr(req_data, "pending_text_queue", None)
        if pending_text_queue is None:
            pending_text_queue = deque()
            req_data.pending_text_queue = pending_text_queue
        pending_text_queue.append(getattr(chunk, "data", chunk))

    def _mark_stream_done(self, req_data: Any) -> None:
        if self._stream_done_handler is None:
            req_data.thinker_chunks_done = True
            return
        self._stream_done_handler(req_data)
