# SPDX-License-Identifier: Apache-2.0
"""OmniScheduler specialization that keeps MiniMax Music 3 CFG rows paired."""

from __future__ import annotations

from typing import Any

from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS

from sglang_omni.scheduling.omni_scheduler import OmniScheduler

from .sglang_request_builder import cfg_uncond_rid, is_cfg_uncond_rid


class MiniMaxMusic3Scheduler(OmniScheduler):
    """Admit, decode and retire every request as a CFG row pair."""

    def _enqueue_built_request(
        self,
        payload: Any,
        pending_stream_done: bool,
        req_data: Any,
        *,
        request_admission_lock_held: bool = False,
    ) -> None:
        super()._enqueue_built_request(
            payload,
            pending_stream_done,
            req_data,
            request_admission_lock_held=request_admission_lock_held,
        )
        uncond = req_data.cfg_uncond
        if uncond is None:
            return
        if request_admission_lock_held:
            self._enqueue_cfg_uncond(req_data, uncond)
            return
        with self._request_admission_lock:
            self._enqueue_cfg_uncond(req_data, uncond)

    def _enqueue_cfg_uncond(self, req_data: Any, uncond: Any) -> None:
        cond_req = req_data.req
        if not self.waiting_queue or self.waiting_queue[-1] is not cond_req:
            return
        req = uncond.req
        self._normalize_req_token_arrays(req)
        req._coalesce_enqueue_t = cond_req._coalesce_enqueue_t
        req._omni_terminal_claimed = False
        req._omni_data = uncond
        self.waiting_queue.append(req)

    def get_new_batch_prefill(self, running_batch: Any) -> Any:
        queue = self.waiting_queue
        prefill_budget = self.max_prefill_tokens
        expanded_pair_budget = False
        if len(queue) >= 2:
            assert queue[0]._omni_data.cfg_uncond is queue[1]._omni_data
            page_size = int(self.page_size)
            pair_input_tokens = sum(
                -(-len(req.origin_input_ids) // page_size) * page_size
                for req in queue[:2]
            )
            if pair_input_tokens >= prefill_budget:
                self.max_prefill_tokens = pair_input_tokens + 1
                expanded_pair_budget = True

        limit = self._pair_admission_limit(queue, running_batch)
        if expanded_pair_budget:
            limit = min(limit, 2)
        elif limit >= len(queue):
            return super().get_new_batch_prefill(running_batch)
        deferred = queue[limit:]
        del queue[limit:]
        try:
            return super().get_new_batch_prefill(running_batch)
        finally:
            self.waiting_queue.extend(deferred)
            self.max_prefill_tokens = prefill_budget

    def _pair_admission_limit(self, queue: list, running_batch: Any) -> int:
        """How many leading queue entries the adder may see, always whole pairs."""
        allocatable = int(self.get_num_allocatable_reqs(len(running_batch.reqs)))
        limit = min(len(queue), max(0, allocatable))
        limit -= limit % 2

        page_size = int(self.page_size)
        remaining_input_tokens = int(self.max_prefill_tokens)
        running_token_reserve = sum(
            min(
                req.sampling_params.max_new_tokens - len(req.output_ids),
                CLIP_MAX_NEW_TOKENS,
            )
            * self.new_token_ratio_tracker.current
            for req in running_batch.reqs
        )
        remaining_total_tokens = (
            self.token_to_kv_pool_allocator.available_size()
            + self.tree_cache.evictable_size()
            - running_token_reserve
        )
        for index in range(0, limit, 2):
            cond, uncond = queue[index : index + 2]
            pair_input_tokens = 0
            pair_total_tokens = 0
            for req in (cond, uncond):
                input_tokens = -(-len(req.origin_input_ids) // page_size) * page_size
                new_tokens = min(
                    max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
                    CLIP_MAX_NEW_TOKENS,
                )
                pair_input_tokens += input_tokens
                pair_total_tokens += input_tokens + new_tokens + page_size
            if (
                pair_input_tokens >= remaining_input_tokens
                or pair_total_tokens >= remaining_total_tokens
            ):
                return index
            remaining_input_tokens -= pair_input_tokens
            remaining_total_tokens -= pair_total_tokens
        return limit

    def stream_output(
        self, reqs: Any, return_logprob: bool = False, skip_req: Any = None
    ) -> None:
        conditioned = []
        for req in reqs:
            if not self._is_cfg_uncond(req):
                conditioned.append(req)
                continue
            if req.finished():
                self._close_completed_request(req)
        super().stream_output(conditioned, return_logprob, skip_req)

    def abort(self, request_id: str, *, defer_running_cleanup: bool = True) -> None:
        super().abort(request_id, defer_running_cleanup=defer_running_cleanup)
        if is_cfg_uncond_rid(request_id):
            return
        super().abort(
            cfg_uncond_rid(request_id), defer_running_cleanup=defer_running_cleanup
        )

    @staticmethod
    def _is_cfg_uncond(req: Any) -> bool:
        data = getattr(req, "_omni_data", None)
        return data is not None and data.is_cfg_uncond


__all__ = ["MiniMaxMusic3Scheduler"]
