# SPDX-License-Identifier: Apache-2.0
"""SGLang execution-contract adapter.

SGLang 0.5.15 made the scheduler/worker boundary an explicit protocol:

* decode tokens cross iterations through ``FutureMap``; and
* a mutable ``ScheduleBatch`` is isolated while ``ForwardBatch.init_new``
  consumes its one-shot fields.

Omni owns its scheduler loop and model-runner wrapper, so it cannot rely on
``sglang.srt.managers.scheduler.Scheduler.run_batch`` to provide that protocol.
This adapter is the single compatibility boundary used by both synchronous and
lookahead execution.

Upstream's protocol also splits scheduler writes and model forwards onto
separate CUDA streams, fenced by the decode CUDA-graph read-done event. That
two-stream contract belongs to the upstream-style overlap loop, which Omni
refuses to run (``OmniScheduler._event_loop_overlap`` raises), so this bridge
is deliberately single-stream: launch-current/resolve-previous on one stream.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any

import torch


def attn_forward_context(attn_backend: Any):
    """Enter SGLang's ambient ``ForwardContext`` unless one is already active.

    0.5.15 attention backends read the context that ``Scheduler.run_batch``
    installs; Omni's wrapped forwards run outside that loop, so each wrapper
    enters the context itself when no caller has.
    """
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
        has_forward_context,
    )

    if has_forward_context():
        return contextlib.nullcontext()
    return forward_context(ForwardContext(attn_backend=attn_backend))


class SGLangExecutionBridge:
    """Adapt Omni's custom runner to SGLang's 0.5.15 execution contract."""

    def __init__(
        self,
        *,
        device: torch.device,
        worker: Any,
        req_to_token_pool: Any,
        spec_algorithm: Any,
    ) -> None:
        from sglang.srt.managers.overlap_utils import RelayPayload

        self.device = device
        self.worker = worker
        self.runner = worker.model_runner
        self.device_module = torch.get_device_module(device)
        self.future_map = spec_algorithm.create_future_map(
            device,
            req_to_token_pool,
            needs_cpu_seq_lens=True,
        )
        self._relay_payload_type = RelayPayload

    @contextlib.contextmanager
    def forward_context(self, batch: Any) -> Iterator[None]:
        """Resolve inputs and isolate ``ScheduleBatch`` for one forward."""
        from sglang.srt.managers.overlap_utils import resolve_forward_inputs

        resolve_forward_inputs(batch, self.future_map)

        snapshot_full = not batch.spec_algorithm.is_none()
        scheduler_snapshot = (
            {
                field.name: getattr(batch, field.name)
                for field in dataclasses.fields(batch)
            }
            if snapshot_full
            else None
        )
        scheduler_sampling_info = batch.sampling_info
        if scheduler_sampling_info is not None:
            batch.sampling_info = scheduler_sampling_info.copy_for_forward()
        try:
            yield
        finally:
            if snapshot_full:
                for name, value in scheduler_snapshot.items():
                    setattr(batch, name, value)
            else:
                batch.sampling_info = scheduler_sampling_info

    def publish_next_tokens(
        self,
        batch: Any,
        next_token_ids: torch.Tensor | None,
    ) -> None:
        """Publish one forward's GPU token relay and retire live ``input_ids``."""
        if next_token_ids is None:
            return
        if next_token_ids.device != self.device:
            next_token_ids = next_token_ids.to(self.device, non_blocking=True)
        indices = batch.req_pool_indices
        self.future_map.stash(
            indices,
            self._relay_payload_type(bonus_tokens=next_token_ids),
        )
        self.future_map.publish(indices, batch.seq_lens + 1)
        # SGLang 0.5.15 resolves the next decode input from FutureMap at forward
        # entry. Keeping a direct tensor here reintroduces the removed 0.5.12
        # contract and is unsafe when the live batch is filtered on another
        # stream.
        batch.input_ids = None

    def record_completion(self):
        """Record completion of the current launch on its producing stream."""
        event = self.device_module.Event()
        event.record()
        return event

