# SPDX-License-Identifier: Apache-2.0
"""SGLang execution-contract adapter.

SGLang 0.5.15 made the scheduler/worker boundary an explicit protocol:

* decode tokens cross iterations through ``FutureMap``;
* scheduler writes and model forwards use separate CUDA streams;
* the decode CUDA-graph runner publishes a read-done event which gates the
  next scheduler mutation; and
* a mutable ``ScheduleBatch`` is isolated while ``ForwardBatch.init_new``
  consumes its one-shot fields.

Omni owns its scheduler loop and model-runner wrapper, so it cannot rely on
``sglang.srt.managers.scheduler.Scheduler.run_batch`` to provide that protocol.
This adapter is the single compatibility boundary used by both synchronous and
lookahead execution.
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Iterator
from typing import Any

import torch


class SGLangExecutionBridge:
    """Adapt Omni's custom runner to SGLang's 0.5.15 execution contract."""

    def __init__(
        self,
        *,
        device: torch.device,
        worker: Any,
        req_to_token_pool: Any,
        spec_algorithm: Any,
        enable_stream_overlap: bool,
    ) -> None:
        from sglang.srt.managers.overlap_utils import RelayPayload

        self.device = device
        self.worker = worker
        self.runner = worker.model_runner
        self.device_module = torch.get_device_module(device)
        self.enable_stream_overlap = bool(
            enable_stream_overlap and device.type == "cuda"
        )
        self.future_map = spec_algorithm.create_future_map(
            device,
            req_to_token_pool,
            needs_cpu_seq_lens=True,
        )
        self._relay_payload_type = RelayPayload

        self.schedule_stream = None
        self.forward_stream = (
            self.runner.forward_stream if self.enable_stream_overlap else None
        )
        self._batch_record_buf: list[Any | None] = [None, None]
        self._batch_record_slot = 0

    def loop_context(self):
        """Return the scheduling-stream context for an overlap event loop."""
        if not self.enable_stream_overlap:
            return contextlib.nullcontext()
        if self.schedule_stream is None:
            self.schedule_stream = self.device_module.Stream(priority=0)
        return self.device_module.stream(self.schedule_stream)

    def before_schedule(self) -> None:
        """Fence scheduler writes on the preceding forward's last shared read."""
        if not self.enable_stream_overlap:
            return
        assert self.schedule_stream is not None
        read_done = self.runner.war_fastpath_read_done_event
        if read_done is not None:
            self.schedule_stream.wait_event(read_done)
            self.runner.war_fastpath_read_done_event = None
        else:
            self.schedule_stream.wait_stream(self.forward_stream)

    @contextlib.contextmanager
    def forward_context(self, batch: Any) -> Iterator[None]:
        """Resolve inputs and isolate ``ScheduleBatch`` for one forward."""
        from sglang.srt.managers.overlap_utils import resolve_forward_inputs

        stream_ctx = contextlib.nullcontext()
        if self.enable_stream_overlap:
            assert self.schedule_stream is not None
            self.forward_stream.wait_stream(self.schedule_stream)
            stream_ctx = self.device_module.stream(self.forward_stream)

        with stream_ctx:
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
            if self.enable_stream_overlap:
                self._record_batch_tensors(batch)
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

    def _record_batch_tensors(self, batch: Any) -> None:
        """Keep every ScheduleBatch field alive for the two-iteration window."""
        snapshot = [
            getattr(batch, field.name, None) for field in dataclasses.fields(batch)
        ]
        self._batch_record_slot ^= 1
        self._batch_record_buf[self._batch_record_slot] = [batch, snapshot]
