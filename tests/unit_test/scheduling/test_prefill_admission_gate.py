# SPDX-License-Identifier: Apache-2.0
"""Behavior tests for work-conserving prefill admission.

Admission defers only while a decode step runs this iteration anyway and the
arrival stream is visibly still delivering (an enqueue since the previous
decision, or builds submitted to the executor). It never defers into an idle
GPU; it admits once the waiting cohort would outnumber the unfinished running
requests plus submitted builds, once it meets the free running slots, or once
it fills the prefill token budget; and it treats requests re-entering the
queue without an enqueue (decode retract) as stalled arrivals rather than
fresh ones. Chunked prefill in flight and the disabled gate pass straight
through. Tested against a stub scheduler with the upstream call patched to a
sentinel.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import NextBatchPlan  # noqa: E402

from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

_UPSTREAM_BATCH = object()


def _req(tokens: int = 100):
    return SimpleNamespace(origin_input_ids=list(range(tokens)))


def _running_req(finished: bool = False):
    return SimpleNamespace(finished=lambda: finished)


def _busy_batch(running: int = 4, finished: int = 0):
    reqs = [_running_req() for _ in range(running)]
    reqs.extend(_running_req(finished=True) for _ in range(finished))
    return SimpleNamespace(reqs=reqs, is_empty=lambda: False, is_prefill_only=False)


def _idle_batch():
    return SimpleNamespace(reqs=[], is_empty=lambda: True, is_prefill_only=False)


def _prefill_only_batch():
    return SimpleNamespace(
        reqs=[_running_req()], is_empty=lambda: False, is_prefill_only=True
    )


class _StubScheduler:
    """The attribute surface get_new_batch_prefill touches."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.defer_prefill_during_decode = enabled
        self.chunked_req = None
        self.waiting_queue: list = []
        self.running_batch = _busy_batch()
        self.max_prefill_tokens = 4096
        self.chunked_prefill_size = None
        self.max_running_requests = 64
        self._request_admission_lock = threading.RLock()
        self._pending_request_builds: dict = {}
        self._backlogged_request_build_payloads: list = []
        self._admission_arrivals_seen = 0
        self._admission_arrivals_at_last_decision = 0

    def enqueue(self, req) -> None:
        self._admission_arrivals_seen += 1
        self.waiting_queue.append(req)

    def get_num_allocatable_reqs(self, running_bs: int) -> int:
        return self.max_running_requests - running_bs

    def get_new_batch_prefill(self):
        plan = OmniScheduler.get_new_batch_prefill(self, self.running_batch)
        return plan.batch_to_run


@pytest.fixture()
def upstream():
    with mock.patch.object(
        omni_scheduler._Upstream,
        "get_new_batch_prefill",
        return_value=NextBatchPlan(batch_to_run=_UPSTREAM_BATCH, running_batch=None),
    ) as patched:
        yield patched


def test_disabled_gate_passes_through(upstream) -> None:
    scheduler = _StubScheduler(enabled=False)
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH
    upstream.assert_called_once()


def test_chunked_prefill_passes_through(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.chunked_req = object()
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_empty_queue_passes_through(upstream) -> None:
    scheduler = _StubScheduler()

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_idle_decode_admits_immediately(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.running_batch = _idle_batch()
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_prefill_only_running_batch_admits_immediately(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.running_batch = _prefill_only_batch()
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_idle_decode_admits_despite_pending_builds(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.running_batch = _idle_batch()
    scheduler.enqueue(_req())
    scheduler._pending_request_builds["r1"] = object()

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_defers_while_arrivals_flow_during_decode(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is None
    upstream.assert_not_called()


def test_deferral_preserves_the_running_batch(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.enqueue(_req())

    plan = OmniScheduler.get_new_batch_prefill(scheduler, scheduler.running_batch)

    assert plan.batch_to_run is None
    assert plan.running_batch is scheduler.running_batch


def test_admits_when_the_arrival_stream_stalls(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is None
    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_each_new_arrival_extends_the_deferral(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is None
    scheduler.enqueue(_req())
    assert scheduler.get_new_batch_prefill() is None
    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_defers_while_builds_pending_without_new_arrivals(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.enqueue(_req())
    scheduler._pending_request_builds["r1"] = object()

    assert scheduler.get_new_batch_prefill() is None
    assert scheduler.get_new_batch_prefill() is None
    scheduler._pending_request_builds.clear()
    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_admission_rearms_for_the_next_wave(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is None
    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH
    scheduler.enqueue(_req())
    assert scheduler.get_new_batch_prefill() is None
    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_retracted_requests_do_not_read_as_arrivals(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.enqueue(_req())
    assert scheduler.get_new_batch_prefill() is None

    scheduler.waiting_queue.append(_req())
    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_cohort_matching_the_running_batch_admits_despite_fresh_arrivals(
    upstream,
) -> None:
    scheduler = _StubScheduler()
    scheduler.running_batch = _busy_batch(running=2)
    scheduler.enqueue(_req())
    assert scheduler.get_new_batch_prefill() is None

    scheduler.enqueue(_req())
    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_submitted_builds_extend_the_hold(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.running_batch = _busy_batch(running=2)
    scheduler._pending_request_builds["r1"] = object()
    scheduler.enqueue(_req())
    scheduler.enqueue(_req())
    assert scheduler.get_new_batch_prefill() is None

    scheduler.enqueue(_req())
    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_finished_running_requests_do_not_count_as_served(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.running_batch = _busy_batch(running=1, finished=3)
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_full_prefill_budget_admits_despite_fresh_arrivals(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.enqueue(_req(tokens=2048))
    scheduler.enqueue(_req(tokens=2048))

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_full_running_slots_admit_despite_fresh_arrivals(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.running_batch = _busy_batch(running=62)
    scheduler.enqueue(_req())
    scheduler.enqueue(_req())

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH


def test_chunked_prefill_size_caps_the_budget(upstream) -> None:
    scheduler = _StubScheduler()
    scheduler.chunked_prefill_size = 512
    scheduler.enqueue(_req(tokens=300))
    scheduler.enqueue(_req(tokens=300))

    assert scheduler.get_new_batch_prefill() is _UPSTREAM_BATCH
