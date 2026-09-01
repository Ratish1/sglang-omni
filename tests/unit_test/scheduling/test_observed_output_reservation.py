# SPDX-License-Identifier: Apache-2.0
"""The KV reservation estimate follows finished outputs, not the cap.

SGLang's PrefillAdder reserves min(max_new_tokens, CLIP_MAX_NEW_TOKENS) times
new_token_ratio per running request. OmniScheduler records the fraction of
the clipped cap each finished request used and applies the largest recent
fraction to the tracker before every prefill admission. Tested against a stub
scheduler with the upstream admission patched to a sentinel.
"""

from __future__ import annotations

import threading
from collections import deque
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import NextBatchPlan  # noqa: E402
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS  # noqa: E402

from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

_UPSTREAM_BATCH = object()
_SGLANG_GUESS = 0.7


class _StubScheduler:
    """The attribute surface the reservation estimate and the admission touch."""

    def __init__(self, *, window: int = 4) -> None:
        self.new_token_ratio_tracker = SimpleNamespace(current=_SGLANG_GUESS)
        self._finished_output_fractions: deque[float] = deque(maxlen=window)
        self._observed_new_token_ratio = None
        self.prefill_coalesce_requests = 0
        self.chunked_req = None
        self.waiting_queue: list = []
        self._request_admission_lock = threading.RLock()
        self._pending_request_builds: dict = {}
        self._pending_request_admissions: dict = {}
        self._backlogged_request_build_payloads: list = []

    def finish(self, output_len: int, cap: int | None) -> None:
        req = SimpleNamespace(
            output_ids=list(range(output_len)),
            sampling_params=SimpleNamespace(max_new_tokens=cap),
        )
        OmniScheduler._record_finished_output(self, req)

    def admit(self):
        plan = OmniScheduler.get_new_batch_prefill(self, running_batch=None)
        return plan.batch_to_run


@pytest.fixture()
def upstream():
    with mock.patch.object(
        omni_scheduler._Upstream,
        "get_new_batch_prefill",
        return_value=NextBatchPlan(batch_to_run=_UPSTREAM_BATCH, running_batch=None),
    ) as patched:
        yield patched


def test_admission_keeps_sglang_guess_until_a_request_finishes(upstream) -> None:
    sched = _StubScheduler()
    assert sched.admit() is _UPSTREAM_BATCH
    assert sched.new_token_ratio_tracker.current == _SGLANG_GUESS


def test_admission_reserves_the_largest_recent_fraction(upstream) -> None:
    sched = _StubScheduler()
    sched.finish(40, 4096)
    sched.finish(93, 4096)
    sched.finish(60, 4096)
    sched.admit()
    assert sched.new_token_ratio_tracker.current == pytest.approx(93 / 4096)


def test_fraction_uses_the_clipped_cap_like_the_adder(upstream) -> None:
    sched = _StubScheduler()
    sched.finish(512, 4 * CLIP_MAX_NEW_TOKENS)
    sched.admit()
    assert sched.new_token_ratio_tracker.current == pytest.approx(
        512 / CLIP_MAX_NEW_TOKENS
    )


def test_old_outputs_leave_the_window(upstream) -> None:
    sched = _StubScheduler(window=2)
    sched.finish(900, 4096)
    sched.finish(40, 4096)
    sched.finish(50, 4096)
    sched.admit()
    assert sched.new_token_ratio_tracker.current == pytest.approx(50 / 4096)


def test_sglang_writes_between_admissions_do_not_survive(upstream) -> None:
    sched = _StubScheduler()
    sched.finish(93, 4096)
    sched.admit()
    sched.new_token_ratio_tracker.current = 0.02
    sched.admit()
    assert sched.new_token_ratio_tracker.current == pytest.approx(93 / 4096)


def test_requests_without_a_cap_are_not_observed(upstream) -> None:
    sched = _StubScheduler()
    sched.finish(93, None)
    sched.finish(93, 0)
    sched.admit()
    assert sched.new_token_ratio_tracker.current == _SGLANG_GUESS
