# SPDX-License-Identifier: Apache-2.0
"""The KV reservation estimate follows finished outputs, not the cap.

SGLang's PrefillAdder reserves min(max_new_tokens, CLIP_MAX_NEW_TOKENS) times
new_token_ratio per running request. RecentFinishedOutputTracker keeps the
fraction of the clipped cap each finished request used for one cohort of
max_running_requests, and OmniScheduler pushes the largest into the tracker
before every prefill admission. The admission side is tested against a stub
scheduler with the upstream admission patched to a sentinel.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("sglang")

from sglang.srt.managers.schedule_batch import NextBatchPlan  # noqa: E402
from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS  # noqa: E402

from sglang_omni.scheduling import omni_scheduler  # noqa: E402
from sglang_omni.scheduling.finished_output_tracker import (  # noqa: E402
    RecentFinishedOutputTracker,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler  # noqa: E402

_UPSTREAM_BATCH = object()
_SGLANG_GUESS = 0.7


def _finished(output_len: int, cap: int | None):
    return SimpleNamespace(
        output_ids=list(range(output_len)),
        sampling_params=SimpleNamespace(max_new_tokens=cap),
    )


def test_no_observation_until_a_request_finishes() -> None:
    assert RecentFinishedOutputTracker(window_size=4).max_fraction is None


def test_tracks_the_largest_recent_fraction() -> None:
    tracker = RecentFinishedOutputTracker(window_size=4)
    tracker.observe_finished(_finished(40, 4096))
    tracker.observe_finished(_finished(93, 4096))
    tracker.observe_finished(_finished(60, 4096))
    assert tracker.max_fraction == pytest.approx(93 / 4096)


def test_fraction_uses_the_clipped_cap_like_the_adder() -> None:
    tracker = RecentFinishedOutputTracker(window_size=4)
    tracker.observe_finished(_finished(512, 4 * CLIP_MAX_NEW_TOKENS))
    assert tracker.max_fraction == pytest.approx(512 / CLIP_MAX_NEW_TOKENS)


def test_outputs_past_the_clip_count_as_the_whole_clipped_cap() -> None:
    tracker = RecentFinishedOutputTracker(window_size=4)
    tracker.observe_finished(_finished(9000, 9001))
    assert tracker.max_fraction == 1.0


def test_outputs_older_than_one_cohort_leave_the_window() -> None:
    tracker = RecentFinishedOutputTracker(window_size=2)
    tracker.observe_finished(_finished(900, 4096))
    tracker.observe_finished(_finished(40, 4096))
    tracker.observe_finished(_finished(50, 4096))
    assert tracker.max_fraction == pytest.approx(50 / 4096)


def test_requests_without_a_cap_are_not_observed() -> None:
    tracker = RecentFinishedOutputTracker(window_size=4)
    tracker.observe_finished(_finished(93, None))
    tracker.observe_finished(_finished(93, 0))
    assert tracker.max_fraction is None


class _StubScheduler:
    """The attribute surface the admission push touches."""

    def __init__(self) -> None:
        self.new_token_ratio_tracker = SimpleNamespace(current=_SGLANG_GUESS)
        self.finished_output_tracker = RecentFinishedOutputTracker(window_size=4)
        self.prefill_coalesce_requests = 0
        self.chunked_req = None
        self.waiting_queue: list = []
        self._request_admission_lock = threading.RLock()
        self._pending_request_builds: dict = {}
        self._pending_request_admissions: dict = {}
        self._backlogged_request_build_payloads: list = []

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


def test_admission_pushes_the_observed_fraction_before_upstream_runs(
    upstream,
) -> None:
    sched = _StubScheduler()
    sched.finished_output_tracker.observe_finished(_finished(93, 4096))
    seen = []
    upstream.side_effect = lambda self, running_batch: (
        seen.append(self.new_token_ratio_tracker.current),
        NextBatchPlan(batch_to_run=_UPSTREAM_BATCH, running_batch=None),
    )[1]
    assert sched.admit() is _UPSTREAM_BATCH
    assert seen == [pytest.approx(93 / 4096)]


def test_sglang_writes_between_admissions_do_not_survive(upstream) -> None:
    sched = _StubScheduler()
    sched.finished_output_tracker.observe_finished(_finished(93, 4096))
    sched.admit()
    sched.new_token_ratio_tracker.current = 0.02
    sched.admit()
    assert sched.new_token_ratio_tracker.current == pytest.approx(93 / 4096)
