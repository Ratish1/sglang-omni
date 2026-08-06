# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner


def test_dots_abort_callback_clears_flow_state() -> None:
    request_data = SimpleNamespace(
        pending_feedback_queue=deque([object()]),
        flow_state=object(),
    )
    runner = object.__new__(DotsTTSModelRunner)
    runner._request_data = {"req-1": request_data}

    runner.reset_request("req-1")
    runner.reset_request("req-1")

    assert runner._request_data == {}
    assert not request_data.pending_feedback_queue
    assert request_data.flow_state is None


def test_dots_post_prefill_skips_prefill_only_batch() -> None:
    runner = object.__new__(DotsTTSModelRunner)

    runner.post_prefill(
        result=object(),
        forward_batch=object(),
        schedule_batch=SimpleNamespace(is_prefill_only=True),
        requests=[object()],
    )
