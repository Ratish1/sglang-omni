# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner


def test_dots_abort_callback_clears_flow_state() -> None:
    flow_state = object()
    released = []
    request_data = SimpleNamespace(
        pending_feedback_queue=deque([object()]),
        flow_state=flow_state,
    )
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        flow=SimpleNamespace(release_request=released.append)
    )
    runner._request_data = {"req-1": request_data}

    runner.reset_request("req-1")
    runner.reset_request("req-1")

    assert runner._request_data == {}
    assert not request_data.pending_feedback_queue
    assert request_data.flow_state is None
    assert released == [flow_state]


def test_dots_finish_callback_releases_flow_state_once() -> None:
    flow_state = object()
    released = []
    request_data = SimpleNamespace(
        pending_feedback_queue=deque([object()]),
        flow_state=flow_state,
    )
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        flow=SimpleNamespace(release_request=released.append)
    )
    runner._request_data = {"req-1": request_data}

    runner.on_request_finished("req-1", request_data)
    runner.reset_request("req-1")

    assert request_data.flow_state is None
    assert released == [flow_state]


def test_dots_post_prefill_skips_prefill_only_batch() -> None:
    runner = object.__new__(DotsTTSModelRunner)

    runner.post_prefill(
        result=object(),
        forward_batch=object(),
        schedule_batch=SimpleNamespace(is_prefill_only=True),
        requests=[object()],
    )


def test_dots_prefill_batches_request_embeddings_in_scheduler_order() -> None:
    old_flow_state = object()
    released = []

    class _Flow:
        def new_request(self, *, speaker_scale: float, **_kwargs):
            prompt = torch.full((1, 1, 4), speaker_scale)
            return object(), prompt

        release_request = released.append

    model = SimpleNamespace(
        flow=_Flow(),
        get_input_embeddings=lambda: nn.Embedding.from_pretrained(
            torch.arange(40, dtype=torch.float32).reshape(10, 4)
        ),
    )
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = model
    runner._request_data = {}

    def _request(request_id: str, speaker_scale: float):
        data = SimpleNamespace(
            flow_state=old_flow_state if request_id == "a" else None,
            generation_schedule=torch.tensor([[1, 2, 3]]),
            span_positions=torch.tensor([1, 2]),
            prompt_span_positions=torch.tensor([1]),
            prefill_end=3,
            req=SimpleNamespace(
                prefix_indices=[], extend_range=SimpleNamespace(length=3)
            ),
            state=SimpleNamespace(
                prompt_latents=torch.zeros(1, 4, 2),
                speaker_embedding=torch.zeros(1, 8),
                speaker_scale=speaker_scale,
                seed=42,
            ),
        )
        return SimpleNamespace(request_id=request_id, data=data)

    requests = [_request("a", 11.0), _request("b", 22.0)]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2, 3, 1, 2, 3]), input_embeds=None
    )

    runner.before_prefill(forward_batch, object(), requests)

    assert forward_batch.input_embeds.shape == (6, 4)
    torch.testing.assert_close(forward_batch.input_embeds[1], torch.full((4,), 11.0))
    torch.testing.assert_close(forward_batch.input_embeds[4], torch.full((4,), 22.0))
    assert released == [old_flow_state]


def test_dots_prefill_failure_releases_materialized_slots() -> None:
    flow_state = object()
    released = []

    class _Flow:
        calls = 0

        def new_request(self, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("materialize failed")
            return flow_state, None

        release_request = released.append

    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        flow=_Flow(),
        get_input_embeddings=lambda: nn.Embedding(10, 4),
    )
    runner._request_data = {}

    def _request(request_id: str):
        data = SimpleNamespace(
            flow_state=None,
            generation_schedule=torch.tensor([[1, 2, 3]]),
            span_positions=torch.tensor([1]),
            prompt_span_positions=None,
            prefill_end=3,
            pending_feedback_queue=deque(),
            req=SimpleNamespace(
                prefix_indices=[], extend_range=SimpleNamespace(length=3)
            ),
            state=SimpleNamespace(
                prompt_latents=None,
                speaker_embedding=None,
                speaker_scale=1.0,
                seed=42,
            ),
        )
        return SimpleNamespace(request_id=request_id, data=data)

    requests = [_request("a"), _request("b")]
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2, 3, 1, 2, 3]), input_embeds=None
    )

    with pytest.raises(RuntimeError, match="materialize failed"):
        runner.before_prefill(forward_batch, object(), requests)

    assert released == [flow_state]
    assert runner._request_data == {}
    assert requests[0].data.flow_state is None
