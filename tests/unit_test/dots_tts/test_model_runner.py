# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.dots_tts.flow_head import (
    DotsFlowState,
    DotsFlowStep,
    DotsTTSFlowHead,
)
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


def test_single_stream_decode_preserves_batch_and_sequence_axes() -> None:
    flow = object.__new__(DotsTTSFlowHead)
    nn.Module.__init__(flow)
    flow._tail = None
    observed: list[tuple[str, tuple[int, ...]]] = []

    def _append_hidden(_state, hidden_states):
        observed.append(("append", tuple(hidden_states.shape)))

    def _decode_next(_state, *, hidden_states, **_kwargs):
        observed.append(("decode", tuple(hidden_states.shape)))
        return DotsFlowStep(
            latent_patch=torch.zeros(1, 4, 2),
            feedback_embedding=torch.zeros(8),
            finished=False,
            emit=True,
        )

    flow.append_hidden = _append_hidden
    flow.decode_next = _decode_next
    state = DotsFlowState(
        fm_sequence=torch.empty(0),
        fm_cfg_sequence=torch.empty(0),
        fm_null_g_cond=torch.empty(0),
        fm_capacity=0,
        g_cond=None,
    )

    [step] = flow.decode_batch(
        [state],
        hidden_states=torch.zeros(1, 8),
        num_steps=[4],
        ode_methods=["euler"],
        guidance_scales=[1.2],
        eos_thresholds=[0.8],
        append_hidden=True,
    )

    assert step.emit is True
    assert observed == [("append", (1, 1, 8)), ("decode", (1, 1, 8))]


def test_single_stream_rng_is_request_local_under_interleaving() -> None:
    flow = object.__new__(DotsTTSFlowHead)
    nn.Module.__init__(flow)

    def _state() -> DotsFlowState:
        return DotsFlowState(
            fm_sequence=torch.empty(0),
            fm_cfg_sequence=torch.empty(0),
            fm_null_g_cond=torch.empty(0),
            fm_capacity=0,
            g_cond=None,
            rng_state=flow._new_rng_state(device=torch.device("cpu"), seed=1234),
        )

    first = _state()
    second = _state()
    global_rng_before = torch.get_rng_state().clone()
    with flow._request_rng(first):
        first_draw_1 = torch.randn(16)
    with flow._request_rng(second):
        second_draw_1 = torch.randn(16)
    with flow._request_rng(first):
        first_draw_2 = torch.randn(16)
    with flow._request_rng(second):
        second_draw_2 = torch.randn(16)

    torch.testing.assert_close(first_draw_1, second_draw_1, rtol=0, atol=0)
    torch.testing.assert_close(first_draw_2, second_draw_2, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.get_rng_state(),
        global_rng_before,
        rtol=0,
        atol=0,
    )


def test_dots_prefill_batches_request_embeddings_in_scheduler_order() -> None:
    old_flow_state = object()
    released = []

    class _Flow:
        def new_request(self, *, speaker_scale: float, **_kwargs):
            prompt = torch.full((1, 1, 4), speaker_scale)
            return object(), prompt

        release_request = released.append

        @staticmethod
        def export_request_rng_state(_state):
            return torch.tensor([1], dtype=torch.uint8)

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
            flow_rng_state=None,
            feedback_embeddings=[],
            pending_feedback_queue=deque(),
            generation_schedule=torch.tensor([[1, 2, 3]]),
            span_positions=torch.tensor([1, 2]),
            prompt_span_positions=torch.tensor([1]),
            prefill_end=3,
            req=SimpleNamespace(
                prefix_indices=[],
                extend_range=SimpleNamespace(length=3),
                output_ids=[],
                is_retracted=False,
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

        @staticmethod
        def export_request_rng_state(_state):
            return torch.tensor([1], dtype=torch.uint8)

    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        flow=_Flow(),
        get_input_embeddings=lambda: nn.Embedding(10, 4),
    )
    runner._request_data = {}

    def _request(request_id: str):
        data = SimpleNamespace(
            flow_state=None,
            flow_rng_state=None,
            feedback_embeddings=[],
            generation_schedule=torch.tensor([[1, 2, 3]]),
            span_positions=torch.tensor([1]),
            prompt_span_positions=None,
            prefill_end=3,
            pending_feedback_queue=deque(),
            req=SimpleNamespace(
                prefix_indices=[],
                extend_range=SimpleNamespace(length=3),
                output_ids=[],
                is_retracted=False,
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


def test_retracted_prefill_replays_feedback_latents_and_rng_position() -> None:
    old_flow_state = object()
    new_flow_state = object()
    saved_rng_state = torch.tensor([3, 1, 4], dtype=torch.uint8)
    released = []
    observed: dict[str, object] = {}

    class _Flow:
        def export_request_rng_state(self, state):
            assert state is old_flow_state
            return saved_rng_state

        def release_request(self, state):
            released.append(state)

        def new_request(self, *, rng_state, **_kwargs):
            observed["restored_rng_state"] = rng_state
            return new_flow_state, torch.full((1, 1, 4), 50.0)

        def initialize_history(self, _state, **kwargs):
            observed["decoded_latent_patches"] = kwargs["decoded_latent_patches"]

        def decode_batch(self, _states, *, hidden_states, **_kwargs):
            observed["next_hidden"] = hidden_states
            return [
                DotsFlowStep(
                    latent_patch=torch.full((1, 2, 2), 9.0),
                    feedback_embedding=torch.full((4,), 10.0),
                    finished=False,
                    emit=True,
                )
            ]

    model = SimpleNamespace(
        flow=_Flow(),
        get_input_embeddings=lambda: nn.Embedding.from_pretrained(
            torch.arange(40, dtype=torch.float32).reshape(10, 4)
        ),
    )
    runner = object.__new__(DotsTTSModelRunner)
    runner.model = model
    runner._request_data = {}
    feedback_history = [
        torch.full((4,), 60.0),
        torch.full((4,), 70.0),
    ]
    decoded_history = [
        torch.full((1, 2, 2), 1.0),
        torch.full((1, 2, 2), 2.0),
    ]
    data = SimpleNamespace(
        flow_state=old_flow_state,
        flow_rng_state=None,
        feedback_embeddings=list(feedback_history),
        decoded_latent_patches=list(decoded_history),
        latent_patches=[decoded_history[1]],
        latest_latent_patch=None,
        pending_feedback_queue=deque([feedback_history[-1]]),
        generation_schedule=torch.tensor([[1, 2, 3, 3, 3]]),
        span_positions=torch.tensor([1, 2, 3, 4]),
        prompt_span_positions=torch.tensor([1]),
        prefill_end=3,
        control_token_id=3,
        req=SimpleNamespace(
            prefix_indices=[],
            extend_range=SimpleNamespace(length=5),
            output_ids=[3, 3],
            is_retracted=False,
            finished_reason=None,
        ),
        state=SimpleNamespace(
            prompt_latents=torch.zeros(1, 2, 2),
            speaker_embedding=torch.zeros(1, 8),
            speaker_scale=1.0,
            seed=42,
            audio_span_token_ids=[3],
            num_steps=4,
            ode_method="euler",
            guidance_scale=1.2,
            eos_threshold=0.8,
        ),
    )
    request = SimpleNamespace(request_id="rid", data=data)
    forward_batch = SimpleNamespace(
        input_ids=torch.arange(5),
        input_embeds=None,
    )

    runner.before_prefill(forward_batch, object(), [request])

    assert released == [old_flow_state]
    assert observed["restored_rng_state"] is saved_rng_state
    torch.testing.assert_close(
        forward_batch.input_embeds[-2:],
        torch.stack(feedback_history),
    )
    assert not data.pending_feedback_queue

    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    runner.post_prefill(
        SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden),
            next_token_ids=None,
        ),
        forward_batch,
        SimpleNamespace(is_prefill_only=False),
        [request],
    )

    replayed = observed["decoded_latent_patches"]
    assert isinstance(replayed, list)
    assert len(replayed) == len(decoded_history)
    assert all(
        actual is expected for actual, expected in zip(replayed, decoded_history)
    )
    torch.testing.assert_close(observed["next_hidden"], hidden[-1:])
    assert len(data.feedback_embeddings) == 3
    assert len(data.decoded_latent_patches) == 3
    assert len(data.latent_patches) == 2


def test_new_prefill_reclaims_slots_held_by_waiting_retractions() -> None:
    events = []
    stale_flow_state = object()

    class _Flow:
        def export_request_rng_state(self, state):
            assert state is stale_flow_state
            events.append("snapshot-stale")
            return torch.tensor([8], dtype=torch.uint8)

        def release_request(self, state):
            assert state is stale_flow_state
            events.append("release-stale")

        def new_request(self, **_kwargs):
            events.append("allocate-new")
            return object(), None

    runner = object.__new__(DotsTTSModelRunner)
    runner.model = SimpleNamespace(
        flow=_Flow(),
        get_input_embeddings=lambda: nn.Embedding(10, 4),
    )
    stale_data = SimpleNamespace(
        flow_state=stale_flow_state,
        flow_rng_state=None,
        pending_feedback_queue=deque([torch.zeros(4)]),
        req=SimpleNamespace(is_retracted=True),
    )
    runner._request_data = {"stale": stale_data}
    new_data = SimpleNamespace(
        flow_state=None,
        flow_rng_state=None,
        feedback_embeddings=[],
        generation_schedule=torch.tensor([[1, 2, 3]]),
        span_positions=torch.tensor([1]),
        prompt_span_positions=None,
        prefill_end=3,
        pending_feedback_queue=deque(),
        req=SimpleNamespace(
            prefix_indices=[],
            extend_range=SimpleNamespace(length=3),
            output_ids=[],
            is_retracted=False,
        ),
        state=SimpleNamespace(
            prompt_latents=None,
            speaker_embedding=None,
            speaker_scale=1.0,
            seed=None,
        ),
    )

    runner.before_prefill(
        SimpleNamespace(input_ids=torch.arange(3), input_embeds=None),
        object(),
        [SimpleNamespace(request_id="new", data=new_data)],
    )

    assert events == ["snapshot-stale", "release-stale", "allocate-new"]
    assert stale_data.flow_state is None
    assert not stale_data.pending_feedback_queue
