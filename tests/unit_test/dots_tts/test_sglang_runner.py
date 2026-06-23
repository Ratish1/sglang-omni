# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner
from sglang_omni.models.dots_tts.request_builders import DotsTTSSGLangRequestData


def make_runner() -> DotsTTSModelRunner:
    runner = DotsTTSModelRunner.__new__(DotsTTSModelRunner)
    runner.device = torch.device("cpu")
    return runner


def test_post_prefill_stores_latest_hidden_state() -> None:
    runner = make_runner()
    data = DotsTTSSGLangRequestData()
    sched_req = SimpleNamespace(data=data)
    hidden = torch.ones(1, 3, 2048)
    result = SimpleNamespace(hidden_states=hidden, next_token_ids=None)

    runner._capture_hidden_states(result, [sched_req], packed_prefill=True)

    assert torch.equal(data.latest_hidden_state, hidden[:, -1:, :])


def test_post_prefill_reads_hidden_state_from_logits_output() -> None:
    runner = make_runner()
    data = DotsTTSSGLangRequestData()
    sched_req = SimpleNamespace(data=data)
    hidden = torch.ones(1, 3, 2048)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden),
        next_token_ids=None,
    )

    runner._capture_hidden_states(result, [sched_req], packed_prefill=True)

    assert torch.equal(data.latest_hidden_state, hidden[:, -1:, :])


def test_post_prefill_reshapes_2d_hidden_state() -> None:
    runner = make_runner()
    data = DotsTTSSGLangRequestData()
    data.req = SimpleNamespace(extend_input_len=3)
    sched_req = SimpleNamespace(data=data)
    hidden = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden),
        next_token_ids=None,
    )

    runner._capture_hidden_states(result, [sched_req], packed_prefill=True)

    assert torch.equal(data.latest_hidden_state, hidden[2:3].unsqueeze(0))


def test_post_prefill_slices_packed_hidden_by_extend_lengths() -> None:
    runner = make_runner()
    first = DotsTTSSGLangRequestData()
    first.req = SimpleNamespace(extend_input_len=2)
    second = DotsTTSSGLangRequestData()
    second.req = SimpleNamespace(extend_input_len=3)
    requests = [SimpleNamespace(data=first), SimpleNamespace(data=second)]
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden),
        next_token_ids=None,
    )

    runner._capture_hidden_states(result, requests, packed_prefill=True)

    assert torch.equal(first.latest_hidden_state, hidden[1:2].unsqueeze(0))
    assert torch.equal(second.latest_hidden_state, hidden[4:5].unsqueeze(0))


def test_post_prefill_reads_2d_batch_hidden_state() -> None:
    runner = make_runner()
    data = DotsTTSSGLangRequestData()
    data.req = SimpleNamespace(extend_input_len=23)
    sched_req = SimpleNamespace(data=data)
    hidden = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden),
        next_token_ids=None,
    )

    runner._capture_hidden_states(result, [sched_req], packed_prefill=True)

    assert torch.equal(data.latest_hidden_state, hidden.unsqueeze(0))


def test_post_decode_reads_2d_hidden_state_by_batch_row() -> None:
    runner = make_runner()
    first = DotsTTSSGLangRequestData()
    first.req = SimpleNamespace(extend_input_len=2)
    second = DotsTTSSGLangRequestData()
    second.req = SimpleNamespace(extend_input_len=3)
    requests = [SimpleNamespace(data=first), SimpleNamespace(data=second)]
    hidden = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden),
        next_token_ids=None,
    )

    runner._capture_hidden_states(result, requests, packed_prefill=False)

    assert torch.equal(first.latest_hidden_state, hidden[0:1].unsqueeze(0))
    assert torch.equal(second.latest_hidden_state, hidden[1:2].unsqueeze(0))


def test_before_decode_writes_pending_feedback_embedding() -> None:
    runner = make_runner()
    data = DotsTTSSGLangRequestData()
    data.decode_input_embeds.append(torch.full((1, 1, 4), 3.0))
    sched_req = SimpleNamespace(data=data)
    forward_batch = SimpleNamespace(input_ids=torch.tensor([0]))

    runner.before_decode(forward_batch, None, [sched_req])

    assert torch.equal(forward_batch.input_embeds, torch.full((1, 4), 3.0))
    assert data.decode_input_embeds == []


def test_before_prefill_does_not_create_latent_stepper() -> None:
    runner = make_runner()

    class FakeModel:
        def create_latent_stepper(self, data):
            raise AssertionError("latent_stepper fallback should not be used")

    runner.model = FakeModel()
    data = DotsTTSSGLangRequestData()
    sched_req = SimpleNamespace(data=data)

    runner.before_prefill(None, None, [sched_req])

    assert not hasattr(data, "latent_stepper")


def test_post_decode_generates_latent_patch_from_hidden_state() -> None:
    runner = make_runner()

    class FakeModel:
        def step_audio_latent(self, data, hidden_state):
            assert hidden_state.shape == (1, 1, 2048)
            assert data.fm_state == {"history": []}
            assert data.generation_kwargs == {"num_steps": 2}
            return SimpleNamespace(
                latent_patch=torch.ones(1, 4, 128),
                feedback_embedding=torch.full((1, 1, 128), 2.0),
                eos_score=torch.tensor([0.1]),
            )

    runner.model = FakeModel()
    data = DotsTTSSGLangRequestData(control_token_id=777)
    data.latest_hidden_state = torch.ones(1, 1, 2048)
    data.fm_state = {"history": []}
    data.generation_kwargs = {"num_steps": 2}
    data.position = 3
    sched_req = SimpleNamespace(data=data)
    result = SimpleNamespace(hidden_states=None, next_token_ids=None)

    runner.post_decode(result, None, None, [sched_req])

    assert result.next_token_ids.tolist() == [777]
    assert data.latest_latent_patch.shape == (1, 4, 128)
    assert len(data.decode_input_embeds) == 1
    assert data.decode_input_embeds[0].shape == (1, 1, 128)
    assert torch.allclose(data.eos_score, torch.tensor([0.1]))
    assert data.position == 4


def test_post_decode_generates_latents_for_multiple_requests() -> None:
    runner = make_runner()

    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def step_audio_latent(self, data, hidden_state):
            self.calls.append((data.control_token_id, hidden_state.clone()))
            token = float(data.control_token_id)
            return SimpleNamespace(
                latent_patch=torch.full((1, 4, 2), token),
                feedback_embedding=torch.full((1, 1, 3), token / 10.0),
                eos_score=torch.tensor([0.0]),
            )

    fake_model = FakeModel()
    runner.model = fake_model

    first = DotsTTSSGLangRequestData(control_token_id=101)
    first.latest_hidden_state = torch.ones(1, 1, 4)
    first.fm_state = {"rid": "first"}
    second = DotsTTSSGLangRequestData(control_token_id=202)
    second.latest_hidden_state = torch.full((1, 1, 4), 2.0)
    second.fm_state = {"rid": "second"}

    result = SimpleNamespace(hidden_states=None, next_token_ids=None)

    runner.post_decode(
        result,
        None,
        None,
        [
            SimpleNamespace(data=first),
            SimpleNamespace(data=second),
        ],
    )

    assert result.next_token_ids.tolist() == [101, 202]
    assert len(fake_model.calls) == 2
    assert torch.equal(first.latest_latent_patch, torch.full((1, 4, 2), 101.0))
    assert torch.equal(second.latest_latent_patch, torch.full((1, 4, 2), 202.0))
    assert torch.equal(first.decode_input_embeds[0], torch.full((1, 1, 3), 10.1))
    assert torch.equal(second.decode_input_embeds[0], torch.full((1, 1, 3), 20.2))
    assert len(first.latent_patches) == 1
    assert len(second.latent_patches) == 1


def test_post_decode_does_not_use_latent_stepper_fallback() -> None:
    runner = make_runner()

    data = DotsTTSSGLangRequestData(
        control_token_id=777,
        latest_hidden_state=torch.ones(1, 1, 4),
    )
    data.req = SimpleNamespace(finished_reason=None)
    sched_req = SimpleNamespace(data=data)
    result = SimpleNamespace(next_token_ids=None)

    runner.post_decode(result, None, None, [sched_req])

    assert result.next_token_ids.tolist() == [777]
    assert data.latest_latent_patch is None
    assert data.latent_patches == []
