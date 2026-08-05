# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch
from sglang.srt.model_executor.forward_context import has_forward_context

from sglang_omni.models.dots_tts.model_runner import DotsTTSModelRunner
from sglang_omni.models.dots_tts.request_builders import DotsTTSSGLangRequestData
from sglang_omni.models.dots_tts.sglang_model import DotsTTSLatentStepOutput


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
    data.req = SimpleNamespace(extend_range=SimpleNamespace(length=3))
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
    first.req = SimpleNamespace(extend_range=SimpleNamespace(length=2))
    second = DotsTTSSGLangRequestData()
    second.req = SimpleNamespace(extend_range=SimpleNamespace(length=3))
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
    data.req = SimpleNamespace(extend_range=SimpleNamespace(length=23))
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
    first.req = SimpleNamespace(extend_range=SimpleNamespace(length=2))
    second = DotsTTSSGLangRequestData()
    second.req = SimpleNamespace(extend_range=SimpleNamespace(length=3))
    requests = [SimpleNamespace(data=first), SimpleNamespace(data=second)]
    hidden = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=hidden),
        next_token_ids=None,
    )

    runner._capture_hidden_states(result, requests, packed_prefill=False)

    assert torch.equal(first.latest_hidden_state, hidden[0:1].unsqueeze(0))
    assert torch.equal(second.latest_hidden_state, hidden[1:2].unsqueeze(0))


def test_prefill_input_embeds_follow_live_extend_range() -> None:
    runner = make_runner()
    data = DotsTTSSGLangRequestData(
        prefill_input_embeds=torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)
    )
    data.req = SimpleNamespace(
        extend_range=SimpleNamespace(length=2),
        prefix_indices=[0, 1, 2],
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([10, 11]))

    actual = runner._build_prefill_input_embeds(
        forward_batch, [SimpleNamespace(data=data)]
    )

    assert torch.equal(actual, data.prefill_input_embeds[3:5])


def test_custom_prefill_forward_enters_attention_context() -> None:
    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(()))

        def forward(self, **kwargs):
            assert has_forward_context()
            return SimpleNamespace(hidden_states=torch.ones(1, 1, 4))

    runner = make_runner()
    runner.model = FakeModel()
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(init_forward_metadata=lambda fb: None)
        )
    )
    data = DotsTTSSGLangRequestData(prefill_input_embeds=torch.ones(2, 4))
    data.req = SimpleNamespace(
        extend_range=SimpleNamespace(length=2), prefix_indices=[]
    )
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([0, 1]),
        positions=torch.tensor([0, 1]),
        mrope_positions=None,
    )

    runner.custom_prefill_forward(forward_batch, None, [SimpleNamespace(data=data)])


def test_before_decode_writes_pending_feedback_embedding() -> None:
    runner = make_runner()
    data = DotsTTSSGLangRequestData()
    data.decode_input_embeds.append(torch.full((1, 1, 4), 3.0))
    sched_req = SimpleNamespace(data=data)
    forward_batch = SimpleNamespace(input_ids=torch.tensor([0]))

    runner.before_decode(forward_batch, None, [sched_req])

    assert torch.equal(forward_batch.input_embeds, torch.full((1, 4), 3.0))
    assert data.decode_input_embeds == []


def test_runner_uses_model_latent_decode_step() -> None:
    calls = []

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.param = torch.nn.Parameter(torch.ones(()))

        def forward(self, *args, **kwargs):
            raise AssertionError("runner should use forward_latent_decode_step")

        def forward_latent_decode_step(self, **kwargs):
            assert has_forward_context()
            calls.append(kwargs)
            batch_result = SimpleNamespace(
                next_token_ids=torch.tensor([0]),
                logits_output=SimpleNamespace(hidden_states=None),
                can_run_cuda_graph=False,
            )
            return DotsTTSLatentStepOutput(
                batch_result=batch_result,
                next_token_ids=batch_result.next_token_ids,
                latent_patches=[torch.ones((1, 4, 128))],
                feedback_embeddings=[torch.ones((1, 1, 4))],
                eos_scores=[torch.zeros(1)],
                finished=[False],
                hidden_states=[torch.ones((1, 1, 4))],
            )

    runner = make_runner()
    runner.model = FakeModel()
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend=SimpleNamespace(init_forward_metadata=lambda fb: None)
        )
    )
    data = DotsTTSSGLangRequestData(control_token_id=0)
    data.fm_state = object()
    sched_req = SimpleNamespace(data=data)
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([0]),
        positions=torch.tensor([0]),
        mrope_positions=None,
        input_embeds=torch.ones((1, 4)),
    )

    result = runner.custom_decode_forward(forward_batch, None, [sched_req])

    assert calls
    assert calls[0]["requests"] == [sched_req]
    assert result.next_token_ids.tolist() == [0]


def test_post_decode_generates_latent_patch_from_hidden_state() -> None:
    runner = make_runner()

    class FakeModel:
        def decode_audio_batch(self, batch):
            assert batch.active_indices == [0]
            assert batch.hidden_states[0].shape == (1, 1, 2048)
            assert batch.requests[0].data.fm_state == {"history": []}
            return DotsTTSLatentStepOutput(
                batch_result=None,
                next_token_ids=torch.tensor([777]),
                latent_patches=[torch.ones(1, 4, 128)],
                feedback_embeddings=[torch.full((1, 1, 128), 2.0)],
                eos_scores=[torch.tensor([0.1])],
                finished=[False],
                hidden_states=batch.hidden_states,
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

        def decode_audio_batch(self, batch):
            self.calls.append(batch)
            return DotsTTSLatentStepOutput(
                batch_result=None,
                next_token_ids=torch.tensor([101, 202]),
                latent_patches=[
                    torch.full((1, 4, 2), 101.0),
                    torch.full((1, 4, 2), 202.0),
                ],
                feedback_embeddings=[
                    torch.full((1, 1, 3), 10.1),
                    torch.full((1, 1, 3), 20.2),
                ],
                eos_scores=[torch.tensor([0.0]), torch.tensor([0.0])],
                finished=[False, False],
                hidden_states=batch.hidden_states,
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
    assert len(fake_model.calls) == 1
    assert fake_model.calls[0].active_indices == [0, 1]
    assert torch.equal(first.latest_latent_patch, torch.full((1, 4, 2), 101.0))
    assert torch.equal(second.latest_latent_patch, torch.full((1, 4, 2), 202.0))
    assert torch.equal(first.decode_input_embeds[0], torch.full((1, 1, 3), 10.1))
    assert torch.equal(second.decode_input_embeds[0], torch.full((1, 1, 3), 20.2))
    assert len(first.latent_patches) == 1
    assert len(second.latent_patches) == 1


def test_post_decode_advances_control_token_when_no_active_requests() -> None:
    runner = make_runner()

    class FakeModel:
        def decode_audio_batch(self, batch):
            assert batch.active_indices == []
            return DotsTTSLatentStepOutput(
                batch_result=None,
                next_token_ids=torch.tensor([777]),
                latent_patches=[None],
                feedback_embeddings=[None],
                eos_scores=[None],
                finished=[False],
                hidden_states=batch.hidden_states,
            )

    runner.model = FakeModel()

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
