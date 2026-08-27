# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.penaltylib import (
    BatchedMinNewTokensPenalizer,
    BatchedPenalizerOrchestrator,
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EOS_ID,
    VOCAB_SIZE,
    FunCosyVoice3SGLangModel,
)


def test_cosyvoice3_runner_collects_speech_tokens_and_skips_eos() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    requests = [
        SimpleNamespace(data=SimpleNamespace(output_codes=[])),
        SimpleNamespace(data=SimpleNamespace(output_codes=[])),
    ]
    result = SimpleNamespace(next_token_ids=torch.tensor([[EOS_ID], [13]]))

    runner._collect_tokens(result, None, None, requests)

    assert requests[0].data.output_codes == []
    assert [code.item() for code in requests[1].data.output_codes] == [13]
    assert requests[1].data.output_codes[0].dtype == torch.long


def test_cosyvoice3_runner_skips_all_control_tokens() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    requests = [SimpleNamespace(data=SimpleNamespace(output_codes=[]))]

    runner._collect_tokens(
        SimpleNamespace(next_token_ids=torch.tensor([VOCAB_SIZE + 3])),
        None,
        None,
        requests,
    )

    assert requests[0].data.output_codes == []


def test_cosyvoice3_runner_samples_before_prefill_and_decode_collection() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)

    assert runner.sample_before_post_prefill(None, None, []) is True
    assert runner.sample_before_post_decode(None, None, []) is True


def test_cosyvoice3_reprefill_restores_sampling_penalty_state() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    tokenizer = SimpleNamespace(additional_stop_token_ids=set(), eos_token_id=None)
    retained_req = SimpleNamespace(
        output_ids=[2, 5, 2, 7],
        sampling_params=SimpleNamespace(
            repetition_penalty=1.1,
            min_new_tokens=3,
            stop_token_ids={6},
        ),
        tokenizer=tokenizer,
    )
    fresh_req = SimpleNamespace(
        output_ids=[],
        sampling_params=SimpleNamespace(
            repetition_penalty=1.0,
            min_new_tokens=3,
            stop_token_ids={6},
        ),
        tokenizer=tokenizer,
    )
    reqs = [retained_req, fresh_req]

    class _ScheduleBatch:
        pass

    schedule_batch = _ScheduleBatch()
    schedule_batch.reqs = reqs
    schedule_batch.device = "cpu"
    schedule_batch.forward_mode = SimpleNamespace(is_extend=lambda: True)
    orchestrator = BatchedPenalizerOrchestrator(
        vocab_size=8,
        batch=schedule_batch,
        penalizers={BatchedRepetitionPenalizer, BatchedMinNewTokensPenalizer},
    )
    schedule_batch.sampling_info = SamplingBatchInfo(
        temperatures=torch.ones(2, 1),
        top_ps=torch.ones(2),
        top_ks=torch.full((2,), 8, dtype=torch.int32),
        min_ps=torch.zeros(2),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=False,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=8,
        penalizer_orchestrator=orchestrator,
        device="cpu",
    )
    repetition_penalizer = orchestrator.penalizers[BatchedRepetitionPenalizer]
    min_new_tokens_penalizer = orchestrator.penalizers[BatchedMinNewTokensPenalizer]

    class _ExecutionBridge:
        @contextlib.contextmanager
        def forward_context(self, batch, *, isolate_sampling):
            assert batch is schedule_batch
            assert isolate_sampling is True
            expected_scaling = torch.ones(2, 8)
            expected_scaling[0, [2, 5, 7]] = 1.1
            assert torch.equal(
                repetition_penalizer.get_scaling_penalties(), expected_scaling
            )
            assert min_new_tokens_penalizer.len_output_tokens.tolist() == [[4], [0]]
            scheduler_sampling_info = batch.sampling_info
            batch.sampling_info = scheduler_sampling_info.copy_for_forward()
            try:
                yield
            finally:
                batch.sampling_info = scheduler_sampling_info

    runner._execution_bridge = _ExecutionBridge()
    requests = [
        SimpleNamespace(data=SimpleNamespace(req=req))
        for req in (retained_req, fresh_req)
    ]
    logits = torch.tensor(
        [
            [0.0, 1.0, 10.0, 3.0, 4.0, -10.0, 6.0, 7.0],
            [0.0, 1.0, 10.0, 3.0, 4.0, -10.0, 6.0, 7.0],
        ]
    )

    with runner._execution_context(schedule_batch, isolate_sampling=True):
        runner._apply_repetition_penalty(
            SimpleNamespace(next_token_logits=logits), requests
        )
        schedule_batch.sampling_info.apply_logits_bias(logits)

    assert logits[0, 2].item() == pytest.approx(10.0 / 1.1)
    assert logits[0, 5].item() == pytest.approx(-10.0 * 1.1)
    assert logits[0, 1].item() == 1.0
    assert logits[0, 6].item() == 6.0
    assert torch.isneginf(logits[1, 6])


def test_cosyvoice3_load_weights_maps_custom_and_backbone_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the loader path, not only the standalone key mapper."""
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    speech_embedding = torch.nn.Parameter(torch.zeros(2, 3))
    decoder = torch.nn.Parameter(torch.zeros(2, 3))
    model._cached_params_dict = {
        "speech_embedding.weight": speech_embedding,
        "llm_decoder.weight": decoder,
    }
    forwarded = []
    monkeypatch.setattr(
        "sglang.srt.models.qwen2.Qwen2ForCausalLM.load_weights",
        lambda _self, weights: forwarded.extend(weights),
    )

    speech_value = torch.ones(2, 3)
    decoder_value = torch.full((2, 3), 2.0)
    model.load_weights(
        [
            ("speech_embedding.weight", speech_value),
            ("llm_decoder.weight", decoder_value),
            ("llm.model.lm_head.weight", torch.ones(2, 3)),
            ("llm.model.model.layers.0.weight", torch.full((3, 3), 3.0)),
        ]
    )

    assert torch.equal(speech_embedding, speech_value)
    assert torch.equal(decoder, decoder_value)
    assert len(forwarded) == 1
    assert forwarded[0][0] == "model.layers.0.weight"
    assert torch.equal(forwarded[0][1], torch.full((3, 3), 3.0))


def test_cosyvoice3_runner_builds_prefill_embedding_slice_after_prefix() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.model = torch.nn.Linear(3, 3, bias=False)
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                req=SimpleNamespace(
                    extend_range=SimpleNamespace(length=2),
                    prefix_indices=[99],
                    output_ids=[],
                ),
                prompt_input_embeds=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            )
        )
    ]
    forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))

    result = runner._build_prefill_input_embeds(forward_batch, requests)

    assert torch.equal(
        result, torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32)
    )


def test_cosyvoice3_runner_reprefill_replays_generated_token_embeddings() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.model = torch.nn.Module()
    runner.model.speech_embedding = torch.nn.Embedding(8, 4)
    with torch.no_grad():
        runner.model.speech_embedding.weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )
    prompt = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    req = Req(
        rid="req",
        origin_input_text="",
        origin_input_ids=array("q", [0, 1, 2]),
        sampling_params=SamplingParams(max_new_tokens=8),
        vocab_size=8,
    )
    req.output_ids.extend([5, 7])
    req.reset_for_retract()
    req.prefix_indices = [0, 1]
    req.extend_range = SimpleNamespace(length=3)
    request = SimpleNamespace(
        data=SimpleNamespace(
            req=req,
            prompt_input_embeds=prompt,
        )
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(3, dtype=torch.long))

    result = runner._build_prefill_input_embeds(forward_batch, [request])

    expected = torch.cat(
        [prompt[2:], runner.model.speech_embedding(torch.tensor([5, 7]))],
        dim=0,
    )
    assert torch.equal(result, expected)
