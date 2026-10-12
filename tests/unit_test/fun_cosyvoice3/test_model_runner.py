# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from array import array
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.models.fun_cosyvoice3.model_runner as model_runner_module
import sglang_omni.sampling.repetition_aware as repetition_aware_module
from sglang_omni.models.fun_cosyvoice3.model_runner import (
    FunCosyVoice3MlxSchedulerModelRunner,
    FunCosyVoice3ModelRunner,
)
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EOS_ID,
    VOCAB_SIZE,
    FunCosyVoice3SGLangModel,
)
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK


def test_cosyvoice3_runner_collects_speech_tokens_and_skips_eos() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    requests = [
        SimpleNamespace(data=CosyVoice3SGLangRequestData()),
        SimpleNamespace(data=CosyVoice3SGLangRequestData()),
    ]
    result = SimpleNamespace(next_token_ids=torch.tensor([[EOS_ID], [13]]))

    runner.collect_tokens(result, None, None, requests)

    assert requests[0].data.output_codes == []
    assert [code.item() for code in requests[1].data.output_codes] == [13]
    assert requests[1].data.output_codes[0].dtype == torch.long
    assert runner.resolve_host_token_ids(result).tolist() == [EOS_ID, 13]


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cosyvoice3_runner_reads_cuda_token_ids_through_pinned_staging() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    requests = [
        SimpleNamespace(data=CosyVoice3SGLangRequestData()),
        SimpleNamespace(data=CosyVoice3SGLangRequestData()),
    ]
    result = SimpleNamespace(
        next_token_ids=torch.tensor([EOS_ID, 13], dtype=torch.int32, device="cuda")
    )

    runner.collect_tokens(result, None, None, requests)

    host_token_ids = runner.resolve_host_token_ids(result)
    assert host_token_ids.is_pinned()
    assert host_token_ids.tolist() == [EOS_ID, 13]
    assert requests[0].data.output_codes == []
    assert [code.item() for code in requests[1].data.output_codes] == [13]


def test_cosyvoice3_runner_skips_all_control_tokens() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    requests = [SimpleNamespace(data=CosyVoice3SGLangRequestData())]

    runner.collect_tokens(
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


def test_cosyvoice3_mlx_runner_collects_exact_scheduler_rows() -> None:
    runner = object.__new__(FunCosyVoice3MlxSchedulerModelRunner)
    runner.resolve_skip_rids = set()
    requests = [
        SimpleNamespace(request_id="first", data=SimpleNamespace(output_codes=[])),
        SimpleNamespace(request_id="second", data=SimpleNamespace(output_codes=[])),
    ]
    scheduler_output = SimpleNamespace(requests=requests)

    runner.post_process_outputs(
        SimpleNamespace(next_token_ids=torch.tensor([13, VOCAB_SIZE + 3])),
        scheduler_output,
        {},
    )

    assert [code.item() for code in requests[0].data.output_codes] == [13]
    assert requests[1].data.output_codes == []

    runner.resolve_skip_rids = {"first"}
    runner.post_process_outputs(
        SimpleNamespace(next_token_ids=torch.tensor([14, 15])),
        scheduler_output,
        {},
    )
    assert [code.item() for code in requests[0].data.output_codes] == [13]
    assert [code.item() for code in requests[1].data.output_codes] == [15]

    with pytest.raises(RuntimeError, match="row count"):
        runner.post_process_outputs(
            SimpleNamespace(next_token_ids=torch.tensor([13])),
            scheduler_output,
            {},
        )


def test_cosyvoice3_mlx_lookahead_accepts_owned_history_constraints() -> None:
    runner = object.__new__(FunCosyVoice3MlxSchedulerModelRunner)
    runner.last_mlx_pending = None
    req = SimpleNamespace(
        rid="req",
        sampling_params=SimpleNamespace(
            frequency_penalty=0.0,
            presence_penalty=0.0,
            repetition_penalty=1.1,
            min_new_tokens=4,
            sampling_seed=7,
        ),
        custom_logit_processor=None,
    )

    assert runner.lookahead_eligible(SimpleNamespace(reqs=[req], has_grammar=False))
    assert not runner.lookahead_eligible(SimpleNamespace(reqs=[req], has_grammar=True))

    runner.last_mlx_pending = SimpleNamespace(
        launch=SimpleNamespace(mode="decode"),
        reqs=[SimpleNamespace(rid="another")],
    )
    assert not runner.lookahead_eligible(SimpleNamespace(reqs=[req], has_grammar=False))


def test_cosyvoice3_torch_mps_seed_avoids_float64_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from contextlib import nullcontext

    sampling_info = SimpleNamespace(sampling_seed=None, is_all_greedy=True)
    sampled_with = []

    class Runner(FunCosyVoice3ModelRunner):
        def apply_repetition_penalty(self, logits_output, requests):
            del logits_output, requests

        def apply_codec_suppress_tokens(self, logits_output, requests):
            del logits_output, requests

        def install_sampling_seeds(self, forward_batch, requests):
            del requests
            forward_batch.sampling_info.sampling_seed = torch.tensor([7])

    runner = object.__new__(Runner)
    runner.cosyvoice3_recent_tokens = {}
    runner.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            sample=lambda logits_output, forward_batch: sampled_with.append(
                forward_batch.sampling_info.sampling_seed
            )
            or torch.tensor([13])
        )
    )
    manual_seeds = []
    compiler_stances = []
    forked_devices = []
    monkeypatch.setattr(torch, "manual_seed", manual_seeds.append)
    monkeypatch.setattr(
        model_runner_module.current_platform,
        "is_float64_supported",
        lambda: False,
    )
    monkeypatch.setattr(
        torch.compiler,
        "set_stance",
        lambda stance: compiler_stances.append(stance) or nullcontext(),
    )
    monkeypatch.setattr(
        torch.random,
        "fork_rng",
        lambda *, devices, device_type: forked_devices.append((devices, device_type))
        or nullcontext(),
    )
    request = SimpleNamespace(
        request_id="req",
        data=SimpleNamespace(
            return_logprob=False,
            req=SimpleNamespace(
                origin_input_ids=[0, 0, 0],
                output_ids=[1, 2],
                sampling_params=SimpleNamespace(sampling_seed=7),
            ),
        ),
    )
    logits_output = SimpleNamespace(
        next_token_logits=SimpleNamespace(
            device=SimpleNamespace(type="mps", index=0),
        )
    )
    forward_batch = SimpleNamespace(sampling_info=sampling_info)

    token_ids = runner.sample_next_token_ids(
        logits_output,
        forward_batch,
        None,
        [request],
    )

    assert token_ids.tolist() == [13]
    assert sampled_with == [None]
    assert sampling_info.sampling_seed.tolist() == [7]
    assert manual_seeds == [(7 + 4 * 0x9E3779B1) & SAMPLING_SEED_MASK]
    assert compiler_stances == ["force_eager"]
    assert forked_devices == [([0], "mps")]


def test_cosyvoice3_torch_mps_ras_redraws_recent_speech_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.cosyvoice3_recent_tokens = {"req": [4, 7, 9]}
    sampling_info = SimpleNamespace(is_all_greedy=False)
    logits_output = SimpleNamespace(
        next_token_logits=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
        ),
        next_token_logprobs=torch.tensor([-0.1]),
    )
    sampled = []

    def fake_multinomial(probs, num_samples):
        sampled.append((probs, num_samples))
        return torch.tensor([[6]], dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)
    request = SimpleNamespace(request_id="req")

    result = runner.apply_ras_fallback(
        logits_output,
        torch.tensor([9], dtype=torch.int32),
        sampling_info,
        [request],
    )

    assert result.tolist() == [6]
    assert len(sampled) == 1
    assert sampled[0][1] == 1
    assert sampled[0][0][0, 9].item() == 0.0
    assert runner.cosyvoice3_recent_tokens["req"][-1] == 6
    assert torch.equal(
        logits_output.next_token_logprobs,
        torch.log(torch.tensor([0.7])),
    )


def test_cosyvoice3_torch_mps_ras_keeps_non_repeated_token() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.cosyvoice3_recent_tokens = {"req": [4, 7, 9]}
    sampling_info = SimpleNamespace(is_all_greedy=False)
    logits_output = SimpleNamespace(
        next_token_logits=torch.ones((1, VOCAB_SIZE), dtype=torch.float32),
        next_token_logprobs=None,
    )
    request = SimpleNamespace(request_id="req")

    result = runner.apply_ras_fallback(
        logits_output,
        torch.tensor([6], dtype=torch.int32),
        sampling_info,
        [request],
    )

    assert result.tolist() == [6]
    assert runner.cosyvoice3_recent_tokens["req"][-1] == 6


def test_cosyvoice3_torch_mps_clears_ras_history_on_finish() -> None:
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.cosyvoice3_recent_tokens = {"req": [1], "keep": [2]}

    runner.on_request_finished("req", None)

    assert runner.cosyvoice3_recent_tokens == {"keep": [2]}


def collect_ras_step(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_ids: list[list[int]],
    rl_on_policy_target: str | None,
    return_original_logprob: bool,
) -> tuple[SimpleNamespace, list[SimpleNamespace], list[torch.Tensor], list[list[int]]]:
    probs = torch.zeros((4, EOS_ID + 1))
    probs[0, 9], probs[0, EOS_ID] = 0.9, 0.1
    probs[1:, 2] = 0.4
    probs[1, 5] = probs[2, 9] = probs[3, EOS_ID] = 0.6
    candidate_ids = torch.tensor([9, 5, 9, EOS_ID], dtype=torch.int32)
    candidate_logprobs = torch.log(
        probs.gather(1, candidate_ids.long().unsqueeze(1)).squeeze(1)
    )
    redraw_probs = []
    synced_ids = []

    def take_most_likely(probs, sampling_seed, positions):
        redraw_probs.append(probs)
        return probs.argmax(dim=1).to(torch.int32)

    monkeypatch.setattr(
        repetition_aware_module, "sampling_from_probs_torch", take_most_likely
    )
    monkeypatch.setattr(
        model_runner_module, "SGLANG_RETURN_ORIGINAL_LOGPROB", return_original_logprob
    )
    sampler = SimpleNamespace(
        rl_on_policy_target=rl_on_policy_target,
        use_ascend_backend=False,
        _sync_token_ids_across_tp=lambda ids, info: synced_ids.append(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            ids.tolist()
        ),
    )
    runner = object.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    runner.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(sampler=sampler))
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                output_codes=[],
                stream_metadata=None,
                return_logprob=row == 0,
                output_token_logprobs=(
                    [[float(candidate_logprobs[0]), 9]] if row == 0 else []
                ),
                req=SimpleNamespace(
                    output_ids=array("q", row_output_ids),
                    sampling_params=SimpleNamespace(top_k=1 if row == 2 else 20),
                ),
            )
        )
        for row, row_output_ids in enumerate(output_ids)
    ]
    result = SimpleNamespace(
        next_token_ids=candidate_ids.clone(),
        logits_output=SimpleNamespace(
            next_token_logits=probs, next_token_logprobs=candidate_logprobs.clone()
        ),
    )
    forward_batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        sampling_info=SimpleNamespace(
            sampling_seed=None, temperatures=torch.full((4, 1), 0.5)
        ),
        positions=torch.arange(4),
    )

    runner.collect_tokens(result, forward_batch, None, requests)
    return result, requests, redraw_probs, synced_ids


@pytest.mark.parametrize("return_original_logprob", [False, True])
def test_cosyvoice3_ras_redraws_repeated_speech_token_from_full_distribution(
    monkeypatch: pytest.MonkeyPatch, return_original_logprob: bool
) -> None:
    result, requests, redraw_probs, synced_ids = collect_ras_step(
        monkeypatch,
        output_ids=[[4, 9, 7], [1, 2, 3], [9], [EOS_ID]],
        rl_on_policy_target=None,
        return_original_logprob=return_original_logprob,
    )

    expected_logprob = (
        0.5 * math.log(0.1) - math.log(math.sqrt(0.9) + math.sqrt(0.1))
        if return_original_logprob
        else math.log(0.1)
    )
    assert result.next_token_ids.tolist() == [EOS_ID, 5, 9, EOS_ID]
    assert [
        [int(code) for code in request.data.output_codes] for request in requests
    ] == [[], [5], [9], []]
    assert synced_ids == [[EOS_ID, 5, 9, EOS_ID]]
    assert requests[0].data.output_token_logprobs == [
        [pytest.approx(expected_logprob), EOS_ID]
    ]
    assert result.logits_output.next_token_logprobs[0].item() == pytest.approx(
        expected_logprob
    )
    assert redraw_probs[0][0, 9].item() == 0.0
    assert redraw_probs[0][0, EOS_ID].item() == pytest.approx(0.1)


def test_cosyvoice3_ras_launches_nothing_without_a_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, requests, redraw_probs, synced_ids = collect_ras_step(
        monkeypatch,
        output_ids=[[4, 7], [1, 2, 3], [9], [EOS_ID]],
        rl_on_policy_target=None,
        return_original_logprob=False,
    )

    assert result.next_token_ids.tolist() == [9, 5, 9, EOS_ID]
    assert redraw_probs == []
    assert synced_ids == []


def test_cosyvoice3_ras_leaves_rl_on_policy_draws_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, redraw_probs, synced_ids = collect_ras_step(
        monkeypatch,
        output_ids=[[4, 9, 7], [1, 2, 3], [9], [EOS_ID]],
        rl_on_policy_target="fsdp",
        return_original_logprob=False,
    )

    assert result.next_token_ids.tolist() == [9, 5, 9, EOS_ID]
    assert redraw_probs == []
    assert synced_ids == []


def test_cosyvoice3_load_weights_maps_custom_and_backbone_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the loader path, not only the standalone key mapper."""
    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    speech_embedding = torch.nn.Parameter(torch.zeros(2, 3))
    decoder = torch.nn.Parameter(torch.zeros(2, 3))
    model.cached_params_dict = {
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
                    extend_range=SimpleNamespace(length=2), prefix_indices=[99]
                ),
                prompt_input_embeds=torch.arange(12, dtype=torch.float32).reshape(3, 4),
            )
        )
    ]
    forward_batch = SimpleNamespace(input_ids=torch.zeros(2, dtype=torch.long))

    result = runner.build_prefill_input_embeds(forward_batch, requests)

    assert torch.equal(
        result, torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]], dtype=torch.float32)
    )
