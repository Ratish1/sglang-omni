# SPDX-License-Identifier: Apache-2.0
"""Check CosyVoice sampling, staging and token relay without model weights."""

from __future__ import annotations

import json
from array import array
from pathlib import Path
from types import SimpleNamespace

import sglang.srt.layers.sampler as sampler_module
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.managers.overlap_utils import FutureMap, resolve_forward_inputs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.sglang_execution import SGLangExecutionBridge
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3SGLangRequestData,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import EOS_ID, TOTAL_VOCAB_SIZE
from sglang_omni.sampling.repetition_aware import repetition_aware_redraw
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest


@torch.no_grad()
def main() -> None:
    device = torch.device("cuda", 0)
    sampler = Sampler.__new__(Sampler)
    torch.nn.Module.__init__(sampler)
    sampler.rl_on_policy_target = None
    sampler.enable_deterministic = False
    sampler.use_log_softmax_logprob = False
    sampler.use_ascend_backend = False
    sampler_module.get_exec = lambda: SimpleNamespace(
        kernel=SimpleNamespace(sampling_backend="pytorch")
    )
    histories = [[4, 9, 7], [1, 2, 3], [9], [], [9], [9] + [4] * 10]
    top_ks = [20, 20, 1, 20, 20, 20]
    requests = []
    for row, (history, top_k) in enumerate(zip(histories, top_ks, strict=True)):
        parameters = SamplingParams(top_k=top_k, stop_token_ids=[EOS_ID])
        parameters.normalize(None)
        request = Req(
            rid=f"row-{row}",
            origin_input_text="",
            origin_input_ids=[0],
            sampling_params=parameters,
            vocab_size=TOTAL_VOCAB_SIZE,
        )
        request.output_ids = array("q", history)
        requests.append(
            SchedulerRequest(
                request_id=request.rid,
                data=CosyVoice3SGLangRequestData(
                    req=request, output_ids=request.output_ids
                ),
            )
        )
    probabilities = torch.zeros((6, TOTAL_VOCAB_SIZE), device=device)
    probabilities[0, 9], probabilities[0, EOS_ID] = 0.9, 0.1
    probabilities[1, 5], probabilities[1, 9] = 0.9, 0.1
    probabilities[2, 9], probabilities[2, EOS_ID] = 0.9, 0.1
    probabilities[3, EOS_ID], probabilities[3, 9] = 0.9, 0.1
    probabilities[4, 9] = 1.0
    probabilities[5, 9], probabilities[5, EOS_ID] = 0.9, 0.1
    sampling_info = SamplingBatchInfo(
        temperatures=torch.ones((6, 1), device=device),
        top_ps=torch.full((6,), 0.5, device=device),
        top_ks=torch.tensor(top_ks, device=device),
        min_ps=torch.zeros(6, device=device),
        is_all_greedy=False,
        is_any_greedy=True,
        need_top_p_sampling=True,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=TOTAL_VOCAB_SIZE,
        sampling_seed=torch.arange(10, 16, device=device),
    )
    positions = torch.arange(100, 106, device=device)
    logits_output = LogitsProcessorOutput(next_token_logits=probabilities.log())
    candidate_ids = sampler(logits_output, sampling_info, False, [], [], positions)
    assert candidate_ids.tolist() == [9, 5, 9, EOS_ID, 9, 9]
    assert float(logits_output.next_token_logits[0, EOS_ID]) > 0
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        sampling_info=sampling_info,
        positions=positions,
    )
    result = GenerationBatchResult(
        logits_output=logits_output, next_token_ids=candidate_ids
    )
    runner = FunCosyVoice3ModelRunner.__new__(FunCosyVoice3ModelRunner)
    runner.outbox = None
    runner.token_id_host_bufs = None
    runner.token_id_host_slot = 0
    runner.tp_worker = SimpleNamespace(model_runner=SimpleNamespace(sampler=sampler))
    runner.output_processor = SGLangOutputProcessor()
    runner.collect_tokens(result, forward_batch, None, requests)
    expected = [EOS_ID, 5, 9, EOS_ID, 9, 9]
    assert result.next_token_ids.tolist() == expected
    assert runner.resolve_host_token_ids(result).tolist() == expected
    assert runner.resolve_host_token_ids(result).is_pinned()
    spec_algorithm = SpeculativeAlgorithm.NONE
    future_map = FutureMap(
        device,
        spec_algorithm,
        SimpleNamespace(
            req_to_token=torch.zeros((7, 1), dtype=torch.int32, device=device)
        ),
        needs_cpu_seq_lens=False,
    )
    runner.execution_bridge = SGLangExecutionBridge(
        device=device,
        worker=runner.tp_worker,
        spec_algorithm=spec_algorithm,
        future_map=future_map,
    )
    batch = SimpleNamespace(
        is_prefill_only=False,
        reqs=[request.data.req for request in requests],
        req_pool_indices=torch.arange(1, 7, device=device),
        input_ids=candidate_ids,
        prefill_input_ids_cpu=None,
        enable_overlap=False,
        spec_algorithm=spec_algorithm,
    )
    runner.publish_next_tokens(result, forward_batch, batch, requests)
    resolve_forward_inputs(batch, future_map)
    assert batch.input_ids.tolist() == expected
    finalized = runner.finalize(
        result, forward_batch, batch, SchedulerOutput(requests, batch)
    )
    assert [
        finalized.outputs[request.request_id].data for request in requests
    ] == expected
    for request, token_id in zip(requests, expected, strict=True):
        request.data.req.output_ids.append(token_id)
        request.data.req.update_finish_state()
    assert requests[0].data.req.finished()
    assert not requests[5].data.req.finished()

    blocked = SamplingBatchInfo(
        temperatures=torch.ones((1, 1), device=device),
        top_ps=torch.tensor([0.5], device=device),
        top_ks=torch.tensor([20], device=device),
        min_ps=torch.zeros(1, device=device),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=True,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=TOTAL_VOCAB_SIZE,
        acc_additive_penalties=torch.zeros((1, TOTAL_VOCAB_SIZE), device=device),
    )
    blocked.acc_additive_penalties[0, EOS_ID] = -float("inf")
    blocked_logits = LogitsProcessorOutput(next_token_logits=probabilities[:1].log())
    blocked.apply_logits_bias(blocked_logits.next_token_logits)
    blocked_ids = sampler(blocked_logits, blocked, False, [], [], positions[:1])
    masked_redraw = repetition_aware_redraw(
        blocked_logits.next_token_logits,
        blocked_ids,
        torch.tensor([0], device=device),
        None,
        positions[:1],
    )
    assert masked_redraw.tolist() == [9]
    subset_probabilities = torch.zeros((6, 8), device=device)
    subset_probabilities[:, 3] = 0.8
    subset_probabilities[:, 5] = 0.12
    subset_probabilities[:, 6] = 0.08
    sampled_ids = torch.full((6,), 3, dtype=torch.int32, device=device)
    all_rows = torch.arange(6, device=device)
    selected_rows = torch.tensor([5, 0, 3], device=device)
    seeds = torch.arange(10, 16, device=device)
    full_redraw = repetition_aware_redraw(
        subset_probabilities, sampled_ids, all_rows, seeds, positions
    )
    subset_redraw = repetition_aware_redraw(
        subset_probabilities, sampled_ids, selected_rows, seeds, positions
    )
    assert torch.equal(subset_redraw[selected_rows], full_redraw[selected_rows])
    assert subset_redraw[torch.tensor([1, 2, 4], device=device)].tolist() == [3, 3, 3]
    permutation = torch.tensor([2, 5, 1, 4, 0, 3], device=device)
    compact_redraw = repetition_aware_redraw(
        subset_probabilities[permutation],
        sampled_ids[permutation],
        all_rows,
        seeds[permutation],
        positions[permutation],
    )
    assert torch.equal(compact_redraw, full_redraw[permutation])
    evidence = {
        "candidates": candidate_ids.tolist(),
        "collected_relayed_finalized": expected,
        "outside_window_kept": True,
        "greedy_kept": True,
        "collapsed_distribution_kept": True,
        "redrawn_eos_finishes": True,
        "suppressed_eos_stays_suppressed": True,
        "seeded_subset_and_permutation_match": True,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
    }
    Path("/data/pr2625-2626-review/real-sampler-probe.json").write_text(
        json.dumps(evidence, indent=2)
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
