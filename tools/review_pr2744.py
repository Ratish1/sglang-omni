import hashlib
import inspect
import json
from array import array
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang_omni.models.fun_cosyvoice3 import sglang_model
from sglang_omni.models.fun_cosyvoice3.sglang_model import FunCosyVoice3SGLangModel
from sglang_omni.platforms.rocm import ROCMOmniPlatform
from tests.unit_test.fun_cosyvoice3.test_model_runner import (
    test_cosyvoice3_speech_head_returns_float32_logits,
)


@torch.no_grad()
def sample_and_complete() -> dict[str, list[int] | bool]:
    from sglang.srt.layers import sampler as sampler_module
    from sglang.srt.layers.sampler import Sampler
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
    from sglang_omni.models.fun_cosyvoice3.request_builders import (
        CosyVoice3SGLangRequestData,
    )
    from sglang_omni.models.fun_cosyvoice3.sglang_model import EOS_ID, TOTAL_VOCAB_SIZE
    from sglang_omni.scheduling.types import SchedulerRequest

    model = make_model(torch.bfloat16, 4)
    model.model.hidden_states.zero_()
    model.model.hidden_states[:, :4].copy_(torch.eye(4, device="cuda"))
    model.llm_decoder = nn.Linear(
        64, TOTAL_VOCAB_SIZE, bias=False, device="cuda", dtype=torch.bfloat16
    )
    model.llm_decoder.weight.fill_(-20)
    for row, candidate in enumerate((9, 5, 9, EOS_ID)):
        model.llm_decoder.weight[candidate, row] = 2.25
        model.llm_decoder.weight[EOS_ID if candidate != EOS_ID else 9, row] = 0
    info = SamplingBatchInfo(
        temperatures=torch.ones(4, 1, device="cuda"),
        top_ps=torch.full((4,), 0.5, device="cuda"),
        top_ks=torch.tensor([20, 20, 1, 20], device="cuda"),
        min_ps=torch.zeros(4, device="cuda"),
        is_all_greedy=False,
        is_any_greedy=True,
        need_top_p_sampling=True,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=TOTAL_VOCAB_SIZE,
        sampling_seed=torch.arange(10, 14, device="cuda"),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        sampling_info=info,
        positions=torch.arange(100, 104, device="cuda"),
    )
    requests = []
    for row, history in enumerate(([9], [4], [9], [])):
        params = SamplingParams(
            max_new_tokens=2, top_k=1 if row == 2 else 20, stop_token_ids=[EOS_ID]
        )
        params.normalize(None)
        request = Req(
            rid=str(row),
            origin_input_text="",
            origin_input_ids=[0],
            sampling_params=params,
            vocab_size=TOTAL_VOCAB_SIZE,
        )
        request.output_ids = array("q", history)
        requests.append(
            SchedulerRequest(
                request_id=str(row),
                data=CosyVoice3SGLangRequestData(
                    req=request,
                    output_ids=request.output_ids,
                    stream_metadata={"stream": True},
                ),
            )
        )
    sampler = object.__new__(Sampler)
    nn.Module.__init__(sampler)
    sampler.rl_on_policy_target = None
    sampler.enable_deterministic = False
    sampler.use_log_softmax_logprob = False
    sampler.use_ascend_backend = False
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            sampler_module,
            "get_exec",
            lambda: SimpleNamespace(kernel=SimpleNamespace(sampling_backend="pytorch")),
        )
        ids = torch.zeros(4, device="cuda", dtype=torch.long)
        logits = model(ids, batch.positions, batch)
        assert logits.next_token_logits.dtype == torch.float32
        info.apply_logits_bias(logits.next_token_logits)
        candidates = sampler(logits, info, False, [], [], batch.positions)
        assert candidates.tolist() == [9, 5, 9, EOS_ID]
        result = GenerationBatchResult(logits_output=logits, next_token_ids=candidates)
        runner = object.__new__(FunCosyVoice3ModelRunner)
        runner.outbox = Queue()
        runner.vocoder_target = "vocoder"
        runner.token_hop_len = 25
        runner.ar_followup_flush_tokens = 25
        runner.token_id_host_bufs = None
        runner.token_id_host_slot = 0
        runner.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(sampler=sampler)
        )
        runner.collect_tokens(result, batch, None, requests)
        expected = [EOS_ID, 5, 9, EOS_ID]
        assert result.next_token_ids.tolist() == expected
        assert runner.resolve_host_token_ids(result).tolist() == expected
        for request, token in zip(requests, expected, strict=True):
            request.data.req.output_ids.append(token)
            request.data.req.update_finish_state()
            assert request.data.req.finished()
            runner.on_request_finished(request.request_id, request.data)
        messages = []
        while not runner.outbox.empty():
            messages.append(int(runner.outbox.get().data[0]))
        assert messages == [5, 9]
        return {
            "candidates": candidates.tolist(),
            "after_ras": expected,
            "streamed_speech_tokens": messages,
            "all_requests_finished": True,
        }


class FixedBackbone(nn.Module):
    def __init__(self, hidden_states: torch.Tensor) -> None:
        super().__init__()
        self.hidden_states = hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: SimpleNamespace,
        input_embeds: torch.Tensor | None,
        pp_proxy_tensors: None,
    ) -> torch.Tensor:
        return self.hidden_states


def make_model(dtype: torch.dtype, rows: int) -> FunCosyVoice3SGLangModel:
    model = object.__new__(FunCosyVoice3SGLangModel)
    nn.Module.__init__(model)
    generator = torch.Generator().manual_seed(0)
    hidden_states = torch.randn(rows, 64, generator=generator).to("cuda", dtype)
    model.speech_embedding = nn.Embedding(8, 64, device="cuda", dtype=dtype)
    model.llm_decoder = nn.Linear(64, 8, bias=False, device="cuda", dtype=dtype)
    model.llm_decoder.weight.data.copy_(
        torch.randn(8, 64, generator=generator).to("cuda", dtype)
    )
    model.model = FixedBackbone(hidden_states)
    return model


@torch.no_grad()
def main() -> None:
    observations: dict[str, str | bool | float | dict[str, list[int] | bool]] = {}
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        model = make_model(dtype, 4)
        token_ids = torch.zeros(4, device="cuda", dtype=torch.long)
        for phase in ("decode", "extend", "extend_without_lengths"):
            is_extend = phase != "decode"
            forward_batch = SimpleNamespace(
                forward_mode=SimpleNamespace(
                    is_decode=lambda: not is_extend, is_extend=lambda: is_extend
                ),
                extend_seq_lens=(
                    torch.tensor([3, 1], device="cuda") if phase == "extend" else None
                ),
            )
            output = model(token_ids, token_ids, forward_batch)
            if phase == "extend":
                selected = model.model.hidden_states[[2, 3]]
            elif phase == "extend_without_lengths":
                selected = model.model.hidden_states[-1:]
            else:
                selected = model.model.hidden_states
            assert output.next_token_logits.dtype == torch.float32
            assert output.hidden_states.dtype == dtype
            reference = selected.double() @ model.llm_decoder.weight.double().T
            error = (output.next_token_logits.double() - reference).abs().max()
            assert float(error) < 1e-3
            observations[f"{dtype}_{phase}_max_error"] = float(error)

    for rows in (1, 4, 32):
        model = make_model(torch.bfloat16, rows)
        token_ids = torch.zeros(rows, device="cuda", dtype=torch.long)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: True, is_extend=lambda: False
            )
        )
        for _ in range(3):
            model(token_ids, token_ids, batch)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = model(token_ids, token_ids, batch)
        for _ in range(2):
            model.model.hidden_states.add_(0.125)
            graph.replay()
            eager = model(token_ids, token_ids, batch)
            torch.testing.assert_close(
                captured.next_token_logits, eager.next_token_logits
            )
            assert captured.next_token_logits.dtype == torch.float32
        observations[f"graph_rows_{rows}"] = True

    torch.manual_seed(0)
    test_cosyvoice3_speech_head_returns_float32_logits()
    original_mm = torch.mm

    def rounded_head(
        hidden_states: torch.Tensor, weight: torch.Tensor, *, out_dtype: torch.dtype
    ) -> torch.Tensor:
        return original_mm(hidden_states, weight).float()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "mm", rounded_head)
        try:
            test_cosyvoice3_speech_head_returns_float32_logits()
        except AssertionError:
            observations["test_rejects_rounded_head_upcast"] = True
        else:
            raise AssertionError(
                "the precision test accepted a BF16 head followed by a cast"
            )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sglang_model, "current_platform", ROCMOmniPlatform())
        assert torch.cuda.is_available()
        try:
            test_cosyvoice3_speech_head_returns_float32_logits()
        except AssertionError:
            observations["cuda_available_with_rocm_policy_test_fails"] = True
        else:
            raise AssertionError(
                "expected the CUDA-only test to disagree with the ROCm policy"
            )

    from sglang.srt.layers import sampler

    observations["sampler_sha256"] = hashlib.sha256(
        Path(inspect.getfile(sampler)).read_bytes()
    ).hexdigest()
    observations["sampling_and_completion"] = sample_and_complete()
    Path("/data/pr2744-review/head-probe.json").write_text(
        json.dumps(observations, indent=2) + "\n"
    )
    print(json.dumps(observations, indent=2), flush=True)


if __name__ == "__main__":
    main()
