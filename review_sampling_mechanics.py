"""Independent CUDA review probes for the sampling changes in PRs 2897 and 2898."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import time
from array import array
from contextlib import ExitStack
from types import SimpleNamespace

import torch
from flashinfer.sampling import get_sampling_module
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_model_runner import (
    register_flashinfer_seeded_sampler,
)
from sglang_omni.models.fun_cosyvoice3.model_runner import FunCosyVoice3ModelRunner
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    EOS_ID,
    TOTAL_VOCAB_SIZE,
    FunCosyVoice3SGLangModel,
)
from sglang_omni.sampling.seed import SAMPLING_SEED_MASK, derive_sampling_seed


def request(seed: int | None, request_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        data=SimpleNamespace(
            req=SimpleNamespace(
                sampling_params=SimpleNamespace(sampling_seed=seed, top_k=20),
                output_ids=array("q"),
            ),
            output_codes=[],
        ),
    )


def seed_batch() -> SimpleNamespace:
    return SimpleNamespace(
        sampling_info=SimpleNamespace(
            sampling_seed=None,
            device="cuda",
            need_min_p_sampling=False,
        )
    )


def sampling_info(rows: int, seeds: torch.Tensor | None) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        temperatures=torch.full((rows, 1), 0.7, device="cuda"),
        top_ps=torch.full((rows,), 0.8, device="cuda"),
        top_ks=torch.full((rows,), 20, dtype=torch.int32, device="cuda"),
        min_ps=torch.zeros(rows, device="cuda"),
        is_all_greedy=False,
        is_any_greedy=False,
        need_top_p_sampling=True,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=TOTAL_VOCAB_SIZE,
        sampling_seed=seeds,
        device="cuda",
    )


def draw(
    backend: str, logits: torch.Tensor, seeds: torch.Tensor | None
) -> tuple[torch.Tensor, LogitsProcessorOutput, SamplingBatchInfo]:
    rows = logits.shape[0]
    information = sampling_info(rows, seeds)
    output = LogitsProcessorOutput(next_token_logits=logits.clone())
    with ExitStack() as contexts:
        contexts.enter_context(
            get_context().override_server_args(sampling_backend=backend)
        )
        contexts.enter_context(
            get_parallel().override(tp_group=SimpleNamespace(device_group=None))
        )
        token_ids = create_sampler()(
            output,
            information,
            False,
            [0] * rows,
            [[] for _ in range(rows)],
            torch.arange(rows, device="cuda"),
        )
    return token_ids, output, information


def check_sampling() -> None:
    register_flashinfer_seeded_sampler()
    generator = torch.Generator().manual_seed(492)
    runner = object.__new__(ModelRunner)
    for rows in (1, 16, 32):
        logits = (torch.randn(rows, TOTAL_VOCAB_SIZE, generator=generator) * 4).cuda()
        batch = seed_batch()
        runner.install_sampling_seeds(
            batch, [request(index + 17, str(index)) for index in range(rows)]
        )
        seeds = batch.sampling_info.sampling_seed
        expected, _, _ = draw("pytorch", logits, seeds)
        actual, _, _ = draw("flashinfer", logits, seeds)
        assert torch.equal(actual, expected)
        _, output, _ = draw("flashinfer", logits, None)
        assert torch.equal(
            output.next_token_logits, torch.softmax(logits / 0.7, dim=-1)
        )
        print(f"seeded-token-parity rows={rows}: passed", flush=True)

    logits = torch.full((2, TOTAL_VOCAB_SIZE), -torch.inf, device="cuda")
    logits[0, 9], logits[0, EOS_ID], logits[1, 13] = 0.0, -2.0, 0.0
    for backend in ("pytorch", "flashinfer"):
        candidates, output, information = draw(backend, logits, None)
        assert candidates.tolist() == [9, 13]
        runner = object.__new__(FunCosyVoice3ModelRunner)
        runner.outbox = None
        runner.token_id_host_bufs = None
        runner.token_id_host_slot = 0
        runner.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=SimpleNamespace(use_ascend_backend=False)
            )
        )
        requests = [request(None, "repeat"), request(None, "ordinary")]
        requests[0].data.req.output_ids.append(9)
        result = SimpleNamespace(next_token_ids=candidates, logits_output=output)
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_decode=lambda: True),
            sampling_info=information,
            positions=torch.arange(2, device="cuda"),
        )
        runner.collect_tokens(result, batch, None, requests)
        assert result.next_token_ids.tolist() == [EOS_ID, 13]
        assert requests[0].data.output_codes == []
        assert [code.item() for code in requests[1].data.output_codes] == [13]
        print(f"live-sampler-to-RAS backend={backend}: passed", flush=True)

    model = object.__new__(FunCosyVoice3SGLangModel)
    torch.nn.Module.__init__(model)
    model.llm_decoder = torch.nn.Linear(64, TOTAL_VOCAB_SIZE, bias=False).cuda()
    get_sampling_module.cache_clear()
    with get_context().override_server_args(sampling_backend="flashinfer"):
        model.precompile_kernels_after_loading()
    after_hook = get_sampling_module.cache_info()
    assert after_hook.currsize == 1
    draw("flashinfer", logits, None)
    after_draw = get_sampling_module.cache_info()
    assert after_draw.misses == after_hook.misses
    print(f"hook-preload: passed cache={after_draw}", flush=True)


def check_seeds() -> None:
    runner = object.__new__(ModelRunner)
    originals = [0, -1, SAMPLING_SEED_MASK, 2**40 + 27, None]
    snapshots: list[tuple[torch.Tensor, list[int]]] = []
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for iteration in range(40):
        rows = (1, 2, 16, 32, 64)[iteration % 5]
        requests = [
            request(originals[index % len(originals)], f"{iteration}-{index}")
            for index in range(rows)
        ]
        expected = [
            (
                derive_sampling_seed("sglang-omni-unseeded-row", item.request_id)
                if item.data.req.sampling_params.sampling_seed is None
                else item.data.req.sampling_params.sampling_seed & SAMPLING_SEED_MASK
            )
            for item in requests
        ]
        with torch.cuda.stream(streams[iteration % 2]):
            torch.cuda._sleep(1_000_000)
            batch = seed_batch()
            runner.install_sampling_seeds(batch, requests)
            snapshots.append((batch.sampling_info.sampling_seed.clone(), expected))
        for _ in range(4):
            torch.empty(rows, dtype=torch.long, pin_memory=True).fill_(-8675309)
    torch.cuda.synchronize()
    for snapshot, expected in snapshots:
        assert snapshot.tolist() == expected
    print(
        "seed-values-and-lifetime: passed 40 alternating-stream snapshots", flush=True
    )

    for rows in (1, 16, 32):
        requests = [request(index + 42, str(index)) for index in range(rows)]
        for _ in range(5):
            runner.install_sampling_seeds(seed_batch(), requests)
        torch.cuda.synchronize()
        for mechanism in ("blocking-control", "actual-install"):
            durations: list[float] = []
            busy_count = 0
            for _ in range(100):
                torch.cuda._sleep(1_500_000)
                started = time.perf_counter_ns()
                if mechanism == "actual-install":
                    runner.install_sampling_seeds(seed_batch(), requests)
                else:
                    torch.tensor(
                        [index + 42 for index in range(rows)],
                        dtype=torch.long,
                        device="cuda",
                    )
                durations.append((time.perf_counter_ns() - started) / 1000)
                busy_count += int(not torch.cuda.current_stream().query())
                torch.cuda.synchronize()
            print(
                json.dumps(
                    {
                        "mechanism": mechanism,
                        "rows": rows,
                        "host_median_us": statistics.median(durations),
                        "stream_busy_on_return": busy_count,
                        "iterations": len(durations),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sampling", "seeds"))
    options = parser.parse_args()
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "sglang": importlib.metadata.version("sglang"),
                "flashinfer": importlib.metadata.version("flashinfer-python"),
                "gpu": torch.cuda.get_device_name(),
                "mode": options.mode,
            }
        ),
        flush=True,
    )
    if options.mode == "sampling":
        check_sampling()
    else:
        check_seeds()
