# SPDX-License-Identifier: Apache-2.0
"""GPU work for higgs_sampling; imported only inside one selected-mode worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any

import torch

from benchmarks.eval.asr_profiling import collect_environment_fingerprint
from sglang_omni.models.higgs_tts.sampler import (
    NO_SEED,
    STOP_CODE,
    HiggsBatchedSamplerState,
    _draw_unseeded_probs,
    _filtered_probs,
    _sample_independent_batched,
    batched_step,
    batched_step_direct,
)
from sglang_omni.models.higgs_tts.sampling_diagnostics import USE_GUMBEL_SAMPLE
from sglang_omni.models.higgs_tts.utils import BOC_ID, EOC_ID

CODEBOOKS = 8
VOCAB = 1026
FIXTURE_SEED = 20260907
WARMUP_STEPS = 10


@dataclass
class SamplerFixture:
    logits: torch.Tensor
    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor
    state: HiggsBatchedSamplerState
    live_rows: int

    def reset(self) -> None:
        self.state.delay_count.fill_(CODEBOOKS)
        self.state.eoc_countdown.fill_(-1)
        self.state.generation_done.zero_()
        self.state.generation_done[self.live_rows :] = True
        self.state.last_codes.zero_()
        self.state.step_count.zero_()

    def sample(self) -> torch.Tensor:
        return _sample_independent_batched(
            self.logits,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k_buf=self.top_k,
            seeds_B=self.state.seeds,
            step_B=self.state.step_count,
        )

    def advance(self) -> torch.Tensor:
        state = self.state
        codes, delay, countdown, done, last_codes, steps = batched_step_direct(
            self.logits,
            state.delay_count,
            state.eoc_countdown,
            state.generation_done,
            state.last_codes,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k_buf=self.top_k,
            seeds=state.seeds,
            step_count=state.step_count,
        )
        # Persist into stable storage just as decode_codebooks_batch_cg does.
        state.delay_count.copy_(delay)
        state.eoc_countdown.copy_(countdown)
        state.generation_done.copy_(done)
        state.last_codes.copy_(last_codes)
        state.step_count.copy_(steps)
        return codes

    def probabilities(self) -> torch.Tensor:
        return _filtered_probs(
            self.logits,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k_buf=self.top_k,
        )


def _fixture(
    batch: int, distribution: str, filtered: bool, live_rows: int | None = None
) -> SamplerFixture:
    generator = torch.Generator(device="cuda").manual_seed(FIXTURE_SEED)
    shape = (batch, CODEBOOKS, VOCAB)
    if distribution == "flat":
        logits = torch.zeros(shape, device="cuda")
    elif distribution == "peaked":
        logits = torch.randn(shape, generator=generator, device="cuda")
        logits[..., 7] += 6
    else:
        logits = torch.randn(shape, generator=generator, device="cuda") * 3
    # Fixed-work timing: cb0 cannot terminate the synthetic stream.
    logits[:, 0, EOC_ID] = -float("inf")
    fixture = SamplerFixture(
        logits=logits,
        temperature=torch.full((batch,), 0.8 if filtered else 1.0, device="cuda"),
        top_p=torch.full((batch,), 0.9 if filtered else 1.0, device="cuda"),
        top_k=torch.full(
            (batch,), 50 if filtered else VOCAB, dtype=torch.long, device="cuda"
        ),
        state=HiggsBatchedSamplerState(batch, CODEBOOKS),
        live_rows=batch if live_rows is None else live_rows,
    )
    fixture.reset()
    return fixture


def _capture(
    function: Callable[[], torch.Tensor]
) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(WARMUP_STEPS):
            function()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = function()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


def _measure(function: Callable, reset: Callable, args: argparse.Namespace) -> dict:
    for _ in range(WARMUP_STEPS):
        function()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    blocks = []
    for _ in range(args.blocks):
        reset()
        torch.cuda.synchronize()
        start.record()
        host_start = time.perf_counter_ns()
        for _ in range(args.iterations):
            function()
        host_end = time.perf_counter_ns()
        end.record()
        end.synchronize()
        blocks.append(
            {
                "cpu_enqueue_us_per_call": (host_end - host_start)
                / args.iterations
                / 1000,
                "cuda_interval_us_per_call": start.elapsed_time(end)
                * 1000
                / args.iterations,
            }
        )
    return {
        "blocks": blocks,
        "median": {
            key: statistics.median(block[key] for block in blocks) for key in blocks[0]
        },
    }


def _benchmark_fixture(fixture: SamplerFixture, args: argparse.Namespace) -> list[dict]:
    probabilities = fixture.probabilities()
    _assert_valid_probabilities(probabilities)
    functions = {
        "draw": lambda: _draw_unseeded_probs(probabilities),
        "sampler": fixture.sample,
        "sampler_fsm": fixture.advance,
    }
    measurements = []
    for scope, function in functions.items():
        for execution in ("eager", "graph"):
            if execution == "graph":
                graph, output = _capture(function)
                call = graph.replay
            else:
                call = function
            measurements.append(
                {
                    "scope": scope,
                    "execution": execution,
                    **_measure(call, fixture.reset, args),
                }
            )
            if execution == "graph":
                # The output and graph must outlive all replays and measurements.
                del call, output, graph
    return measurements


def _bench(args: argparse.Namespace) -> dict:
    measurements = []
    for batch in args.batch_sizes:
        for distribution in ("flat", "peaked", "tail"):
            for filtered in (False, True):
                print(
                    f"batch={batch} distribution={distribution} filtered={filtered}",
                    flush=True,
                )
                fixture = _fixture(batch, distribution, filtered)
                measurements.append(
                    {
                        "batch_size": batch,
                        "live_rows": batch,
                        "distribution": distribution,
                        "filtered": filtered,
                        "measurements": _benchmark_fixture(fixture, args),
                    }
                )
    padded_batch = max(2, max(args.batch_sizes))
    fixture = _fixture(padded_batch, "tail", True, live_rows=padded_batch - 1)
    measurements.append(
        {
            "batch_size": padded_batch,
            "live_rows": fixture.live_rows,
            "distribution": "tail",
            "filtered": True,
            "measurements": _benchmark_fixture(fixture, args),
        }
    )
    return {"measurements": measurements}


def _assert_valid_probabilities(probabilities: torch.Tensor) -> None:
    assert probabilities.dtype == torch.float32
    assert bool(torch.isfinite(probabilities).all())
    assert bool((probabilities >= 0).all())
    assert bool((probabilities.sum(-1) > 0).all())


def _distribution_check(probabilities: torch.Tensor, draws: int) -> dict:
    expected = probabilities.double() / probabilities.double().sum()
    counts = torch.zeros(probabilities.numel(), dtype=torch.long, device="cuda")
    for offset in range(0, draws, 4096):
        rows = probabilities.expand(min(4096, draws - offset), -1).contiguous()
        selected = _draw_unseeded_probs(rows)
        counts += torch.bincount(selected, minlength=probabilities.numel())
    observed = counts.double() / draws
    assert bool((counts[probabilities == 0] == 0).all()), "Sampled a masked token"
    # Bernstein bound with a union bound across categories and all 16 fixtures.
    log_factor = math.log(2 * probabilities.numel() * 16 / 0.001)
    bound = torch.sqrt(2 * expected * (1 - expected) * log_factor / draws)
    bound += 2 * log_factor / (3 * draws)
    assert bool(
        ((observed - expected).abs() <= bound).all()
    ), "Categorical frequencies outside confidence bounds"
    return {
        "max_absolute_error": float((observed - expected).abs().max()),
        "draws": draws,
    }


def _validate_draws(draws: int) -> dict:
    torch.cuda.manual_seed(FIXTURE_SEED)
    reports = {}
    for distribution in ("flat", "peaked", "tail"):
        for filtered in (False, True):
            fixture = _fixture(1, distribution, filtered)
            probabilities = fixture.probabilities()
            _assert_valid_probabilities(probabilities)
            original = probabilities.clone()
            rng_state = torch.cuda.get_rng_state()
            selected = _draw_unseeded_probs(probabilities)
            assert selected.dtype == torch.int64 and selected.shape == (CODEBOOKS,)
            assert torch.equal(original, probabilities), "Draw mutated probabilities"
            if not USE_GUMBEL_SAMPLE:
                torch.cuda.set_rng_state(rng_state)
                assert torch.equal(selected, probabilities.multinomial(1).squeeze(-1))
            reports[f"{distribution}_{filtered}"] = _distribution_check(
                probabilities[0], draws
            )
    sparse = torch.zeros(VOCAB, device="cuda")
    sparse[11], sparse[97] = 0.9, 0.1
    reports["sparse"] = _distribution_check(sparse, draws)
    onehot = torch.zeros(3, VOCAB, device="cuda")
    onehot[:, 101] = 1
    assert bool((_draw_unseeded_probs(onehot) == 101).all())
    return reports


def _validate_deterministic_rows() -> list:
    fixture = _fixture(5, "tail", True)
    fixture.temperature.copy_(torch.tensor([0.0, 0.8, 1.0, 0.8, 1.0], device="cuda"))
    fixture.top_k.copy_(torch.tensor([VOCAB, 1, 50, 50, 50], device="cuda"))
    fixture.state.seeds.copy_(
        torch.tensor([NO_SEED, NO_SEED, 123, 456, NO_SEED], device="cuda")
    )
    fixture.state.step_count.fill_(9)
    first = fixture.sample()
    assert torch.equal(
        first[:4], fixture.sample()[:4]
    ), "Seeded/greedy selection depends on global RNG"
    assert torch.equal(first[:2], fixture.logits[:2].argmax(-1))
    order = torch.tensor([4, 3, 0, 2, 1], device="cuda")
    reordered = _sample_independent_batched(
        fixture.logits[order],
        temperature=fixture.temperature[order],
        top_p=fixture.top_p[order],
        top_k_buf=fixture.top_k[order],
        seeds_B=fixture.state.seeds[order],
        step_B=fixture.state.step_count[order],
    )
    assert torch.equal(
        reordered[1:], first[order[1:]]
    ), "Seeded rows depend on batch order"
    graph, captured = _capture(fixture.sample)
    graph.replay()
    assert torch.equal(
        captured[:4], first[:4]
    ), "Graph changed deterministic selections"
    return first[:4].cpu().tolist()


def _validate_graph_rng() -> None:
    fixture = _fixture(16, "flat", False)
    graph, output = _capture(fixture.sample)
    graph.replay()
    first = output.clone()
    graph.replay()
    assert not torch.equal(first, output), "CUDA graph replay reused captured RNG noise"


def _validate_fsm() -> list:
    fixture = _fixture(2, "flat", False)
    state = fixture.state
    fixture.temperature.zero_()
    fixture.logits.fill_(-float("inf"))
    fixture.logits[..., 7] = 0
    state.delay_count.zero_()
    snapshots = []
    graph, output = _capture(fixture.advance)
    state.delay_count.zero_()
    state.eoc_countdown.fill_(-1)
    state.generation_done.copy_(torch.tensor([False, True], device="cuda"))
    state.step_count.zero_()
    state.last_codes.zero_()
    for step in range(CODEBOOKS):
        graph.replay()
        expected = torch.full((CODEBOOKS,), 7, device="cuda")
        expected[step + 1 :] = BOC_ID
        assert torch.equal(output[0], expected), "Initial delay mask changed"
        assert bool((output[1] == STOP_CODE).all()), "Finished row emitted codes"
    fixture.logits[0, 0, 7] = -float("inf")
    fixture.logits[0, 0, EOC_ID] = 0
    graph.replay()
    assert int(state.eoc_countdown[0]) == CODEBOOKS - 2
    for _ in range(CODEBOOKS - 2):
        graph.replay()
    assert bool(state.generation_done[0]), "EOC wind-down did not finish"
    steps = state.step_count.clone()
    graph.replay()
    assert bool((output == STOP_CODE).all())
    assert torch.equal(steps, state.step_count), "Done rows advanced their position"
    snapshots.append(state.step_count.cpu().tolist())
    # Pool reuse is the eager wrapper's ownership boundary.
    state.reset_row(0)
    assert int(state.seeds[0]) == NO_SEED and int(state.step_count[0]) == 0
    fixture.logits[0, 0, EOC_ID] = -float("inf")
    fixture.logits[0, 0, 7] = 0
    codes = batched_step(
        fixture.logits[:1],
        state,
        torch.tensor([0], device="cuda"),
        temperature=fixture.temperature[:1],
        top_p=fixture.top_p[:1],
        top_k_buf=fixture.top_k[:1],
    )
    assert int(codes[0, 0]) == 7 and bool((codes[0, 1:] == BOC_ID).all())
    snapshots.append(codes.cpu().tolist())
    return snapshots


def _validate_filter_boundaries() -> None:
    fixture = _fixture(4, "flat", False)
    fixture.temperature.copy_(torch.tensor([0.0, 1e-8, 0.8, 1.0], device="cuda"))
    fixture.top_k.copy_(torch.tensor([1, 50, VOCAB, 50], device="cuda"))
    fixture.top_p.copy_(torch.tensor([1.0, 1e-6, 0.5, 0.9], device="cuda"))
    # Include ties and nearby FP32 values at the filtering boundary.
    fixture.logits[3, :, 0] = torch.nextafter(
        torch.tensor(0.0, device="cuda"), torch.tensor(1.0, device="cuda")
    )
    probabilities = fixture.probabilities()
    _assert_valid_probabilities(probabilities)
    codes = fixture.sample().reshape(-1, 1)
    # Greedy rows select raw-logit argmax independently of the filtered support.
    assert bool((probabilities.gather(1, codes)[2 * CODEBOOKS :] > 0).all())


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This harness requires CUDA; run it in the H100 container")
    torch.cuda.set_device(0)
    environment = collect_environment_fingerprint()
    environment["flashinfer_version"] = version("flashinfer-python")
    with torch.inference_mode():
        if args.mode == "validate":
            distributions = _validate_draws(args.draws)
            signature = [_validate_deterministic_rows(), _validate_fsm()]
            _validate_graph_rng()
            _validate_filter_boundaries()
            payload = {
                "validation": "passed",
                "distributions": distributions,
                "deterministic_signature": hashlib.sha256(
                    json.dumps(signature).encode()
                ).hexdigest(),
            }
        elif args.trace:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as profiler:
                payload = _bench(args)
            profiler.export_chrome_trace(str(args.output_dir / "trace.json"))
        else:
            payload = _bench(args)
    return {
        "backend": args.worker,
        "instrumented": args.trace,
        "environment": environment,
        "fixture_seed": FIXTURE_SEED,
        "codebooks": CODEBOOKS,
        "vocab": VOCAB,
        **payload,
    }
