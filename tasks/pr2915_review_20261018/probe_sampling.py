"""Exercise the real thinker sampler's seed routing, filter support and cost."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.sglang_model_runner import (
    register_flashinfer_seeded_sampler,
)

VOCABULARY_SIZE = 152064


def sampling_inputs(
    row_count: int,
    top_k: int,
    top_p: float,
    min_p: float = 0.0,
    seeds: list[int] | None = None,
    capture_support: bool = False,
    greedy: bool = False,
) -> SamplingBatchInfo:
    return SamplingBatchInfo(
        temperatures=torch.ones((row_count, 1), device="cuda"),
        top_ps=torch.full((row_count,), top_p, device="cuda"),
        top_ks=torch.full((row_count,), top_k, device="cuda", dtype=torch.int32),
        min_ps=torch.full((row_count,), min_p, device="cuda"),
        is_all_greedy=greedy,
        is_any_greedy=greedy,
        need_top_k_sampling=top_k < VOCABULARY_SIZE,
        need_top_p_sampling=top_p < 1.0,
        need_min_p_sampling=min_p > 0.0,
        vocab_size=VOCABULARY_SIZE,
        sampling_seed=(
            torch.tensor(seeds, device="cuda", dtype=torch.long)
            if seeds is not None
            else None
        ),
        sampling_mask_batch_indices=(
            torch.arange(row_count, device="cuda") if capture_support else None
        ),
        device="cuda",
    )


def sample(
    backend: str, logits: torch.Tensor, sampling: SamplingBatchInfo
) -> tuple[torch.Tensor, LogitsProcessorOutput]:
    with (
        get_context().override_server_args(sampling_backend=backend),
        get_parallel().override(tp_group=SimpleNamespace(device_group=None)),
    ):
        sampler = create_sampler()
        output = LogitsProcessorOutput(next_token_logits=logits.clone())
        tokens = sampler(
            output,
            sampling,
            False,
            [0] * logits.shape[0],
            [[] for _ in range(logits.shape[0])],
            torch.arange(logits.shape[0], device="cuda"),
        )
        return tokens, output


def check_contracts() -> dict[str, bool | dict[str, list[int]]]:
    generator = torch.Generator(device="cuda").manual_seed(2915)
    logits = torch.randn((4, VOCABULARY_SIZE), device="cuda", generator=generator)
    checks: dict[str, bool | dict[str, list[int]]] = {}
    for name, top_k, top_p, greedy in (
        ("greedy", 1, 1.0, True),
        ("temperature_only", VOCABULARY_SIZE, 1.0, False),
        ("seeded_joint", 20, 0.8, False),
    ):
        seeds = [11, 22, 33, 44] if name == "seeded_joint" else None
        torch.manual_seed(77)
        previous, _ = sample(
            "pytorch",
            logits,
            sampling_inputs(4, top_k, top_p, seeds=seeds, greedy=greedy),
        )
        torch.manual_seed(77)
        current, _ = sample(
            "flashinfer",
            logits,
            sampling_inputs(4, top_k, top_p, seeds=seeds, greedy=greedy),
        )
        assert torch.equal(previous, current), name
        checks[name] = True

    runner = ModelRunner.__new__(ModelRunner)
    sampling = sampling_inputs(4, 20, 0.8)
    requests = [
        SimpleNamespace(
            request_id=f"row-{index}",
            data=SimpleNamespace(
                req=SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=seed))
            ),
        )
        for index, seed in enumerate([11, None, 33, None])
    ]
    runner.install_sampling_seeds(SimpleNamespace(sampling_info=sampling), requests)
    previous, _ = sample("pytorch", logits, sampling)
    current, _ = sample("flashinfer", logits, sampling)
    assert torch.equal(previous, current)
    assert requests[1].data.req.sampling_params.sampling_seed is None
    checks["mixed_seeded_batch"] = True

    four_token_logits = torch.full((1, VOCABULARY_SIZE), -torch.inf, device="cuda")
    four_token_logits[0, :4] = torch.tensor([0.4, 0.3, 0.2, 0.1], device="cuda").log()
    supports: dict[str, list[int]] = {}
    for backend in ("pytorch", "flashinfer"):
        for min_p in (0.0, 0.01):
            _, output = sample(
                backend,
                four_token_logits,
                sampling_inputs(1, 2, 0.5, min_p=min_p, capture_support=True),
            )
            mask = output.sampling_mask_output
            assert mask.statuses.tolist() == [0]
            supports[f"{backend}_min_p_{min_p}"] = sorted(
                mask.token_ids[0, : int(mask.lengths[0])].tolist()
            )
    checks["filter_support"] = supports

    mixed_min_p_support: dict[str, list[int]] = {}
    for backend in ("pytorch", "flashinfer"):
        mixed_min_p = sampling_inputs(2, 2, 0.5, capture_support=True)
        mixed_min_p.min_ps[1] = 0.01
        mixed_min_p.need_min_p_sampling = True
        _, mixed_min_p_output = sample(
            backend, four_token_logits.repeat(2, 1), mixed_min_p
        )
        mixed_mask = mixed_min_p_output.sampling_mask_output
        mixed_min_p_support[backend] = sorted(
            mixed_mask.token_ids[0, : int(mixed_mask.lengths[0])].tolist()
        )
    checks["mixed_min_p_neighbor_support"] = mixed_min_p_support

    tied_logits = torch.full((2, VOCABULARY_SIZE), -20.0, device="cuda")
    tied_logits[:, [7, 300, 2048]] = 10.0
    mixed_greedy_support: dict[str, list[int]] = {}
    for backend in ("pytorch", "flashinfer"):
        mixed_greedy = sampling_inputs(2, 20, 1.0, capture_support=True)
        mixed_greedy.top_ks[0] = 1
        mixed_greedy.is_any_greedy = True
        _, mixed_output = sample(backend, tied_logits, mixed_greedy)
        mask = mixed_output.sampling_mask_output
        mixed_greedy_support[backend] = sorted(
            mask.token_ids[0, : int(mask.lengths[0])].tolist()
        )
    checks["mixed_greedy_tie_support"] = mixed_greedy_support
    return checks


def measure_sampling() -> list[dict[str, int | str | float]]:
    measurements: list[dict[str, int | str | float]] = []
    for row_count in (1, 16, 64):
        logits = torch.randn((row_count, VOCABULARY_SIZE), device="cuda")
        sampling = sampling_inputs(row_count, 20, 0.8)
        for backend in ("pytorch", "flashinfer"):
            with (
                get_context().override_server_args(sampling_backend=backend),
                get_parallel().override(tp_group=SimpleNamespace(device_group=None)),
            ):
                sampler = create_sampler()
                positions = torch.arange(row_count, device="cuda")
                for iteration in range(120):
                    if iteration == 20:
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()
                    else:
                        pass
                    sampler(
                        LogitsProcessorOutput(next_token_logits=logits.clone()),
                        sampling,
                        False,
                        [0] * row_count,
                        [[] for _ in range(row_count)],
                        positions,
                    )
                end.record()
                end.synchronize()
                measurements.append(
                    {
                        "rows": row_count,
                        "backend": backend,
                        "microseconds": start.elapsed_time(end) * 10,
                    }
                )
    return measurements


parser = argparse.ArgumentParser()
parser.add_argument("--assert-min-p-backend", choices=("pytorch", "flashinfer"))
arguments = parser.parse_args()
register_flashinfer_seeded_sampler()
with torch.no_grad():
    evidence = {"contracts": check_contracts(), "sampling_cost": measure_sampling()}
Path("/data/review2915/sampling.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps(evidence, indent=2))
if arguments.assert_min_p_backend is not None:
    backend = arguments.assert_min_p_backend
    assert evidence["contracts"]["filter_support"][f"{backend}_min_p_0.01"] == [0, 1]
    assert evidence["contracts"]["mixed_min_p_neighbor_support"][backend] == [0, 1]
else:
    pass
