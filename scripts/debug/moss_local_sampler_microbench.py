#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parity and timing harness for the MOSS Local seeded sampler.

This is a debug-branch tool for developing a future fused sampler/custom op.
It isolates the exact sampler contract used by MOSS Local frame decode:

    logits, temperature, top_p, top_k, seeds, positions -> sampled token ids

The production path calls this sampler once for text stop and once per audio
codebook channel. Any replacement must first pass this harness before it is
tested inside frame CUDA graphs or full SeedTTS serving.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SGLANG_PYTHON = REPO_ROOT.parent / "sglang" / "python"

from sglang_omni.models.moss_tts_local.local_transformer import (  # noqa: E402
    sample_seeded_branchless,
)

SamplerFn = Callable[..., torch.Tensor]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _load_candidate(name: str, compile_mode: str) -> SamplerFn:
    if name == "same":
        return sample_seeded_branchless
    if name == "compile":
        return torch.compile(sample_seeded_branchless, mode=compile_mode)
    if ":" not in name:
        raise ValueError(
            "candidate must be 'same', 'compile', or 'module:function', "
            f"got {name!r}"
        )
    module_name, func_name = name.rsplit(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"candidate {name!r} is not callable")
    return fn


def _add_sglang_python_path(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SGLang Python path does not exist: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return str(resolved)


def _make_params(
    *,
    batch_size: int,
    vocab: int,
    scenario: str,
    device: torch.device,
    case_seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(case_seed)

    if scenario == "wide":
        logits = (
            torch.randn(
                batch_size,
                vocab,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            * 3.0
        )
    elif scenario == "near_tie":
        logits = (
            torch.randn(
                batch_size,
                vocab,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            * 1.0e-3
        )
    elif scenario == "descending":
        base = torch.linspace(4.0, -4.0, vocab, device=device, dtype=torch.float32)
        noise = (
            torch.randn(
                batch_size,
                vocab,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            * 1.0e-3
        )
        logits = base.unsqueeze(0) + noise
    elif scenario == "spiky":
        logits = torch.randn(
            batch_size, vocab, device=device, dtype=torch.float32, generator=generator
        )
        hot = torch.randint(vocab, (batch_size,), device=device, generator=generator)
        logits[torch.arange(batch_size, device=device), hot] += 12.0
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    row = torch.arange(batch_size, device=device)
    temperature_values = torch.tensor([1.7, 1.0, 0.5, 0.0], device=device)
    temperature = temperature_values[row % temperature_values.numel()].float()

    top_p_values = torch.tensor([0.8, 0.9, 1.0, 0.95], device=device)
    top_p = top_p_values[row % top_p_values.numel()].float()

    if vocab <= 2:
        top_k_values = torch.tensor([1, vocab, 0, vocab], device=device)
    else:
        top_k_values = torch.tensor([25, min(50, vocab), 0, vocab], device=device)
    top_k = top_k_values[row % top_k_values.numel()].long()

    seeds = (row.long() + 1) * 1_234_567 + case_seed
    positions = row.long() * 13 + (case_seed % 257)

    return {
        "logits": logits,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "seeds": seeds,
        "positions": positions,
    }


def _call(fn: SamplerFn, params: dict[str, torch.Tensor]) -> torch.Tensor:
    return fn(
        params["logits"],
        temperature=params["temperature"],
        top_p=params["top_p"],
        top_k=params["top_k"],
        seeds=params["seeds"],
        positions=params["positions"],
    )


def _time_eager(
    fn: SamplerFn,
    params: dict[str, torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> float:
    for _ in range(warmup):
        _call(fn, params)
    if params["logits"].is_cuda:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            _call(fn, params)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)) / float(iters)

    start_time = time.perf_counter()
    for _ in range(iters):
        _call(fn, params)
    return (time.perf_counter() - start_time) * 1000.0 / float(iters)


def _time_cuda_graph(
    fn: SamplerFn,
    params: dict[str, torch.Tensor],
    *,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor, float]:
    if not params["logits"].is_cuda:
        raise RuntimeError("CUDA graph timing requires CUDA tensors")

    for _ in range(warmup):
        _call(fn, params)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(fn, params)

    graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return out.detach().clone(), float(start.elapsed_time(end)) / float(iters)


def _run_case(
    *,
    reference: SamplerFn,
    candidate: SamplerFn,
    params: dict[str, torch.Tensor],
    warmup: int,
    iters: int,
    cuda_graph: bool,
) -> dict[str, object]:
    ref = _call(reference, params).detach().clone()
    cand = _call(candidate, params).detach().clone()
    if params["logits"].is_cuda:
        torch.cuda.synchronize()

    mismatch = cand.ne(ref)
    result: dict[str, object] = {
        "mismatches": int(mismatch.sum().item()),
        "rows": int(ref.numel()),
        "ref": ref.detach().cpu().tolist(),
        "candidate": cand.detach().cpu().tolist(),
        "candidate_eager_ms": _time_eager(
            candidate, params, warmup=warmup, iters=iters
        ),
    }

    if cuda_graph:
        graph_out, graph_ms = _time_cuda_graph(
            candidate, params, warmup=warmup, iters=iters
        )
        graph_mismatch = graph_out.ne(ref)
        result.update(
            {
                "candidate_cuda_graph_ms": graph_ms,
                "cuda_graph_mismatches": int(graph_mismatch.sum().item()),
                "cuda_graph_candidate": graph_out.detach().cpu().tolist(),
            }
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="same")
    parser.add_argument(
        "--compile-mode", default=os.environ.get("SGLANG_TORCH_COMPILE_MODE", "default")
    )
    parser.add_argument(
        "--sglang-python-path",
        default=os.environ.get(
            "SGLANG_PYTHON_PATH",
            str(DEFAULT_SGLANG_PYTHON) if DEFAULT_SGLANG_PYTHON.exists() else "",
        ),
        help="Optional path to the sibling SGLang python package.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--vocabs", default="2,1024")
    parser.add_argument("--scenarios", default="wide,near_tie,descending,spiky")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")
    if args.cuda_graph and device.type != "cuda":
        raise RuntimeError("--cuda-graph requires --device cuda")
    sglang_python_path = ""
    if args.sglang_python_path:
        sglang_python_path = _add_sglang_python_path(args.sglang_python_path)

    candidate = _load_candidate(args.candidate, args.compile_mode)
    batch_sizes = _parse_int_csv(args.batch_sizes)
    vocabs = _parse_int_csv(args.vocabs)
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]

    results: list[dict[str, object]] = []
    failures = 0
    for batch_size in batch_sizes:
        for vocab in vocabs:
            for scenario in scenarios:
                case_seed = args.seed + batch_size * 1_000 + vocab * 10 + len(results)
                params = _make_params(
                    batch_size=batch_size,
                    vocab=vocab,
                    scenario=scenario,
                    device=device,
                    case_seed=case_seed,
                )
                result = _run_case(
                    reference=sample_seeded_branchless,
                    candidate=candidate,
                    params=params,
                    warmup=args.warmup,
                    iters=args.iters,
                    cuda_graph=args.cuda_graph,
                )
                failures += int(result["mismatches"] != 0)
                failures += int(result.get("cuda_graph_mismatches", 0) != 0)
                result.update(
                    {
                        "batch_size": batch_size,
                        "vocab": vocab,
                        "scenario": scenario,
                    }
                )
                results.append(result)
                graph_part = ""
                if args.cuda_graph:
                    graph_part = (
                        f" graph_ms={result['candidate_cuda_graph_ms']:.4f}"
                        f" graph_mismatches={result['cuda_graph_mismatches']}"
                    )
                print(
                    f"bs={batch_size} vocab={vocab} scenario={scenario} "
                    f"mismatches={result['mismatches']} "
                    f"eager_ms={result['candidate_eager_ms']:.4f}{graph_part}",
                    flush=True,
                )

    report = {
        "candidate": args.candidate,
        "compile_mode": args.compile_mode,
        "sglang_python_path": sglang_python_path,
        "device": str(device),
        "cuda_graph": bool(args.cuda_graph),
        "warmup": args.warmup,
        "iters": args.iters,
        "failures": failures,
        "results": results,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"wrote {out_path}", flush=True)

    if args.fail_on_mismatch and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
