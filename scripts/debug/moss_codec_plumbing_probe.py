#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe MOSS Local codec decoder plumbing candidates in one process.

This harness keeps the semantic path fixed:

    processor.decode_audio_codes(audio_codes)

Candidates are temporary in-process patches around decoder plumbing. The gate
is strict waveform parity against the unpatched processor path before any
timing result is considered useful.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.debug.moss_local_codec_same_process_bench import (  # noqa: E402
    _case_lengths,
    _load_processor,
    _make_codes,
    _parity,
    _parse_int_csv,
    _percentile,
    _scheduler_processor_decode,
    _sync,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import (  # noqa: E402
    MossTTSLocalStreamingVocoderScheduler,
)


@contextlib.contextmanager
def _no_patch() -> Iterator[dict[str, Any]]:
    yield {"patch": "none"}


@contextlib.contextmanager
def _cuda_cudnn_sdp_disabled() -> Iterator[dict[str, Any]]:
    if not torch.cuda.is_available() or not hasattr(
        torch.backends.cuda, "enable_cudnn_sdp"
    ):
        yield {"cudnn_sdp_supported": False}
        return

    previous = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        yield {
            "cudnn_sdp_supported": True,
            "previous_cudnn_sdp_enabled": bool(previous),
        }
    finally:
        torch.backends.cuda.enable_cudnn_sdp(bool(previous))


@contextlib.contextmanager
def _torch_arange_cache() -> Iterator[dict[str, Any]]:
    """Cache torch.arange outputs for immutable decoder position ranges.

    This is a probe, not product code. It intentionally returns the same tensor
    object for identical arange calls; exact waveform parity catches any unsafe
    caller that mutates an arange result in-place.
    """

    original_arange = torch.arange
    cache: dict[tuple[Any, ...], torch.Tensor] = {}
    stats = {"hits": 0, "misses": 0, "bypassed": 0}

    def normalize_device(device: Any) -> str | None:
        if device is None:
            return None
        return str(torch.device(device))

    def cached_arange(*args: Any, **kwargs: Any) -> torch.Tensor:
        if "out" in kwargs and kwargs["out"] is not None:
            stats["bypassed"] += 1
            return original_arange(*args, **kwargs)
        layout = kwargs.get("layout", torch.strided)
        requires_grad = bool(kwargs.get("requires_grad", False))
        pin_memory = bool(kwargs.get("pin_memory", False))
        if layout is not torch.strided or requires_grad or pin_memory:
            stats["bypassed"] += 1
            return original_arange(*args, **kwargs)

        dtype = kwargs.get("dtype")
        device = kwargs.get("device")
        key = (
            tuple(args),
            str(dtype) if dtype is not None else None,
            normalize_device(device),
        )
        cached = cache.get(key)
        if cached is not None:
            stats["hits"] += 1
            return cached
        result = original_arange(*args, **kwargs)
        cache[key] = result
        stats["misses"] += 1
        return result

    torch.arange = cached_arange  # type: ignore[assignment]
    try:
        yield stats
    finally:
        torch.arange = original_arange  # type: ignore[assignment]


def _candidate_context(name: str) -> contextlib.AbstractContextManager[dict[str, Any]]:
    if name == "baseline":
        return _no_patch()
    if name == "arange_cache":
        return _torch_arange_cache()
    if name == "cudnn_sdp_disabled":
        return _cuda_cudnn_sdp_disabled()
    raise ValueError(f"unknown candidate {name!r}")


@contextlib.contextmanager
def _applied_candidate(name: str) -> Iterator[dict[str, Any]]:
    if name == "arange_cache+cudnn_sdp_disabled":
        with contextlib.ExitStack() as stack:
            arange_stats = stack.enter_context(_torch_arange_cache())
            sdp_stats = stack.enter_context(_cuda_cudnn_sdp_disabled())
            yield {"arange_cache": arange_stats, "cudnn_sdp_disabled": sdp_stats}
        return
    with _candidate_context(name) as stats:
        yield stats


def _time_once(
    *,
    fn: Callable[[], list[torch.Tensor]],
    device: torch.device,
) -> float:
    _sync(device)
    start = time.perf_counter()
    fn()
    _sync(device)
    return (time.perf_counter() - start) * 1000.0


def _time_block(
    *,
    fn: Callable[[], list[torch.Tensor]],
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    _sync(device)

    samples = [_time_once(fn=fn, device=device) for _ in range(iters)]
    return {
        "avg_ms": statistics.fmean(samples),
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples": len(samples),
    }


def _compare_timing(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    delta_pct = (
        (candidate["avg_ms"] - baseline["avg_ms"]) / baseline["avg_ms"] * 100.0
        if baseline["avg_ms"] > 0
        else 0.0
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_delta_pct": delta_pct,
        "candidate_speedup_pct": -delta_pct,
    }


def _run_candidate_case(
    *,
    processor: Any,
    scheduler: MossTTSLocalStreamingVocoderScheduler,
    candidate: str,
    batch_size: int,
    scenario: str,
    seed: int,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, Any]:
    codes_list = _make_codes(
        processor=processor,
        batch_size=batch_size,
        scenario=scenario,
        seed=seed,
    )

    def baseline_fn() -> list[torch.Tensor]:
        return _scheduler_processor_decode(scheduler, codes_list)

    with torch.inference_mode():
        baseline_wavs = baseline_fn()
        baseline_timing = _time_block(
            fn=baseline_fn,
            device=device,
            warmup=warmup,
            iters=iters,
        )
        with _applied_candidate(candidate) as patch_stats:
            candidate_wavs = _scheduler_processor_decode(scheduler, codes_list)
            parity = _parity(baseline_wavs, candidate_wavs)
            candidate_timing = _time_block(
                fn=lambda: _scheduler_processor_decode(scheduler, codes_list),
                device=device,
                warmup=warmup,
                iters=iters,
            )
            timing = _compare_timing(
                baseline=baseline_timing,
                candidate=candidate_timing,
            )

    return {
        "candidate": candidate,
        "batch_size": batch_size,
        "scenario": scenario,
        "lengths": _case_lengths(batch_size, scenario),
        "parity": parity,
        "timing": timing,
        "patch_stats": patch_stats,
    }


def _is_exact_parity(parity: dict[str, Any]) -> bool:
    return (
        bool(parity.get("shape_equal"))
        and float(parity.get("max_abs", 1.0)) == 0.0
        and float(parity.get("mean_abs", 1.0)) == 0.0
        and float(parity.get("max_rel", 1.0)) == 0.0
    )


def summarize_candidate_cases(
    cases: list[dict[str, Any]],
    *,
    min_speedup_pct: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case["candidate"]), []).append(case)

    rows = []
    for candidate, candidate_cases in grouped.items():
        parity_pass = sum(_is_exact_parity(case["parity"]) for case in candidate_cases)
        speedups = [
            float(case["timing"]["candidate_speedup_pct"])
            for case in candidate_cases
            if _is_exact_parity(case["parity"])
        ]
        avg_speedup = statistics.fmean(speedups) if speedups else 0.0
        min_speedup = min(speedups) if speedups else 0.0
        max_speedup = max(speedups) if speedups else 0.0
        accepted = (
            parity_pass == len(candidate_cases)
            and len(candidate_cases) > 0
            and avg_speedup >= min_speedup_pct
            and min_speedup > 0.0
        )
        rows.append(
            {
                "candidate": candidate,
                "cases": len(candidate_cases),
                "parity_pass": parity_pass,
                "parity_fail": len(candidate_cases) - parity_pass,
                "avg_speedup_pct": avg_speedup,
                "min_speedup_pct": min_speedup,
                "max_speedup_pct": max_speedup,
                "accepted": accepted,
            }
        )
    rows.sort(key=lambda row: (not row["accepted"], -float(row["avg_speedup_pct"])))
    return rows


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# MOSS Local Codec Plumbing Probe",
        "",
        f"model: `{report['model_path']}`",
        f"device: `{report['device']}`",
        f"warmup: `{report['warmup']}`",
        f"iters: `{report['iters']}`",
        f"min_speedup_pct: `{report['min_speedup_pct']}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate | cases | parity pass | avg speedup | min speedup | max speedup | accepted |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["candidate_summary"]:
        lines.append(
            "| {candidate} | {cases} | {parity_pass}/{cases} | {avg:.2f}% | "
            "{minv:.2f}% | {maxv:.2f}% | {accepted} |".format(
                candidate=row["candidate"],
                cases=row["cases"],
                parity_pass=row["parity_pass"],
                avg=float(row["avg_speedup_pct"]),
                minv=float(row["min_speedup_pct"]),
                maxv=float(row["max_speedup_pct"]),
                accepted="yes" if row["accepted"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| candidate | bs | scenario | parity max_abs | parity mean_abs | "
            "baseline avg ms | candidate avg ms | speedup |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in report["cases"]:
        timing = case["timing"]
        lines.append(
            "| {candidate} | {bs} | {scenario} | {max_abs:.6g} | {mean_abs:.6g} | "
            "{base:.3f} | {cand:.3f} | {speedup:.2f}% |".format(
                candidate=case["candidate"],
                bs=case["batch_size"],
                scenario=case["scenario"],
                max_abs=float(case["parity"]["max_abs"]),
                mean_abs=float(case["parity"]["mean_abs"]),
                base=float(timing["baseline"]["avg_ms"]),
                cand=float(timing["candidate"]["avg_ms"]),
                speedup=float(timing["candidate_speedup_pct"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--candidates",
        default="arange_cache,cudnn_sdp_disabled,arange_cache+cudnn_sdp_disabled",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--scenarios", default="under100,exact100,above100,mixed")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--min-speedup-pct", type=float, default=3.0)
    parser.add_argument("--fail-on-parity", action="store_true")
    parser.add_argument("--fail-on-no-accepted", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")

    processor = _load_processor(args.model_path, str(device))
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        processor,
        max_batch_size=max(_parse_int_csv(args.batch_sizes)),
    )
    candidates = [item.strip() for item in args.candidates.split(",") if item.strip()]
    batch_sizes = _parse_int_csv(args.batch_sizes)
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]

    cases = []
    try:
        for candidate in candidates:
            for batch_size in batch_sizes:
                for scenario in scenarios:
                    case = _run_candidate_case(
                        processor=processor,
                        scheduler=scheduler,
                        candidate=candidate,
                        batch_size=batch_size,
                        scenario=scenario,
                        seed=args.seed + len(cases) * 17 + batch_size,
                        warmup=args.warmup,
                        iters=args.iters,
                        device=device,
                    )
                    cases.append(case)
                    print(
                        "candidate={candidate} bs={bs} scenario={scenario} "
                        "parity_max_abs={max_abs:.6g} speedup={speedup:.2f}%".format(
                            candidate=candidate,
                            bs=batch_size,
                            scenario=scenario,
                            max_abs=float(case["parity"]["max_abs"]),
                            speedup=float(case["timing"]["candidate_speedup_pct"]),
                        ),
                        flush=True,
                    )
    finally:
        scheduler.stop()

    candidate_summary = summarize_candidate_cases(
        cases,
        min_speedup_pct=float(args.min_speedup_pct),
    )
    report = {
        "model_path": args.model_path,
        "device": str(device),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "min_speedup_pct": float(args.min_speedup_pct),
        "cases": cases,
        "candidate_summary": candidate_summary,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, out_path.with_suffix(".md"))
    print(f"wrote {out_path}", flush=True)
    print(f"wrote {out_path.with_suffix('.md')}", flush=True)

    parity_failures = sum(1 for case in cases if not _is_exact_parity(case["parity"]))
    accepted = any(row["accepted"] for row in candidate_summary)
    if args.fail_on_parity and parity_failures:
        return 1
    if args.fail_on_no_accepted and not accepted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
