#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Same-process MOSS-TTS Local codec parity and timing harness.

This debug-branch tool isolates the non-streaming codec question from server
batching, ASR, request profiling, and scheduler noise:

    same audio codes -> processor.decode_audio_codes vs exact_session
    same audio codes -> processor.decode_audio_codes vs direct_chunked

Use this before making any performance claim for the non-streaming codec path.
The correctness gate is exact waveform parity. The performance gate is
same-process wall time with CUDA synchronized around every measured decode.
Use ``--include-direct-chunked`` for the parity-safe direct tokenizer control.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
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

from sglang_omni.models.moss_tts_local.streaming_vocoder import (  # noqa: E402
    MossTTSLocalStreamingVocoderScheduler,
)


def _parse_int_csv(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _load_processor(model_path: str, device: str) -> Any:
    from sglang_omni.models.moss_tts_local.stages import _load_moss_tts_local_processor

    return _load_moss_tts_local_processor(model_path, device=device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    frac = index - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


def _audio_vocab_size(processor: Any) -> int:
    model_config = getattr(processor, "model_config", None)
    return int(getattr(model_config, "audio_vocab_size", 1024) or 1024)


def _n_vq(processor: Any) -> int:
    model_config = getattr(processor, "model_config", None)
    return int(getattr(model_config, "n_vq", 12) or 12)


def _case_lengths(batch_size: int, scenario: str) -> list[int]:
    mixed = [37, 100, 217, 163, 5, 99, 101, 260]
    if scenario == "under100":
        base = [37, 63, 81, 99, 11, 25, 49, 75]
    elif scenario == "exact100":
        base = [100] * 8
    elif scenario == "above100":
        base = [217, 163, 301, 121, 250, 199, 145, 333]
    elif scenario == "mixed":
        base = mixed
    else:
        raise ValueError(f"unknown scenario {scenario!r}")
    return [base[i % len(base)] for i in range(batch_size)]


def _make_codes(
    *,
    processor: Any,
    batch_size: int,
    scenario: str,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vocab_size = _audio_vocab_size(processor)
    n_vq = _n_vq(processor)
    rows = []
    for frames in _case_lengths(batch_size, scenario):
        rows.append(
            torch.randint(
                0,
                vocab_size,
                (frames, n_vq),
                dtype=torch.long,
                generator=generator,
            )
        )
    return rows


def _to_wave_list(wavs: list[Any]) -> list[torch.Tensor]:
    return [
        torch.as_tensor(wav).detach().to("cpu", torch.float32).contiguous()
        for wav in wavs
    ]


def _scheduler_processor_decode(
    scheduler: MossTTSLocalStreamingVocoderScheduler,
    codes_list: list[torch.Tensor],
) -> list[torch.Tensor]:
    return scheduler._decode_codes_rows_processor(  # noqa: SLF001
        codes_list,
        request_ids=[f"bench-{i}" for i in range(len(codes_list))],
    )


def _exact_session_decode(
    scheduler: MossTTSLocalStreamingVocoderScheduler,
    codes_list: list[torch.Tensor],
) -> list[torch.Tensor]:
    return scheduler._decode_codes_rows_exact_session(  # noqa: SLF001
        codes_list,
        request_ids=[f"bench-{i}" for i in range(len(codes_list))],
    )


def _direct_chunked_decode(
    scheduler: MossTTSLocalStreamingVocoderScheduler,
    codes_list: list[torch.Tensor],
) -> list[torch.Tensor]:
    return scheduler._decode_codes_rows_direct_batch(  # noqa: SLF001
        codes_list,
        request_ids=[f"bench-{i}" for i in range(len(codes_list))],
        chunk_duration=scheduler._nonstream_chunk_duration,  # noqa: SLF001
        actual_path="direct_chunked",
    )


def _parity(
    reference: list[torch.Tensor],
    candidate: list[torch.Tensor],
) -> dict[str, Any]:
    shape_equal = len(reference) == len(candidate) and all(
        tuple(ref.shape) == tuple(cand.shape) for ref, cand in zip(reference, candidate)
    )
    max_abs = 0.0
    mean_abs_weighted = 0.0
    max_rel = 0.0
    total = 0
    first_divergent = None
    for index, (ref, cand) in enumerate(zip(reference, candidate)):
        if tuple(ref.shape) != tuple(cand.shape):
            if first_divergent is None:
                first_divergent = index
            continue
        diff = (ref - cand).abs()
        if diff.numel() == 0:
            continue
        item_max = float(diff.max().item())
        if item_max != 0.0 and first_divergent is None:
            first_divergent = index
        max_abs = max(max_abs, item_max)
        mean_abs_weighted += float(diff.sum().item())
        total += int(diff.numel())
        rel = diff / ref.abs().clamp_min(1.0e-12)
        max_rel = max(max_rel, float(rel.max().item()))
    return {
        "shape_equal": shape_equal,
        "length_equal": shape_equal,
        "max_abs": max_abs,
        "mean_abs": mean_abs_weighted / float(total) if total else 0.0,
        "max_rel": max_rel,
        "first_divergent_sample": first_divergent,
    }


@contextlib.contextmanager
def _count_decode_frame(codec: Any) -> Iterator[Callable[[], int]]:
    original = getattr(codec, "_decode_frame")
    count = 0

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        count += 1
        return original(*args, **kwargs)

    setattr(codec, "_decode_frame", wrapped)
    try:
        yield lambda: count
    finally:
        setattr(codec, "_decode_frame", original)


def _time_path(
    *,
    name: str,
    fn: Callable[[], list[torch.Tensor]],
    codec: Any,
    device: torch.device,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    _sync(device)

    samples = []
    with _count_decode_frame(codec) as frame_count:
        for _ in range(iters):
            _sync(device)
            start = time.perf_counter()
            fn()
            _sync(device)
            samples.append((time.perf_counter() - start) * 1000.0)
        calls = frame_count()

    return {
        "path": name,
        "avg_ms": statistics.fmean(samples),
        "p50_ms": _percentile(samples, 0.50),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "iters": iters,
        "decode_frame_calls_total": calls,
        "decode_frame_calls_per_iter": calls / float(iters),
    }


def _disable_request_profiling_env() -> None:
    for name in (
        "SGLANG_OMNI_PROFILE_REQUESTS",
        "SGLANG_MOSS_TTS_LOCAL_VOCODER_EVENTS",
        "SGLANG_MOSS_TTS_LOCAL_VOCODER_DEEP_PROFILE",
    ):
        os.environ.pop(name, None)


def _run_case(
    *,
    processor: Any,
    scheduler: MossTTSLocalStreamingVocoderScheduler,
    batch_size: int,
    scenario: str,
    seed: int,
    warmup: int,
    iters: int,
    include_direct_chunked: bool,
    device: torch.device,
) -> dict[str, Any]:
    codes_list = _make_codes(
        processor=processor,
        batch_size=batch_size,
        scenario=scenario,
        seed=seed,
    )
    codec = processor.audio_tokenizer

    with torch.inference_mode():
        processor_wavs = _scheduler_processor_decode(scheduler, codes_list)
        exact_wavs = _exact_session_decode(scheduler, codes_list)
        scheduler._close_offline_session()  # noqa: SLF001
        direct_wavs = (
            _direct_chunked_decode(scheduler, codes_list)
            if include_direct_chunked and hasattr(codec, "batch_decode")
            else None
        )
        _sync(device)

        timings = [
            _time_path(
                name="processor",
                fn=lambda: _scheduler_processor_decode(scheduler, codes_list),
                codec=codec,
                device=device,
                warmup=warmup,
                iters=iters,
            ),
            _time_path(
                name="exact_session",
                fn=lambda: _exact_session_decode(scheduler, codes_list),
                codec=codec,
                device=device,
                warmup=warmup,
                iters=iters,
            ),
        ]
        scheduler._close_offline_session()  # noqa: SLF001
        if direct_wavs is not None:
            timings.append(
                _time_path(
                    name="direct_chunked",
                    fn=lambda: _direct_chunked_decode(scheduler, codes_list),
                    codec=codec,
                    device=device,
                    warmup=warmup,
                    iters=iters,
                )
            )

    parity = {"exact_session": _parity(processor_wavs, exact_wavs)}
    if direct_wavs is not None:
        parity["direct_chunked"] = _parity(processor_wavs, direct_wavs)

    return {
        "batch_size": batch_size,
        "scenario": scenario,
        "lengths": [int(codes.shape[0]) for codes in codes_list],
        "parity": parity,
        "timings": timings,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# MOSS Local Codec Same-Process Bench",
        "",
        f"model: `{report['model_path']}`",
        f"device: `{report['device']}`",
        f"warmup: `{report['warmup']}`",
        f"iters: `{report['iters']}`",
        "",
        "## Cases",
        "",
        "| bs | scenario | path | parity max_abs | parity mean_abs | avg ms | "
        "p50 ms | p95 ms | frame calls/iter |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        parity_by_path = case["parity"]
        for timing in case["timings"]:
            path_name = timing["path"]
            parity = (
                {"max_abs": 0.0, "mean_abs": 0.0}
                if path_name == "processor"
                else parity_by_path[path_name]
            )
            lines.append(
                "| {bs} | {scenario} | {path} | {max_abs:.6g} | {mean_abs:.6g} | "
                "{avg:.3f} | {p50:.3f} | {p95:.3f} | {calls:.2f} |".format(
                    bs=case["batch_size"],
                    scenario=case["scenario"],
                    path=path_name,
                    max_abs=float(parity["max_abs"]),
                    mean_abs=float(parity["mean_abs"]),
                    avg=float(timing["avg_ms"]),
                    p50=float(timing["p50_ms"]),
                    p95=float(timing["p95_ms"]),
                    calls=float(timing["decode_frame_calls_per_iter"]),
                )
            )
    path.write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument(
        "--scenarios",
        default="under100,exact100,above100,mixed",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--include-direct-chunked", action="store_true")
    parser.add_argument("--fail-on-parity", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    _disable_request_profiling_env()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA but torch.cuda.is_available() is false")

    processor = _load_processor(args.model_path, str(device))
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        processor,
        max_batch_size=max(_parse_int_csv(args.batch_sizes)),
    )

    batch_sizes = _parse_int_csv(args.batch_sizes)
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]

    cases = []
    failures = 0
    try:
        for batch_size in batch_sizes:
            for scenario in scenarios:
                case = _run_case(
                    processor=processor,
                    scheduler=scheduler,
                    batch_size=batch_size,
                    scenario=scenario,
                    seed=args.seed + batch_size * 1000 + len(cases),
                    warmup=args.warmup,
                    iters=args.iters,
                    include_direct_chunked=args.include_direct_chunked,
                    device=device,
                )
                cases.append(case)
                exact_parity = case["parity"]["exact_session"]
                failures += int(
                    not exact_parity["shape_equal"]
                    or float(exact_parity["max_abs"]) != 0.0
                    or float(exact_parity["mean_abs"]) != 0.0
                    or float(exact_parity["max_rel"]) != 0.0
                )
                timing_summary = {
                    item["path"]: round(float(item["avg_ms"]), 3)
                    for item in case["timings"]
                }
                print(
                    "bs={bs} scenario={scenario} parity_max_abs={max_abs:.6g} "
                    "timing_avg_ms={timing}".format(
                        bs=batch_size,
                        scenario=scenario,
                        max_abs=float(exact_parity["max_abs"]),
                        timing=timing_summary,
                    ),
                    flush=True,
                )
    finally:
        scheduler.stop()

    report = {
        "model_path": args.model_path,
        "device": str(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "include_direct_chunked": bool(args.include_direct_chunked),
        "failures": failures,
        "cases": cases,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    _write_markdown(report, out_path.with_suffix(".md"))
    print(f"wrote {out_path}", flush=True)
    print(f"wrote {out_path.with_suffix('.md')}", flush=True)

    if args.fail_on_parity and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
