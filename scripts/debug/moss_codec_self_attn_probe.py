#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Probe exact MOSS codec self-attention plumbing candidates.

This harness narrows the codec investigation to one semantic boundary:

    audio_tokenizer.decoder[decoder_index].transformer.layers[layer_index].self_attn

It runs the normal ``processor.decode_audio_codes`` path, captures the target
self-attention output and streaming state after each call, then reruns with a
candidate patch on that same target. A candidate is useful only if full waveform
parity, target output parity, and target streaming-state parity all pass exactly.
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


def _target_self_attn(processor: Any, decoder_index: int, layer_index: int) -> Any:
    codec = getattr(processor, "audio_tokenizer", None)
    if codec is None:
        raise RuntimeError("processor has no audio_tokenizer")
    decoder = getattr(codec, "decoder", None)
    if decoder is None:
        raise RuntimeError("audio_tokenizer has no decoder")
    try:
        decoder_module = decoder[int(decoder_index)]
    except IndexError as exc:
        raise RuntimeError(f"decoder index {decoder_index} is out of range") from exc
    transformer = getattr(decoder_module, "transformer", None)
    layers = getattr(transformer, "layers", None)
    if layers is None:
        raise RuntimeError(f"decoder[{decoder_index}] has no transformer layers")
    try:
        layer = layers[int(layer_index)]
    except IndexError as exc:
        raise RuntimeError(
            f"decoder[{decoder_index}] layer index {layer_index} is out of range"
        ) from exc
    self_attn = getattr(layer, "self_attn", None)
    if self_attn is None:
        raise RuntimeError(
            f"decoder[{decoder_index}].layer[{layer_index}] has no self_attn"
        )
    return self_attn


def _to_cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to("cpu").contiguous().clone()


def _snapshot_streaming_state(module: Any) -> dict[str, Any]:
    state = getattr(module, "_streaming_state", None)
    if state is None:
        return {"is_streaming": False}

    snapshot: dict[str, Any] = {"is_streaming": True}
    for attr in ("exec_mask", "offset"):
        value = getattr(state, attr, None)
        if isinstance(value, torch.Tensor):
            snapshot[attr] = _to_cpu_tensor(value)
    offset_cpu = getattr(state, "offset_cpu", None)
    if offset_cpu is not None:
        snapshot["offset_cpu"] = int(offset_cpu)

    kv_cache = getattr(state, "kv_cache", None)
    if kv_cache is not None:
        kv_snapshot: dict[str, Any] = {}
        cache = getattr(kv_cache, "cache", None)
        if isinstance(cache, torch.Tensor):
            kv_snapshot["cache"] = _to_cpu_tensor(cache)
        end_offset = getattr(kv_cache, "end_offset", None)
        if isinstance(end_offset, torch.Tensor):
            kv_snapshot["end_offset"] = _to_cpu_tensor(end_offset)
        snapshot["kv_cache"] = kv_snapshot
    return snapshot


def _detach_output(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return _to_cpu_tensor(output)
    if isinstance(output, tuple):
        return tuple(_detach_output(item) for item in output)
    if isinstance(output, list):
        return [_detach_output(item) for item in output]
    if isinstance(output, dict):
        return {str(key): _detach_output(value) for key, value in output.items()}
    return output


def _flatten_leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, torch.Tensor):
        return [(prefix or "tensor", value)]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return [(prefix or "scalar", value)]
    if isinstance(value, dict):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten_leaves(value[key], child_prefix))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for index, item in enumerate(value):
            child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            rows.extend(_flatten_leaves(item, child_prefix))
        return rows
    return []


def _tensor_parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    shape_equal = tuple(reference.shape) == tuple(candidate.shape)
    if not shape_equal:
        return {
            "shape_equal": False,
            "max_abs": 1.0,
            "mean_abs": 1.0,
            "max_rel": 1.0,
        }
    if reference.numel() == 0:
        return {
            "shape_equal": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "max_rel": 0.0,
        }
    ref = reference.to(torch.float32)
    cand = candidate.to(torch.float32)
    diff = (ref - cand).abs()
    rel = diff / ref.abs().clamp_min(1.0e-12)
    return {
        "shape_equal": True,
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float(rel.max().item()),
    }


def _merge_parity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "shape_equal": True,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "max_rel": 0.0,
            "first_divergent": None,
        }
    shape_equal = all(bool(row["shape_equal"]) for row in rows)
    first_divergent = next(
        (
            index
            for index, row in enumerate(rows)
            if (
                not bool(row["shape_equal"])
                or float(row["max_abs"]) != 0.0
                or float(row["mean_abs"]) != 0.0
                or float(row["max_rel"]) != 0.0
            )
        ),
        None,
    )
    return {
        "shape_equal": shape_equal,
        "max_abs": max(float(row["max_abs"]) for row in rows),
        "mean_abs": statistics.fmean(float(row["mean_abs"]) for row in rows),
        "max_rel": max(float(row["max_rel"]) for row in rows),
        "first_divergent": first_divergent,
    }


def _tree_parity(reference: Any, candidate: Any) -> dict[str, Any]:
    ref_leaves = _flatten_leaves(reference)
    cand_leaves = dict(_flatten_leaves(candidate))
    rows = []
    missing = []
    for name, ref_value in ref_leaves:
        sentinel = object()
        cand_value = cand_leaves.get(name, sentinel)
        if cand_value is sentinel:
            missing.append(name)
            rows.append(
                {
                    "shape_equal": False,
                    "max_abs": 1.0,
                    "mean_abs": 1.0,
                    "max_rel": 1.0,
                }
            )
        elif isinstance(ref_value, torch.Tensor) and isinstance(
            cand_value, torch.Tensor
        ):
            rows.append(_tensor_parity(ref_value, cand_value))
        elif ref_value == cand_value:
            rows.append(
                {
                    "shape_equal": True,
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "max_rel": 0.0,
                }
            )
        else:
            rows.append(
                {
                    "shape_equal": False,
                    "max_abs": 1.0,
                    "mean_abs": 1.0,
                    "max_rel": 1.0,
                }
            )
    extra = sorted(set(cand_leaves) - {name for name, _ in ref_leaves})
    if extra:
        rows.append(
            {
                "shape_equal": False,
                "max_abs": 1.0,
                "mean_abs": 1.0,
                "max_rel": 1.0,
            }
        )
    parity = _merge_parity(rows)
    parity["missing"] = missing
    parity["extra"] = extra
    return parity


def _call_parity(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    rows = []
    for index, ref_call in enumerate(reference):
        if index >= len(candidate):
            rows.append(
                {
                    "shape_equal": False,
                    "max_abs": 1.0,
                    "mean_abs": 1.0,
                    "max_rel": 1.0,
                }
            )
            continue
        rows.append(_tree_parity(ref_call[field], candidate[index][field]))
    if len(candidate) > len(reference):
        rows.append(
            {
                "shape_equal": False,
                "max_abs": 1.0,
                "mean_abs": 1.0,
                "max_rel": 1.0,
            }
        )
    parity = _merge_parity(rows)
    parity["reference_calls"] = len(reference)
    parity["candidate_calls"] = len(candidate)
    return parity


def _is_exact_parity(parity: dict[str, Any]) -> bool:
    return (
        bool(parity.get("shape_equal"))
        and float(parity.get("max_abs", 1.0)) == 0.0
        and float(parity.get("mean_abs", 1.0)) == 0.0
        and float(parity.get("max_rel", 1.0)) == 0.0
    )


class _ScopedArangeCache:
    def __init__(self) -> None:
        self._original_arange = torch.arange
        self._active = 0
        self._cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self.stats = {"hits": 0, "misses": 0, "bypassed": 0}

    def _normalize_device(self, device: Any) -> str | None:
        if device is None:
            return None
        return str(torch.device(device))

    def _cached_arange(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if not self._active:
            return self._original_arange(*args, **kwargs)
        if "out" in kwargs and kwargs["out"] is not None:
            self.stats["bypassed"] += 1
            return self._original_arange(*args, **kwargs)
        layout = kwargs.get("layout", torch.strided)
        requires_grad = bool(kwargs.get("requires_grad", False))
        pin_memory = bool(kwargs.get("pin_memory", False))
        if layout is not torch.strided or requires_grad or pin_memory:
            self.stats["bypassed"] += 1
            return self._original_arange(*args, **kwargs)
        dtype = kwargs.get("dtype")
        device = kwargs.get("device")
        key = (
            tuple(args),
            str(dtype) if dtype is not None else None,
            self._normalize_device(device),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self.stats["hits"] += 1
            return cached
        result = self._original_arange(*args, **kwargs)
        self._cache[key] = result
        self.stats["misses"] += 1
        return result

    @contextlib.contextmanager
    def installed(self) -> Iterator[None]:
        torch.arange = self._cached_arange  # type: ignore[assignment]
        try:
            yield
        finally:
            torch.arange = self._original_arange  # type: ignore[assignment]

    @contextlib.contextmanager
    def active(self) -> Iterator[None]:
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1


@contextlib.contextmanager
def _target_probe_patch(
    target: Any,
    candidate: str,
) -> Iterator[tuple[list[dict[str, Any]], dict[str, Any]]]:
    original_forward = target.forward
    calls: list[dict[str, Any]] = []
    patch_stats: dict[str, Any] = {"candidate": candidate}

    if candidate == "target_identity":
        arange_cache = None
    elif candidate == "target_arange_cache":
        arange_cache = _ScopedArangeCache()
    else:
        raise ValueError(f"unknown candidate {candidate!r}")

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if arange_cache is None:
            output = original_forward(*args, **kwargs)
        else:
            with arange_cache.active():
                output = original_forward(*args, **kwargs)
        calls.append(
            {
                "output": _detach_output(output),
                "state": _snapshot_streaming_state(target),
            }
        )
        return output

    context = (
        arange_cache.installed()
        if arange_cache is not None
        else contextlib.nullcontext()
    )
    with context:
        target.forward = wrapped  # type: ignore[method-assign]
        try:
            yield calls, patch_stats
        finally:
            target.forward = original_forward  # type: ignore[method-assign]
            if arange_cache is not None:
                patch_stats["arange_cache"] = dict(arange_cache.stats)


def _run_decode_with_target_probe(
    *,
    scheduler: MossTTSLocalStreamingVocoderScheduler,
    codes_list: list[torch.Tensor],
    target: Any,
    candidate: str,
) -> dict[str, Any]:
    with _target_probe_patch(target, candidate) as (calls, patch_stats):
        wavs = _scheduler_processor_decode(scheduler, codes_list)
    return {"wavs": wavs, "calls": calls, "patch_stats": patch_stats}


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


def _candidate_decode_fn(
    *,
    scheduler: MossTTSLocalStreamingVocoderScheduler,
    codes_list: list[torch.Tensor],
    target: Any,
    candidate: str,
) -> Callable[[], list[torch.Tensor]]:
    if candidate == "target_identity":
        return lambda: _scheduler_processor_decode(scheduler, codes_list)

    def run_candidate() -> list[torch.Tensor]:
        with _target_probe_patch(target, candidate):
            return _scheduler_processor_decode(scheduler, codes_list)

    return run_candidate


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
    target: Any,
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

    with torch.inference_mode():
        baseline = _run_decode_with_target_probe(
            scheduler=scheduler,
            codes_list=codes_list,
            target=target,
            candidate="target_identity",
        )
        candidate_run = _run_decode_with_target_probe(
            scheduler=scheduler,
            codes_list=codes_list,
            target=target,
            candidate=candidate,
        )
        waveform_parity = _parity(baseline["wavs"], candidate_run["wavs"])
        output_parity = _call_parity(
            baseline["calls"], candidate_run["calls"], "output"
        )
        state_parity = _call_parity(baseline["calls"], candidate_run["calls"], "state")
        baseline_timing = _time_block(
            fn=lambda: _scheduler_processor_decode(scheduler, codes_list),
            device=device,
            warmup=warmup,
            iters=iters,
        )
        candidate_timing = _time_block(
            fn=_candidate_decode_fn(
                scheduler=scheduler,
                codes_list=codes_list,
                target=target,
                candidate=candidate,
            ),
            device=device,
            warmup=warmup,
            iters=iters,
        )

    return {
        "candidate": candidate,
        "batch_size": batch_size,
        "scenario": scenario,
        "lengths": _case_lengths(batch_size, scenario),
        "waveform_parity": waveform_parity,
        "self_attn_output_parity": output_parity,
        "self_attn_state_parity": state_parity,
        "patch_stats": candidate_run["patch_stats"],
        "timing": _compare_timing(
            baseline=baseline_timing,
            candidate=candidate_timing,
        ),
    }


def _case_accepted(case: dict[str, Any]) -> bool:
    return (
        _is_exact_parity(case["waveform_parity"])
        and _is_exact_parity(case["self_attn_output_parity"])
        and _is_exact_parity(case["self_attn_state_parity"])
    )


def summarize_candidate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(str(case["candidate"]), []).append(case)
    rows = []
    for candidate, candidate_cases in grouped.items():
        accepted_cases = sum(_case_accepted(case) for case in candidate_cases)
        speedups = [
            float(case["timing"]["candidate_speedup_pct"])
            for case in candidate_cases
            if _case_accepted(case)
        ]
        rows.append(
            {
                "candidate": candidate,
                "cases": len(candidate_cases),
                "accepted_cases": accepted_cases,
                "rejected_cases": len(candidate_cases) - accepted_cases,
                "avg_speedup_pct": statistics.fmean(speedups) if speedups else 0.0,
                "min_speedup_pct": min(speedups) if speedups else 0.0,
                "max_speedup_pct": max(speedups) if speedups else 0.0,
                "accepted": accepted_cases == len(candidate_cases)
                and bool(candidate_cases),
            }
        )
    rows.sort(key=lambda row: (not row["accepted"], -float(row["avg_speedup_pct"])))
    return rows


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# MOSS Codec Self-Attention Probe",
        "",
        f"model: `{report['model_path']}`",
        f"device: `{report['device']}`",
        f"target: `decoder[{report['decoder_index']}].layer[{report['layer_index']}].self_attn`",
        f"warmup: `{report['warmup']}`",
        f"iters: `{report['iters']}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate | cases | accepted | avg speedup | min speedup | max speedup |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["candidate_summary"]:
        lines.append(
            "| {candidate} | {cases} | {accepted_cases}/{cases} | {avg:.2f}% | "
            "{minv:.2f}% | {maxv:.2f}% |".format(
                candidate=row["candidate"],
                cases=row["cases"],
                accepted_cases=row["accepted_cases"],
                avg=float(row["avg_speedup_pct"]),
                minv=float(row["min_speedup_pct"]),
                maxv=float(row["max_speedup_pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| candidate | bs | scenario | waveform max_abs | output max_abs | "
            "state max_abs | calls | speedup |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in report["cases"]:
        output_parity = case["self_attn_output_parity"]
        state_parity = case["self_attn_state_parity"]
        lines.append(
            "| {candidate} | {bs} | {scenario} | {wave:.6g} | {out:.6g} | "
            "{state:.6g} | {calls}/{cand_calls} | {speedup:.2f}% |".format(
                candidate=case["candidate"],
                bs=case["batch_size"],
                scenario=case["scenario"],
                wave=float(case["waveform_parity"]["max_abs"]),
                out=float(output_parity["max_abs"]),
                state=float(state_parity["max_abs"]),
                calls=output_parity["reference_calls"],
                cand_calls=output_parity["candidate_calls"],
                speedup=float(case["timing"]["candidate_speedup_pct"]),
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
    parser.add_argument("--decoder-index", type=int, default=0)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--candidates", default="target_identity,target_arange_cache")
    parser.add_argument("--batch-sizes", default="1")
    parser.add_argument("--scenarios", default="exact100")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--fail-on-parity", action="store_true")
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
    target = _target_self_attn(processor, args.decoder_index, args.layer_index)

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
                        target=target,
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
                        "waveform_max_abs={wave:.6g} output_max_abs={out:.6g} "
                        "state_max_abs={state:.6g} speedup={speedup:.2f}%".format(
                            candidate=candidate,
                            bs=batch_size,
                            scenario=scenario,
                            wave=float(case["waveform_parity"]["max_abs"]),
                            out=float(case["self_attn_output_parity"]["max_abs"]),
                            state=float(case["self_attn_state_parity"]["max_abs"]),
                            speedup=float(case["timing"]["candidate_speedup_pct"]),
                        ),
                        flush=True,
                    )
    finally:
        scheduler.stop()

    report = {
        "model_path": args.model_path,
        "device": str(device),
        "decoder_index": int(args.decoder_index),
        "layer_index": int(args.layer_index),
        "warmup": int(args.warmup),
        "iters": int(args.iters),
        "cases": cases,
        "candidate_summary": summarize_candidate_cases(cases),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    _write_markdown(report, out_path.with_suffix(".md"))
    print(f"wrote {out_path}", flush=True)
    print(f"wrote {out_path.with_suffix('.md')}", flush=True)

    if args.fail_on_parity and any(not _case_accepted(case) for case in cases):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
