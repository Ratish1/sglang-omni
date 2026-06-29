# SPDX-License-Identifier: Apache-2.0
"""Compare MOSS-TTS Local non-stream vocoder single vs batched decode.

This is an offline correctness gate for batched codec decode. It loads the real
MOSS-TTS Local vocoder stage, decodes the same code rows one-by-one and in
batches, then reports waveform drift per length/batch-size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from sglang_omni.models.moss_tts_local.stages import create_vocoder_executor

DEFAULT_MODEL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
DEFAULT_LENGTHS = (
    1,
    4,
    5,
    8,
    9,
    10,
    11,
    12,
    13,
    20,
    22,
    24,
    25,
    31,
    32,
    55,
    79,
    91,
    117,
)
DEFAULT_BATCH_SIZES = (2, 4, 8)
SECONDS_PER_MOSS_LOCAL_FRAME = 0.08
BACKENDS = ("packed", "hf")
MODES = ("ragged", "same_length", "duplicate", "length_bucket")


def _parse_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("all values must be positive")
    return out


def _parse_modes(value: str) -> list[str]:
    modes = [part.strip() for part in value.split(",") if part.strip()]
    if not modes:
        raise argparse.ArgumentTypeError("expected at least one mode")
    invalid = [mode for mode in modes if mode not in MODES]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid modes {invalid}; expected values from {MODES}"
        )
    return modes


def _lengths_from_generated_json(path: Path, limit: int) -> list[int]:
    with path.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of generated rows")
    lengths = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        tokens = row.get("completion_tokens")
        if tokens is not None:
            length = int(tokens)
        else:
            duration = row.get("audio_duration_s")
            if duration is None:
                continue
            length = max(1, round(float(duration) / SECONDS_PER_MOSS_LOCAL_FRAME))
        lengths.append(length)
        if len(lengths) >= limit:
            break
    if not lengths:
        raise ValueError(
            f"{path} did not contain completion_tokens or audio_duration_s"
        )
    return lengths


def _find_audio_codes(obj: Any) -> Any | None:
    if isinstance(obj, dict):
        if "audio_codes" in obj:
            return obj["audio_codes"]
        for value in obj.values():
            found = _find_audio_codes(value)
            if found is not None:
                return found
    return None


def _codes_from_json(path: Path, limit: int, n_vq: int) -> list[torch.Tensor]:
    with path.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list")

    out = []
    for row in rows:
        codes = _find_audio_codes(row)
        if codes is None:
            continue
        tensor = torch.as_tensor(codes, dtype=torch.long)
        if tensor.ndim != 2:
            raise ValueError(f"audio_codes must be 2D, got shape {tuple(tensor.shape)}")
        if int(tensor.shape[1]) < n_vq:
            raise ValueError(
                f"audio_codes must have at least {n_vq} columns, got {tensor.shape[1]}"
            )
        out.append(tensor[:, :n_vq].contiguous())
        if len(out) >= limit:
            break
    if not out:
        raise ValueError(f"{path} did not contain any audio_codes rows")
    return out


def _make_codes(
    lengths: list[int],
    *,
    n_vq: int,
    vocab_size: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return [
        torch.randint(
            low=0,
            high=vocab_size,
            size=(length, n_vq),
            generator=generator,
            dtype=torch.long,
        )
        for length in lengths
    ]


def _codes_for_mode(
    *,
    mode: str,
    source_codes: list[torch.Tensor],
    batch_size: int,
    n_vq: int,
    vocab_size: int,
    seed: int,
) -> list[torch.Tensor]:
    if mode == "ragged":
        return source_codes
    if mode == "same_length":
        max_items = min(len(source_codes), max(batch_size * 4, batch_size))
        length = int(source_codes[min(len(source_codes) - 1, 1)].shape[0])
        return _make_codes(
            [length] * max_items,
            n_vq=n_vq,
            vocab_size=vocab_size,
            seed=seed + 10_000 + batch_size,
        )
    if mode == "duplicate":
        base = source_codes[min(len(source_codes) - 1, 0)]
        return [base.clone() for _ in range(max(batch_size, 2))]
    if mode == "length_bucket":
        buckets: dict[int, list[torch.Tensor]] = {}
        for codes in source_codes:
            buckets.setdefault(int(codes.shape[0]), []).append(codes)
        largest = max(buckets.values(), key=len)
        if len(largest) < 2:
            return []
        return largest[: max(batch_size, min(len(largest), batch_size * 4))]
    raise ValueError(f"unknown parity mode {mode!r}")


def _compare(ref: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    same_shape = tuple(ref.shape) == tuple(candidate.shape)
    has_nan = bool(torch.isnan(ref).any().item() or torch.isnan(candidate).any().item())
    if same_shape:
        diff = (ref - candidate).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    else:
        max_abs = None
        mean_abs = None
    return {
        "same_shape": same_shape,
        "shape_single": list(ref.shape),
        "shape_batch": list(candidate.shape),
        "has_nan": has_nan,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def _worst_case(
    case_results: list[dict[str, Any]]
) -> tuple[float, dict[str, Any] | None]:
    worst_max_abs = 0.0
    worst_case = None
    for case in case_results:
        max_abs = case["max_abs"]
        if max_abs is not None and max_abs > worst_max_abs:
            worst_max_abs = float(max_abs)
            worst_case = {
                key: value
                for key, value in case.items()
                if key
                in {
                    "mode",
                    "backend",
                    "comparison",
                    "batch_size",
                    "index",
                    "frames",
                }
            }
    return worst_max_abs, worst_case


def _write_reports(result: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = output.with_suffix(".md")

    lines = [
        "# MOSS-TTS Local Vocoder Batch Parity",
        "",
        "## Summary",
        "",
        f"- cases: `{result['cases']}`",
        f"- all_same_shape: `{result['all_same_shape']}`",
        f"- any_nan: `{result['any_nan']}`",
        f"- worst_max_abs: `{result['worst_max_abs']}`",
        f"- worst_case: `{result['worst_case']}`",
        "",
        "## Mode Summary",
        "",
        "| mode | backend | comparison | cases | worst max_abs |",
        "|---|---|---|---:|---:|",
    ]
    for row in result["summary_rows"]:
        lines.append(
            "| {mode} | {backend} | {comparison} | {cases} | {worst_max_abs} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| mode | backend | comparison | batch_size | index | frames | same shape | max_abs | mean_abs |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in result["case_results"]:
        lines.append(
            "| {mode} | {backend} | {comparison} | {batch_size} | {index} | {frames} | {same_shape} | {max_abs} | {mean_abs} |".format(
                **case
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `single_vs_batch` compares one-by-one decode against a batched decode on the same backend.",
            "- `packed_vs_hf_single` compares SGLang packed attention against the original HF decoder for single-row decode.",
            "- `packed_vs_hf_batch` compares SGLang packed attention against the original HF decoder for batched decode.",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_scheduler_dtype(scheduler: Any, dtype: str) -> None:
    if dtype == "loaded":
        return
    if dtype == "float32":
        scheduler._codec.float()
        scheduler._nonstream_decoder.float()
        return
    if dtype == "bfloat16":
        scheduler._codec.bfloat16()
        scheduler._nonstream_decoder.bfloat16()
        return
    raise ValueError(f"unknown dtype mode {dtype!r}")


def _decode_backend(
    scheduler: Any,
    codes: list[torch.Tensor],
    *,
    backend: str,
) -> list[torch.Tensor]:
    if backend == "packed":
        decoder = scheduler._nonstream_decoder
    elif backend == "hf":
        decoder = None
    else:
        raise ValueError(f"unknown backend {backend!r}")

    original_decoder = scheduler._codec.decoder
    if decoder is not None:
        scheduler._codec.decoder = decoder
    try:
        return scheduler._decode_codes_rows_nonstream(codes)
    finally:
        scheduler._codec.decoder = original_decoder


def _append_single_vs_batch_cases(
    case_results: list[dict[str, Any]],
    *,
    scheduler: Any,
    mode: str,
    backend: str,
    codes: list[torch.Tensor],
    batch_size: int,
) -> None:
    single = [_decode_backend(scheduler, [row], backend=backend)[0] for row in codes]
    for start in range(0, len(codes), batch_size):
        chunk = codes[start : start + batch_size]
        if len(chunk) < 2:
            continue
        decoded = _decode_backend(scheduler, chunk, backend=backend)
        for offset, candidate in enumerate(decoded):
            index = start + offset
            case_results.append(
                {
                    "mode": mode,
                    "backend": backend,
                    "comparison": "single_vs_batch",
                    "batch_size": len(chunk),
                    "index": index,
                    "frames": int(codes[index].shape[0]),
                    **_compare(single[index], candidate),
                }
            )


def _append_backend_cases(
    case_results: list[dict[str, Any]],
    *,
    scheduler: Any,
    mode: str,
    codes: list[torch.Tensor],
    batch_size: int,
) -> None:
    for index, row in enumerate(codes):
        packed = _decode_backend(scheduler, [row], backend="packed")[0]
        hf = _decode_backend(scheduler, [row], backend="hf")[0]
        case_results.append(
            {
                "mode": mode,
                "backend": "packed_vs_hf",
                "comparison": "packed_vs_hf_single",
                "batch_size": 1,
                "index": index,
                "frames": int(row.shape[0]),
                **_compare(hf, packed),
            }
        )

    for start in range(0, len(codes), batch_size):
        chunk = codes[start : start + batch_size]
        if len(chunk) < 2:
            continue
        packed_batch = _decode_backend(scheduler, chunk, backend="packed")
        hf_batch = _decode_backend(scheduler, chunk, backend="hf")
        for offset, (hf, packed) in enumerate(zip(hf_batch, packed_batch)):
            index = start + offset
            case_results.append(
                {
                    "mode": mode,
                    "backend": "packed_vs_hf",
                    "comparison": "packed_vs_hf_batch",
                    "batch_size": len(chunk),
                    "index": index,
                    "frames": int(codes[index].shape[0]),
                    **_compare(hf, packed),
                }
            )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.disable_cudnn_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    if args.codes_json is not None:
        source_codes = _codes_from_json(args.codes_json, args.max_generated, args.n_vq)
    else:
        lengths = list(args.lengths)
        if args.generated_json is not None:
            lengths = _lengths_from_generated_json(
                args.generated_json, args.max_generated
            )
        source_codes = _make_codes(
            lengths,
            n_vq=args.n_vq,
            vocab_size=args.vocab_size,
            seed=args.seed,
        )

    scheduler = create_vocoder_executor(
        args.model,
        device=args.device,
        codec_model_path=args.codec_model,
        max_batch_size=max(args.batch_sizes),
        max_batch_wait_ms=0,
        cuda_graph=args.cuda_graph,
    )
    _set_scheduler_dtype(scheduler, args.dtype)

    case_results: list[dict[str, Any]] = []
    modes = args.modes or list(MODES)
    for batch_size in args.batch_sizes:
        for mode in modes:
            codes = _codes_for_mode(
                mode=mode,
                source_codes=source_codes,
                batch_size=batch_size,
                n_vq=args.n_vq,
                vocab_size=args.vocab_size,
                seed=args.seed,
            )
            if len(codes) < 2:
                continue
            if args.backends in ("packed", "both"):
                _append_single_vs_batch_cases(
                    case_results,
                    scheduler=scheduler,
                    mode=mode,
                    backend="packed",
                    codes=codes,
                    batch_size=batch_size,
                )
            if args.backends in ("hf", "both"):
                _append_single_vs_batch_cases(
                    case_results,
                    scheduler=scheduler,
                    mode=mode,
                    backend="hf",
                    codes=codes,
                    batch_size=batch_size,
                )
            if args.compare_backends:
                _append_backend_cases(
                    case_results,
                    scheduler=scheduler,
                    mode=mode,
                    codes=codes,
                    batch_size=batch_size,
                )

    if not case_results:
        raise RuntimeError("No parity cases were produced by the selected modes")

    worst_max_abs, worst_case = _worst_case(case_results)
    summary_rows = []
    summary_keys = sorted(
        {(case["mode"], case["backend"], case["comparison"]) for case in case_results}
    )
    for mode, backend, comparison in summary_keys:
        subset = [
            case
            for case in case_results
            if case["mode"] == mode
            and case["backend"] == backend
            and case["comparison"] == comparison
        ]
        summary_rows.append(
            {
                "mode": mode,
                "backend": backend,
                "comparison": comparison,
                "cases": len(subset),
                "worst_max_abs": _worst_case(subset)[0],
            }
        )

    result = {
        "model": args.model,
        "codec_model": args.codec_model,
        "device": args.device,
        "dtype": args.dtype,
        "n_vq": args.n_vq,
        "vocab_size": args.vocab_size,
        "lengths": [int(codes.shape[0]) for codes in source_codes],
        "batch_sizes": args.batch_sizes,
        "cases": len(case_results),
        "all_same_shape": all(case["same_shape"] for case in case_results),
        "any_nan": any(case["has_nan"] for case in case_results),
        "worst_max_abs": worst_max_abs,
        "worst_case": worst_case,
        "summary_rows": summary_rows,
        "case_results": case_results,
    }
    if args.output is not None:
        _write_reports(result, args.output)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--codec-model", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-vq", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dtype",
        choices=("loaded", "float32", "bfloat16"),
        default="loaded",
        help="Optionally cast the codec before parity checks.",
    )
    parser.add_argument(
        "--backends",
        choices=("packed", "hf", "both"),
        default="packed",
        help="Backend(s) for single-vs-batch comparisons.",
    )
    parser.add_argument(
        "--compare-backends",
        action="store_true",
        help="Also compare packed SGLang decoder output against the original HF decoder.",
    )
    parser.add_argument(
        "--modes",
        type=_parse_modes,
        default=None,
        help=f"Comma-separated subset of {MODES}; default runs all.",
    )
    parser.add_argument("--lengths", type=_parse_ints, default=list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--batch-sizes",
        type=_parse_ints,
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument(
        "--generated-json",
        type=Path,
        default=None,
        help="Optional generated.json. Uses completion_tokens or audio_duration_s as frame lengths.",
    )
    parser.add_argument(
        "--codes-json",
        type=Path,
        default=None,
        help="Optional JSON containing real audio_codes rows. Takes precedence over generated-json.",
    )
    parser.add_argument("--max-generated", type=int, default=128)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--disable-cudnn-sdp", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    result = run(_build_parser().parse_args())
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "cases",
                    "all_same_shape",
                    "any_nan",
                    "worst_max_abs",
                    "worst_case",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
