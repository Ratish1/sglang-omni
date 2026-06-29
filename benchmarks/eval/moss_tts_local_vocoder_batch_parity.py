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


def _parse_ints(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in out):
        raise argparse.ArgumentTypeError("all values must be positive")
    return out


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
        "## Cases",
        "",
        "| batch_size | index | frames | same shape | max_abs | mean_abs |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["case_results"]:
        lines.append(
            "| {batch_size} | {index} | {frames} | {same_shape} | {max_abs} | {mean_abs} |".format(
                **case
            )
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.disable_cudnn_sdp and torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    lengths = list(args.lengths)
    if args.generated_json is not None:
        lengths = _lengths_from_generated_json(args.generated_json, args.max_generated)
    codes = _make_codes(
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
    single = [scheduler._decode_codes_rows([row])[0] for row in codes]

    case_results: list[dict[str, Any]] = []
    worst_max_abs = 0.0
    worst_case = None
    for batch_size in args.batch_sizes:
        for start in range(0, len(codes), batch_size):
            chunk = codes[start : start + batch_size]
            decoded = scheduler._decode_codes_rows(chunk)
            for offset, candidate in enumerate(decoded):
                index = start + offset
                comparison = _compare(single[index], candidate)
                max_abs = comparison["max_abs"]
                if max_abs is not None and max_abs > worst_max_abs:
                    worst_max_abs = max_abs
                    worst_case = {
                        "batch_size": batch_size,
                        "index": index,
                        "frames": lengths[index],
                    }
                case_results.append(
                    {
                        "batch_size": batch_size,
                        "index": index,
                        "frames": lengths[index],
                        **comparison,
                    }
                )

    result = {
        "model": args.model,
        "codec_model": args.codec_model,
        "device": args.device,
        "n_vq": args.n_vq,
        "vocab_size": args.vocab_size,
        "lengths": lengths,
        "batch_sizes": args.batch_sizes,
        "cases": len(case_results),
        "all_same_shape": all(case["same_shape"] for case in case_results),
        "any_nan": any(case["has_nan"] for case in case_results),
        "worst_max_abs": worst_max_abs,
        "worst_case": worst_case,
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
