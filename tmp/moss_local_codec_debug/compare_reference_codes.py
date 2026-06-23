# SPDX-License-Identifier: Apache-2.0
"""Compare upstream MOSS processor reference codes with the separated codec path.

This debug utility is for the MOSS-TTS Local codec-ownership branch. It checks
whether the separately loaded audio tokenizer preserves the AR prompt
conditioning produced by the upstream processor's reference encode path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
    resolve_moss_checkpoint,
)
from sglang_omni.models.moss_tts_local.audio_tokenizer import (
    load_moss_tts_local_audio_tokenizer,
)
from sglang_omni.models.moss_tts_local.stages import (
    _normalize_processor_config,
    _resolve_audio_tokenizer_model_path,
)


@dataclass(frozen=True)
class RefCase:
    sample_id: str
    ref_audio: str
    target_text: str | None = None


def _load_processor_with_codec(model_path: str, device: str) -> Any:
    checkpoint_dir = resolve_moss_checkpoint(model_path)
    with moss_transformers_processor_compat():
        processor_cls = load_moss_processor_class(checkpoint_dir)
        processor = processor_cls.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
    _normalize_processor_config(processor)
    audio_tokenizer = processor.audio_tokenizer
    audio_tokenizer.eval()
    audio_tokenizer.to(device)
    return processor


def _load_processor_without_codec(model_path: str) -> Any:
    from transformers import AutoConfig, AutoTokenizer

    checkpoint_dir = resolve_moss_checkpoint(model_path)
    with moss_transformers_processor_compat():
        processor_cls = load_moss_processor_class(checkpoint_dir)
        model_config = AutoConfig.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
        processor = processor_cls(
            tokenizer=tokenizer,
            audio_tokenizer=None,
            model_config=model_config,
        )
    _normalize_processor_config(processor)
    return processor


def _cases_from_meta(
    meta: str,
    *,
    sample_ids: list[str],
    max_samples: int | None,
    lang: str,
) -> list[RefCase]:
    samples = load_seedtts_samples(meta, max_samples=max_samples, split=lang)
    by_id: dict[str, SampleInput] = {sample.sample_id: sample for sample in samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"sample IDs not found in metadata: {missing}")
    return [
        RefCase(
            sample_id=sample_id,
            ref_audio=by_id[sample_id].ref_audio,
            target_text=by_id[sample_id].target_text,
        )
        for sample_id in sample_ids
    ]


def _cases_from_paths(paths: list[str]) -> list[RefCase]:
    return [
        RefCase(sample_id=Path(path).stem, ref_audio=str(Path(path).expanduser()))
        for path in paths
    ]


def _compare_codes(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    reference = reference.detach().cpu().to(torch.long)
    candidate = candidate.detach().cpu().to(torch.long)
    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    summary: dict[str, Any] = {
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "same_shape": same_shape,
        "equal": bool(same_shape and torch.equal(reference, candidate)),
    }
    if not same_shape:
        summary["mismatch_count"] = None
        summary["mismatch_ratio"] = None
        summary["first_mismatches"] = []
        return summary

    mismatches = (reference != candidate).nonzero(as_tuple=False)
    mismatch_count = int(mismatches.shape[0])
    summary["mismatch_count"] = mismatch_count
    summary["mismatch_ratio"] = (
        float(mismatch_count) / float(reference.numel()) if reference.numel() else 0.0
    )
    first = []
    for pos in mismatches[:20]:
        frame = int(pos[0])
        codebook = int(pos[1])
        first.append(
            {
                "frame": frame,
                "codebook": codebook,
                "reference": int(reference[frame, codebook]),
                "candidate": int(candidate[frame, codebook]),
            }
        )
    summary["first_mismatches"] = first
    if mismatch_count:
        per_codebook = (reference != candidate).sum(dim=0)
        summary["mismatches_by_codebook"] = [
            int(value) for value in per_codebook.tolist()
        ]
    else:
        summary["mismatches_by_codebook"] = [0 for _ in range(reference.shape[1])]
    return summary


def _encode_upstream(
    processor: Any,
    cases: list[RefCase],
    *,
    n_vq: int,
) -> dict[str, torch.Tensor]:
    paths = [case.ref_audio for case in cases]
    encoded = processor.encode_audios_from_path(paths, n_vq=n_vq)
    if len(encoded) != len(cases):
        raise RuntimeError(
            f"upstream encode returned {len(encoded)} items for {len(cases)} paths"
        )
    return {case.sample_id: codes for case, codes in zip(cases, encoded)}


def _encode_candidate(
    audio_tokenizer: Any,
    cases: list[RefCase],
    *,
    n_vq: int,
) -> dict[str, torch.Tensor]:
    paths = [case.ref_audio for case in cases]
    encoded = audio_tokenizer.encode_paths(paths, num_quantizers=n_vq)
    if len(encoded) != len(cases):
        raise RuntimeError(
            f"candidate encode returned {len(encoded)} items for {len(cases)} paths"
        )
    return {case.sample_id: codes for case, codes in zip(cases, encoded)}


def _encode_mode(
    mode: str,
    cases: list[RefCase],
    *,
    encode_fn,
    n_vq: int,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}
    for group in _mode_groups(mode, cases):
        outputs.update(encode_fn(group, n_vq=n_vq))
    return outputs


def _mode_groups(mode: str, cases: list[RefCase]) -> list[list[RefCase]]:
    if mode == "single":
        return [[case] for case in cases]
    if mode == "batch-all":
        return [cases]
    raise ValueError(f"unknown mode: {mode}")


def _comparison_rows(
    *,
    comparison_kind: str,
    left_label: str,
    right_label: str,
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    cases: list[RefCase],
    mode: str,
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        comparison = _compare_codes(left[case.sample_id], right[case.sample_id])
        rows.append(
            {
                "comparison_kind": comparison_kind,
                "mode": mode,
                "left": left_label,
                "right": right_label,
                "sample_id": case.sample_id,
                "ref_audio": case.ref_audio,
                "target_text": case.target_text,
                **comparison,
            }
        )
    return rows


def _run_modes(
    modes: list[str],
    cases: list[RefCase],
    *,
    processor: Any,
    audio_tokenizer: Any,
    n_vq: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    upstream_single = _encode_mode(
        "single",
        cases,
        encode_fn=lambda group, *, n_vq: _encode_upstream(processor, group, n_vq=n_vq),
        n_vq=n_vq,
    )
    candidate_single = _encode_mode(
        "single",
        cases,
        encode_fn=lambda group, *, n_vq: _encode_candidate(
            audio_tokenizer, group, n_vq=n_vq
        ),
        n_vq=n_vq,
    )
    rows.extend(
        _comparison_rows(
            comparison_kind="candidate_vs_upstream_same_mode",
            left_label="upstream-single",
            right_label="candidate-single",
            left=upstream_single,
            right=candidate_single,
            cases=cases,
            mode="single",
        )
    )

    for mode in modes:
        if mode == "single":
            continue
        upstream = _encode_mode(
            mode,
            cases,
            encode_fn=lambda group, *, n_vq: _encode_upstream(
                processor, group, n_vq=n_vq
            ),
            n_vq=n_vq,
        )
        candidate = _encode_mode(
            mode,
            cases,
            encode_fn=lambda group, *, n_vq: _encode_candidate(
                audio_tokenizer, group, n_vq=n_vq
            ),
            n_vq=n_vq,
        )
        rows.extend(
            _comparison_rows(
                comparison_kind="candidate_vs_upstream_same_mode",
                left_label=f"upstream-{mode}",
                right_label=f"candidate-{mode}",
                left=upstream,
                right=candidate,
                cases=cases,
                mode=mode,
            )
        )
        rows.extend(
            _comparison_rows(
                comparison_kind="candidate_mode_vs_upstream_single",
                left_label="upstream-single",
                right_label=f"candidate-{mode}",
                left=upstream_single,
                right=candidate,
                cases=cases,
                mode=mode,
            )
        )
        rows.extend(
            _comparison_rows(
                comparison_kind="upstream_mode_vs_upstream_single",
                left_label="upstream-single",
                right_label=f"upstream-{mode}",
                left=upstream_single,
                right=upstream,
                cases=cases,
                mode=mode,
            )
        )
    return rows


def _write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reference_code_parity.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# MOSS-TTS Local Reference-Code Parity",
        "",
        f"- model: `{report['model']}`",
        f"- codec: `{report['codec_model_path']}`",
        f"- device: `{report['device']}`",
        f"- n_vq: `{report['n_vq']}`",
        "",
        "| comparison | mode | sample_id | equal | mismatches | mismatch ratio | left shape | right shape |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in report["comparisons"]:
        mismatch_count = row["mismatch_count"]
        mismatch_text = "" if mismatch_count is None else str(mismatch_count)
        ratio = row["mismatch_ratio"]
        ratio_text = "" if ratio is None else f"{ratio:.6f}"
        lines.append(
            "| {kind} | {mode} | `{sample_id}` | {equal} | {mismatch} | "
            "{ratio} | `{ref_shape}` | `{cand_shape}` |".format(
                kind=f"`{row['left']}` -> `{row['right']}`",
                mode=row["mode"],
                sample_id=row["sample_id"],
                equal=row["equal"],
                mismatch=mismatch_text,
                ratio=ratio_text,
                ref_shape=row["reference_shape"],
                cand_shape=row["candidate_shape"],
            )
        )
    lines.append("")
    lines.append("## First Mismatches")
    for row in report["comparisons"]:
        if not row["first_mismatches"]:
            continue
        lines.append("")
        lines.append(f"### {row['left']} -> {row['right']} / {row['sample_id']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(row["first_mismatches"], indent=2))
        lines.append("```")
    (out_dir / "reference_code_parity.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    )
    parser.add_argument(
        "--meta",
        default="zhaochenyang20/seed-tts-eval-arrow",
    )
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--ref-audio-paths", nargs="*", default=[])
    parser.add_argument("--codec-model-path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-vq", type=int, default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["single", "batch-all"],
        choices=["single", "batch-all"],
    )
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.sample_ids and not args.ref_audio_paths:
        raise ValueError("provide --sample-ids or --ref-audio-paths")

    cases = []
    if args.sample_ids:
        cases.extend(
            _cases_from_meta(
                args.meta,
                sample_ids=args.sample_ids,
                max_samples=args.max_samples,
                lang=args.lang,
            )
        )
    if args.ref_audio_paths:
        cases.extend(_cases_from_paths(args.ref_audio_paths))

    processor = _load_processor_with_codec(args.model, args.device)
    processor_without_codec = _load_processor_without_codec(args.model)
    codec_model_path = args.codec_model_path or _resolve_audio_tokenizer_model_path(
        processor_without_codec,
        None,
    )
    audio_tokenizer = load_moss_tts_local_audio_tokenizer(
        codec_model_path,
        device=args.device,
    )
    n_vq = int(args.n_vq or processor.model_config.n_vq)

    comparisons = _run_modes(
        args.modes,
        cases,
        processor=processor,
        audio_tokenizer=audio_tokenizer,
        n_vq=n_vq,
    )

    report = {
        "model": args.model,
        "codec_model_path": codec_model_path,
        "device": args.device,
        "n_vq": n_vq,
        "cases": [case.__dict__ for case in cases],
        "comparisons": comparisons,
        "all_equal": all(row["equal"] for row in comparisons),
    }
    _write_report(report, Path(args.out))
    print(json.dumps({"out": args.out, "all_equal": report["all_equal"]}, indent=2))


if __name__ == "__main__":
    main()
