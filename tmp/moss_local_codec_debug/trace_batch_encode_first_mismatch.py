# SPDX-License-Identifier: Apache-2.0
"""Trace first upstream codec mismatch between singleton and batched encode.

This script is debug-only. It intentionally traces the upstream HF codec path,
because candidate-vs-upstream same-mode parity has already passed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
    resolve_moss_checkpoint,
)
from sglang_omni.models.moss_tts_local.stages import _normalize_processor_config


@dataclass(frozen=True)
class RefCase:
    sample_id: str
    ref_audio: str
    target_text: str | None = None


@dataclass
class TensorRecord:
    key: str
    shape: list[int]
    dtype: str
    tensor: torch.Tensor


def _load_processor_with_codec(model_path: str, device: str) -> Any:
    checkpoint_dir = resolve_moss_checkpoint(model_path)
    with moss_transformers_processor_compat():
        processor_cls = load_moss_processor_class(checkpoint_dir)
        processor = processor_cls.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
        )
    _normalize_processor_config(processor)
    processor.audio_tokenizer.eval()
    processor.audio_tokenizer.to(device)
    return processor


def _cases_from_meta(
    meta: str,
    sample_ids: list[str],
    max_samples: int | None,
    lang: str,
) -> list[RefCase]:
    samples = load_seedtts_samples(meta, max_samples, split=lang)
    by_id = {sample.sample_id: sample for sample in samples}
    if sample_ids:
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(f"sample IDs not found in metadata: {missing}")
        samples = [by_id[sample_id] for sample_id in sample_ids]
    return [
        RefCase(sample.sample_id, sample.ref_audio, sample.target_text)
        for sample in samples
    ]


def _capture_prepared_waveforms(
    processor: Any,
    cases: list[RefCase],
    n_vq: int,
) -> list[torch.Tensor]:
    captured: dict[str, list[torch.Tensor]] = {}
    original_batch_encode = processor.audio_tokenizer.batch_encode

    def wrapped_batch_encode(
        wavs: list[torch.Tensor], *args: Any, **kwargs: Any
    ) -> Any:
        captured["wavs"] = [wav.detach().cpu().clone() for wav in wavs]
        return original_batch_encode(wavs, *args, **kwargs)

    processor.audio_tokenizer.batch_encode = wrapped_batch_encode
    try:
        processor.encode_audios_from_path(
            [case.ref_audio for case in cases],
            n_vq=n_vq,
        )
    finally:
        processor.audio_tokenizer.batch_encode = original_batch_encode

    if "wavs" not in captured:
        raise RuntimeError("processor path did not call audio_tokenizer.batch_encode")
    return captured["wavs"]


def _compare_tensor(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    crop_common_shape: bool = False,
) -> dict[str, Any]:
    left = left.detach().cpu()
    right = right.detach().cpu()
    same_shape = tuple(left.shape) == tuple(right.shape)
    common_shape: list[int] | None = None
    if not same_shape and crop_common_shape and left.ndim == right.ndim:
        common_shape = [min(int(a), int(b)) for a, b in zip(left.shape, right.shape)]
        slices = tuple(slice(0, size) for size in common_shape)
        left_cmp = left[slices]
        right_cmp = right[slices]
    else:
        left_cmp = left
        right_cmp = right

    comparable = tuple(left_cmp.shape) == tuple(right_cmp.shape)
    if not comparable:
        return {
            "same_shape": same_shape,
            "common_shape": common_shape,
            "equal": False,
            "max_abs": None,
            "mean_abs": None,
        }

    if left_cmp.is_floating_point() or right_cmp.is_floating_point():
        diff = (left_cmp.to(torch.float32) - right_cmp.to(torch.float32)).abs()
        equal = bool(torch.equal(left_cmp, right_cmp))
        max_abs = float(diff.max()) if diff.numel() else 0.0
        mean_abs = float(diff.mean()) if diff.numel() else 0.0
    else:
        diff = left_cmp != right_cmp
        equal = bool(torch.equal(left_cmp, right_cmp))
        max_abs = float(diff.to(torch.float32).max()) if diff.numel() else 0.0
        mean_abs = float(diff.to(torch.float32).mean()) if diff.numel() else 0.0
    return {
        "same_shape": same_shape,
        "common_shape": common_shape,
        "equal": equal,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
    }


def _extract_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_extract_tensors(item))
        return tensors
    if isinstance(value, dict):
        tensors = []
        for item in value.values():
            tensors.extend(_extract_tensors(item))
        return tensors
    if hasattr(value, "__dict__"):
        tensors = []
        for item in value.__dict__.values():
            tensors.extend(_extract_tensors(item))
        return tensors
    return []


def _select_sample_tensor(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    sample_index: int,
) -> torch.Tensor:
    if batch_size <= 1:
        if tensor.ndim > 0 and int(tensor.shape[0]) == 1:
            return tensor[0]
        if tensor.ndim > 1 and int(tensor.shape[1]) == 1:
            return tensor[:, 0]
        return tensor
    if tensor.ndim > 0 and int(tensor.shape[0]) == batch_size:
        return tensor[sample_index]
    if tensor.ndim > 1 and int(tensor.shape[1]) == batch_size:
        return tensor[:, sample_index]
    return tensor


def _is_leaf_module(module: torch.nn.Module) -> bool:
    return not any(module.children())


def _trace_batch_encode(
    codec: torch.nn.Module,
    prepared: list[torch.Tensor],
    *,
    n_vq: int,
    sample_index: int,
    max_records: int,
) -> tuple[dict[str, TensorRecord], Any]:
    records: dict[str, TensorRecord] = {}
    call_counts: dict[str, int] = {}
    hooks = []
    batch_size = len(prepared)

    def add_records(name: str, values: Any, kind: str) -> None:
        tensors = _extract_tensors(values)
        for tensor_index, tensor in enumerate(tensors):
            if len(records) >= max_records:
                return
            call_index = call_counts.get(name, 0)
            key = f"{name}#{call_index:04d}.{kind}.t{tensor_index}"
            selected = _select_sample_tensor(
                tensor.detach(),
                batch_size=batch_size,
                sample_index=sample_index,
            )
            records[key] = TensorRecord(
                key=key,
                shape=[int(dim) for dim in selected.shape],
                dtype=str(selected.dtype),
                tensor=selected.detach().cpu().clone(),
            )

    def make_pre_hook(name: str) -> Any:
        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...]) -> None:
            add_records(name, inputs, "pre")

        return hook

    def make_post_hook(name: str) -> Any:
        def hook(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            add_records(name, output, "post")
            call_counts[name] = call_counts.get(name, 0) + 1

        return hook

    for name, module in codec.named_modules():
        if name and _is_leaf_module(module):
            hooks.append(module.register_forward_pre_hook(make_pre_hook(name)))
            hooks.append(module.register_forward_hook(make_post_hook(name)))

    try:
        with torch.inference_mode():
            output = codec.batch_encode(prepared, num_quantizers=n_vq)
    finally:
        for hook in hooks:
            hook.remove()

    return records, output


def _record_summary(record: TensorRecord) -> dict[str, Any]:
    return {
        "key": record.key,
        "shape": record.shape,
        "dtype": record.dtype,
    }


def _first_record_mismatch(
    singleton: dict[str, TensorRecord],
    batched: dict[str, TensorRecord],
) -> dict[str, Any] | None:
    common_keys = [key for key in singleton if key in batched]
    for key in common_keys:
        left = singleton[key]
        right = batched[key]
        comparison = _compare_tensor(
            left.tensor,
            right.tensor,
            crop_common_shape=True,
        )
        if not comparison["equal"]:
            return {
                "key": key,
                "singleton": _record_summary(left),
                "batched": _record_summary(right),
                "comparison": comparison,
            }
    if len(common_keys) != len(singleton) or len(common_keys) != len(batched):
        return {
            "key": None,
            "missing_singleton": [key for key in batched if key not in singleton][:20],
            "missing_batched": [key for key in singleton if key not in batched][:20],
        }
    return None


def _encoded_codes(output: Any, sample_index: int = 0) -> torch.Tensor:
    audio_codes = output.audio_codes.detach().cpu().to(torch.long)
    audio_lengths = output.audio_codes_lengths.detach().cpu()
    length = int(audio_lengths[sample_index])
    return audio_codes[:, sample_index, :length].transpose(0, 1).contiguous()


def _write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_encode_trace.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# MOSS-TTS Local Batch Encode Trace",
        "",
        f"- model: `{report['model']}`",
        f"- device: `{report['device']}`",
        f"- traced sample: `{report['trace_sample_id']}`",
        "",
        "## Prepared Waveform Comparisons",
        "",
        "| sample_id | equal | max_abs | mean_abs |",
        "|---|---:|---:|---:|",
    ]
    for row in report["prepared_comparisons"]:
        lines.append(
            f"| `{row['sample_id']}` | {row['equal']} | "
            f"{row['max_abs']} | {row['mean_abs']} |"
        )
    lines.extend(
        [
            "",
            "## First Module Mismatch",
            "",
            "```json",
            json.dumps(report["first_mismatch"], indent=2),
            "```",
            "",
            "## Code Comparison For Traced Sample",
            "",
            "```json",
            json.dumps(report["code_comparison"], indent=2),
            "```",
        ]
    )
    (out_dir / "batch_encode_trace.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
    )
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-ids", nargs="*", required=True)
    parser.add_argument("--trace-sample-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-vq", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = _load_processor_with_codec(args.model, args.device)
    n_vq = int(args.n_vq or processor.model_config.n_vq)
    cases = _cases_from_meta(args.meta, args.sample_ids, args.max_samples, args.lang)
    sample_ids = [case.sample_id for case in cases]
    if args.trace_sample_id not in sample_ids:
        raise ValueError(f"trace sample id not present: {args.trace_sample_id}")
    trace_index = sample_ids.index(args.trace_sample_id)

    singleton_prepared_by_id: dict[str, torch.Tensor] = {}
    for case in cases:
        singleton_prepared_by_id[case.sample_id] = _capture_prepared_waveforms(
            processor,
            [case],
            n_vq,
        )[0]
    batched_prepared = _capture_prepared_waveforms(processor, cases, n_vq)

    prepared_comparisons = []
    for case, batch_wav in zip(cases, batched_prepared):
        comparison = _compare_tensor(
            singleton_prepared_by_id[case.sample_id], batch_wav
        )
        prepared_comparisons.append({"sample_id": case.sample_id, **comparison})

    singleton_prepared = [
        singleton_prepared_by_id[args.trace_sample_id].to(args.device)
    ]
    batched_prepared_gpu = [wav.to(args.device) for wav in batched_prepared]

    singleton_records, singleton_output = _trace_batch_encode(
        processor.audio_tokenizer,
        singleton_prepared,
        n_vq=n_vq,
        sample_index=0,
        max_records=args.max_records,
    )
    batched_records, batched_output = _trace_batch_encode(
        processor.audio_tokenizer,
        batched_prepared_gpu,
        n_vq=n_vq,
        sample_index=trace_index,
        max_records=args.max_records,
    )

    code_comparison = _compare_tensor(
        _encoded_codes(singleton_output, 0),
        _encoded_codes(batched_output, trace_index),
    )
    report = {
        "model": args.model,
        "device": args.device,
        "n_vq": n_vq,
        "trace_sample_id": args.trace_sample_id,
        "cases": [case.__dict__ for case in cases],
        "prepared_comparisons": prepared_comparisons,
        "singleton_record_count": len(singleton_records),
        "batched_record_count": len(batched_records),
        "first_mismatch": _first_record_mismatch(singleton_records, batched_records),
        "code_comparison": code_comparison,
    }
    _write_report(report, Path(args.out))
    print(
        json.dumps(
            {
                "out": args.out,
                "first_mismatch": report["first_mismatch"],
                "code_comparison": code_comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
