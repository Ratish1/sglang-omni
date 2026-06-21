#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trace MOSS-TTS Local vocoder quality drift.

This is an offline H100 debug tool. It compares the upstream processor decoder
against the packed non-streaming SGLang decoder wrapper and records where tensor
drift first appears. It also wraps packed FlashAttention calls with an SDPA
oracle so attention-level drift can be grouped by stage/layer/context.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import types
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class TensorRecord:
    key: str
    shape: list[int]
    dtype: str
    device: str
    numel: int
    nan_count: int
    inf_count: int
    stored: bool
    tensor: torch.Tensor | None


class TensorTrace:
    def __init__(self, *, max_store_elements: int) -> None:
        self.max_store_elements = max_store_elements
        self.records: dict[str, TensorRecord] = {}
        self._counts: dict[str, int] = defaultdict(int)

    def add(self, key: str, value: Any) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        index = self._counts[key]
        self._counts[key] += 1
        record_key = f"{key}#{index:03d}"
        detached = tensor.detach()
        cpu_tensor = None
        stored = detached.numel() <= self.max_store_elements
        if stored:
            cpu_tensor = detached.to("cpu")
        finite_tensor = detached.float()
        nan_count = int(torch.isnan(finite_tensor).sum().item())
        inf_count = int(torch.isinf(finite_tensor).sum().item())
        self.records[record_key] = TensorRecord(
            key=record_key,
            shape=list(detached.shape),
            dtype=str(detached.dtype),
            device=str(detached.device),
            numel=int(detached.numel()),
            nan_count=nan_count,
            inf_count=inf_count,
            stored=stored,
            tensor=cpu_tensor,
        )


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if hasattr(value, "audio") and isinstance(value.audio, torch.Tensor):
        return value.audio
    return None


def _normalize_for_compare(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() >= 2 and tensor.shape[0] == 1:
        return tensor.reshape(*tensor.shape[1:])
    return tensor


def _compare_tensors(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    reference = _normalize_for_compare(reference).float()
    candidate = _normalize_for_compare(candidate).float()
    same_shape = tuple(reference.shape) == tuple(candidate.shape)
    if not same_shape:
        return {
            "same_shape": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    delta = (reference - candidate).abs()
    reference_norm = torch.linalg.vector_norm(reference).item()
    delta_norm = torch.linalg.vector_norm(delta).item()
    mean_abs = delta.mean().item() if delta.numel() else 0.0
    max_abs = delta.max().item() if delta.numel() else 0.0
    p99_abs = torch.quantile(delta.flatten(), 0.99).item() if delta.numel() else 0.0
    snr_db = (
        "inf"
        if delta_norm == 0
        else 20.0 * math.log10(max(reference_norm, 1e-12) / delta_norm)
    )
    return {
        "same_shape": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "p99_abs": p99_abs,
        "relative_l2": delta_norm / max(reference_norm, 1e-12),
        "snr_db": snr_db,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
    }


def _compare_records(
    reference: TensorTrace,
    candidate: TensorTrace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    common = sorted(set(reference.records) & set(candidate.records))
    for key in common:
        ref = reference.records[key]
        cand = candidate.records[key]
        if ref.tensor is None or cand.tensor is None:
            rows.append(
                {
                    "key": key,
                    "stored": False,
                    "reference_shape": ref.shape,
                    "candidate_shape": cand.shape,
                }
            )
            continue
        comparison = _compare_tensors(ref.tensor, cand.tensor)
        comparison.update(
            {
                "key": key,
                "stored": True,
                "reference_nan_count": ref.nan_count,
                "candidate_nan_count": cand.nan_count,
                "reference_inf_count": ref.inf_count,
                "candidate_inf_count": cand.inf_count,
            }
        )
        rows.append(comparison)

    comparable = [row for row in rows if row.get("same_shape") is True]
    worst = sorted(
        comparable,
        key=lambda row: (
            float(row.get("max_abs", 0.0)),
            float(row.get("p99_abs", 0.0)),
        ),
        reverse=True,
    )[:20]
    summary = {
        "reference_record_count": len(reference.records),
        "candidate_record_count": len(candidate.records),
        "common_record_count": len(common),
        "missing_candidate_count": len(set(reference.records) - set(candidate.records)),
        "extra_candidate_count": len(set(candidate.records) - set(reference.records)),
        "stored_common_count": sum(1 for row in rows if row.get("stored")),
        "worst": worst,
    }
    return rows, summary


@contextmanager
def _decoder_swap(codec: Any, decoder: torch.nn.Module):
    original_decoder = codec.decoder
    codec.decoder = decoder
    try:
        yield
    finally:
        codec.decoder = original_decoder


def _decode_with_trace(
    processor: Any,
    codes_list: list[torch.Tensor],
    *,
    decoder: torch.nn.Module,
    trace_level: str,
    max_store_elements: int,
) -> tuple[list[torch.Tensor], TensorTrace]:
    trace = TensorTrace(max_store_elements=max_store_elements)
    handles = _register_decoder_hooks(decoder, trace, trace_level=trace_level)
    try:
        with torch.no_grad():
            wavs = processor.decode_audio_codes(codes_list)
            wavs = [torch.as_tensor(wav).detach().to("cpu") for wav in wavs]
            for index, wav in enumerate(wavs):
                trace.add(f"waveform.{index:03d}", wav)
            return wavs, trace
    finally:
        for handle in handles:
            handle.remove()


def _register_decoder_hooks(
    decoder: torch.nn.Module,
    trace: TensorTrace,
    *,
    trace_level: str,
) -> list[Any]:
    handles = []
    stages = list(decoder.stages) if hasattr(decoder, "stages") else list(decoder)
    for stage_index, stage in enumerate(stages):
        stage_key = f"stage_{stage_index:02d}"
        handles.extend(_hook_module(stage, trace, stage_key))
        is_transformer_stage = getattr(stage, "module_type", None) == "Transformer" or (
            hasattr(stage, "input_proj")
            and hasattr(stage, "transformer")
            and hasattr(stage, "output_proj")
        )
        if not is_transformer_stage:
            continue
        handles.extend(_hook_module(stage.input_proj, trace, f"{stage_key}.input_proj"))
        handles.extend(
            _hook_module(stage.output_proj, trace, f"{stage_key}.output_proj")
        )
        transformer = stage.transformer
        handles.extend(_hook_module(transformer, trace, f"{stage_key}.transformer"))
        if trace_level == "stage":
            continue
        for layer_index, layer in enumerate(transformer.layers):
            layer_key = f"{stage_key}.layer_{layer_index:02d}"
            handles.extend(_hook_module(layer, trace, layer_key))
            handles.extend(_hook_module(layer.norm1, trace, f"{layer_key}.norm1"))
            handles.extend(
                _hook_module(layer.self_attn, trace, f"{layer_key}.self_attn")
            )
            handles.extend(_hook_module(layer.norm2, trace, f"{layer_key}.norm2"))
            handles.extend(_hook_module(layer.ffn, trace, f"{layer_key}.ffn"))
    return handles


def _hook_module(module: torch.nn.Module, trace: TensorTrace, key: str) -> list[Any]:
    def pre_hook(_: torch.nn.Module, args: tuple[Any, ...]) -> None:
        if args:
            trace.add(f"{key}.pre", args[0])

    def post_hook(_: torch.nn.Module, __: tuple[Any, ...], output: Any) -> None:
        trace.add(f"{key}.post", output)

    return [
        module.register_forward_pre_hook(pre_hook),
        module.register_forward_hook(post_hook),
    ]


def _patch_attention_oracles(
    decoder: torch.nn.Module,
    oracle_rows: list[dict[str, Any]],
    *,
    attention_cls: type,
) -> list[tuple[Any, Any]]:
    originals = []
    for stage_index, stage in enumerate(decoder.stages):
        if not hasattr(stage, "transformer"):
            continue
        for layer_index, layer in enumerate(stage.transformer.layers):
            attn = layer.self_attn
            if not isinstance(attn, attention_cls):
                continue
            label = f"stage_{stage_index:02d}.layer_{layer_index:02d}.self_attn"
            original = attn._flash_attn_varlen
            originals.append((attn, original))
            attn._flash_attn_varlen = _make_oracle_flash_attn(
                original,
                oracle_rows,
                label=label,
                context=attn.context,
            )
    return originals


def _restore_attention_oracles(originals: list[tuple[Any, Any]]) -> None:
    for attn, original in originals:
        attn._flash_attn_varlen = original


def _delegate_layer_forwards(decoder: torch.nn.Module) -> None:
    for stage in decoder.stages:
        if not hasattr(stage, "transformer"):
            continue
        for layer in stage.transformer.layers:

            def forward(self: torch.nn.Module, x: torch.Tensor, **kwargs: Any) -> Any:
                return self.source(x, **kwargs)

            layer.forward = types.MethodType(forward, layer)


def _make_oracle_flash_attn(
    flash_attn_func: Any,
    oracle_rows: list[dict[str, Any]],
    *,
    label: str,
    context: int | None,
) -> Any:
    def wrapped(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
        **kwargs: Any,
    ) -> torch.Tensor:
        out = flash_attn_func(q, k, v, cu_q, cu_k, max_q, max_k, **kwargs)
        row: dict[str, Any] = {
            "label": label,
            "context": context,
            "q_shape": list(q.shape),
            "k_shape": list(k.shape),
            "v_shape": list(v.shape),
            "max_seqlen_q": int(max_q),
            "max_seqlen_k": int(max_k),
            "causal": bool(kwargs.get("causal", False)),
            "window_size": list(kwargs.get("window_size", (-1, -1))),
        }
        try:
            reference = _sdpa_varlen_reference(
                q,
                k,
                v,
                cu_q,
                cu_k,
                causal=bool(kwargs.get("causal", False)),
                window_size=tuple(kwargs.get("window_size", (-1, -1))),
            )
            comparison = _compare_tensors(
                reference.detach().to("cpu"),
                out.detach().to("cpu"),
            )
            row.update(comparison)
            row["nan_count"] = int(torch.isnan(out.float()).sum().item())
            row["oracle_error"] = None
        except Exception as exc:  # pragma: no cover - debug script records the error.
            row["oracle_error"] = repr(exc)
        oracle_rows.append(row)
        return out

    return wrapped


def _sdpa_varlen_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    causal: bool,
    window_size: tuple[int, int],
) -> torch.Tensor:
    q_offsets = [int(x) for x in cu_q.detach().to("cpu").tolist()]
    k_offsets = [int(x) for x in cu_k.detach().to("cpu").tolist()]
    pieces = []
    for index in range(len(q_offsets) - 1):
        q_start, q_end = q_offsets[index], q_offsets[index + 1]
        k_start, k_end = k_offsets[index], k_offsets[index + 1]
        q_item = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_item = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        v_item = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        mask = _local_attention_mask(
            q_len=q_end - q_start,
            k_len=k_end - k_start,
            causal=causal,
            window_size=window_size,
            device=q.device,
        )
        out = F.scaled_dot_product_attention(
            q_item,
            k_item,
            v_item,
            attn_mask=mask,
            is_causal=False,
        )
        pieces.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(pieces, dim=0)


def _local_attention_mask(
    *,
    q_len: int,
    k_len: int,
    causal: bool,
    window_size: tuple[int, int],
    device: torch.device,
) -> torch.Tensor | None:
    left, right = window_size
    if not causal and (left, right) == (-1, -1):
        return None
    q_pos = torch.arange(q_len, device=device).view(q_len, 1)
    k_pos = torch.arange(k_len, device=device).view(1, k_len)
    mask = torch.ones(q_len, k_len, dtype=torch.bool, device=device)
    if causal:
        mask &= k_pos <= q_pos + (k_len - q_len)
    if (left, right) != (-1, -1):
        mask &= k_pos >= q_pos + (k_len - q_len) - left
        mask &= k_pos <= q_pos + (k_len - q_len) + right
    return mask.view(1, 1, q_len, k_len)


def _make_synthetic_codes(
    *,
    batch_size: int,
    frames: int,
    codebooks: int,
    vocab_size: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [
        torch.randint(
            0,
            vocab_size,
            (frames, codebooks),
            dtype=torch.long,
            generator=generator,
        )
        for _ in range(batch_size)
    ]


def _load_codes_file(path: Path) -> list[torch.Tensor]:
    value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        if "codes_list" in value:
            value = value["codes_list"]
        elif "audio_codes" in value:
            value = value["audio_codes"]
        else:
            raise ValueError(f"{path} must contain codes_list or audio_codes")
    if isinstance(value, torch.Tensor):
        if value.dim() == 2:
            return [value.to(torch.long)]
        if value.dim() == 3:
            return [item.to(torch.long) for item in value]
        raise ValueError(
            f"{path} tensor must have rank 2 or 3, got {tuple(value.shape)}"
        )
    if isinstance(value, list):
        return [torch.as_tensor(item, dtype=torch.long) for item in value]
    raise TypeError(f"{path} must load to a tensor, list, or dict, got {type(value)!r}")


def _parse_probe(value: str) -> tuple[int, int]:
    if "x" not in value:
        raise argparse.ArgumentTypeError(f"probe must be BxT, got {value!r}")
    batch, frames = value.lower().split("x", 1)
    return int(batch), int(frames)


def _summarize_oracles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("oracle_error") is None]
    errors = [row for row in rows if row.get("oracle_error") is not None]
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        by_context[str(row.get("context"))].append(row)
        by_stage[str(row["label"].split(".")[0])].append(row)
    return {
        "count": len(rows),
        "valid_count": len(valid),
        "error_count": len(errors),
        "nan_call_count": sum(1 for row in valid if int(row.get("nan_count", 0)) > 0),
        "worst": sorted(
            valid,
            key=lambda row: (
                float(row.get("max_abs", 0.0)),
                float(row.get("p99_abs", 0.0)),
            ),
            reverse=True,
        )[:20],
        "by_context": {
            key: _group_oracle_stats(items) for key, items in sorted(by_context.items())
        },
        "by_stage": {
            key: _group_oracle_stats(items) for key, items in sorted(by_stage.items())
        },
        "errors": errors[:20],
    }


def _group_oracle_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    max_abs = [float(row.get("max_abs", 0.0)) for row in rows]
    p99_abs = [float(row.get("p99_abs", 0.0)) for row in rows]
    mean_abs = [float(row.get("mean_abs", 0.0)) for row in rows]
    return {
        "count": len(rows),
        "max_abs": max(max_abs) if max_abs else 0.0,
        "mean_abs_max": max(mean_abs) if mean_abs else 0.0,
        "p99_abs_max": max(p99_abs) if p99_abs else 0.0,
    }


def _write_report(
    *,
    out_dir: Path,
    results: list[dict[str, Any]],
    dependencies: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "moss_tts_local_vocoder_quality_trace_v1",
        "dependencies": dependencies,
        "results": results,
    }
    (out_dir / "moss_tts_local_vocoder_quality_trace.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# MOSS-TTS Local Vocoder Quality Trace",
        "",
        "## Probes",
        "",
        "| probe | waveform max_abs | waveform mean_abs | records compared | oracle calls | oracle errors | oracle worst max_abs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        waveform = item["waveform_comparison"]
        oracle = item["oracle_summary"]
        worst = oracle["worst"][0]["max_abs"] if oracle["worst"] else 0.0
        lines.append(
            "| {probe} | {max_abs:.6g} | {mean_abs:.6g} | {records} | {oracle_count} | {errors} | {worst:.6g} |".format(
                probe=item["probe"],
                max_abs=float(waveform.get("max_abs", 0.0)),
                mean_abs=float(waveform.get("mean_abs", 0.0)),
                records=item["record_summary"]["stored_common_count"],
                oracle_count=oracle["count"],
                errors=oracle["error_count"],
                worst=float(worst),
            )
        )
    lines.extend(["", "## Worst Tensor Boundaries", ""])
    for item in results:
        lines.extend([f"### {item['probe']}", ""])
        lines.append("| key | max_abs | p99_abs | mean_abs | snr_db |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in item["record_summary"]["worst"][:10]:
            lines.append(
                "| {key} | {max_abs:.6g} | {p99_abs:.6g} | {mean_abs:.6g} | {snr} |".format(
                    key=row["key"],
                    max_abs=float(row.get("max_abs", 0.0)),
                    p99_abs=float(row.get("p99_abs", 0.0)),
                    mean_abs=float(row.get("mean_abs", 0.0)),
                    snr=row.get("snr_db"),
                )
            )
        lines.append("")
    lines.extend(["", "## Oracle Groups", ""])
    for item in results:
        lines.extend([f"### {item['probe']}", "", "By stage:", ""])
        lines.append("| group | calls | max_abs | p99_abs_max | mean_abs_max |")
        lines.append("|---|---:|---:|---:|---:|")
        for group, stats in item["oracle_summary"]["by_stage"].items():
            lines.append(
                f"| {group} | {stats['count']} | {stats['max_abs']:.6g} | "
                f"{stats['p99_abs_max']:.6g} | {stats['mean_abs_max']:.6g} |"
            )
        lines.extend(["", "By context:", ""])
        lines.append("| context | calls | max_abs | p99_abs_max | mean_abs_max |")
        lines.append("|---|---:|---:|---:|---:|")
        for group, stats in item["oracle_summary"]["by_context"].items():
            lines.append(
                f"| {group} | {stats['count']} | {stats['max_abs']:.6g} | "
                f"{stats['p99_abs_max']:.6g} | {stats['mean_abs_max']:.6g} |"
            )
        lines.append("")
    (out_dir / "moss_tts_local_vocoder_quality_trace.md").write_text(
        "\n".join(lines) + "\n"
    )


def run(args: argparse.Namespace) -> None:
    from sglang_omni.models.moss_tts_local.stages import _load_moss_tts_local_processor
    from sglang_omni.models.moss_tts_local.vocoder_decoder import (
        MossTTSLocalAttention,
        MossTTSLocalVocoderDecoder,
    )

    device = torch.device(args.device)
    processor = _load_moss_tts_local_processor(args.model_path, device=args.device)
    codec = processor.audio_tokenizer
    candidate_decoder = MossTTSLocalVocoderDecoder(codec.decoder)
    if args.candidate_attention == "sdpa":
        for module in candidate_decoder.modules():
            if isinstance(module, MossTTSLocalAttention):
                module._can_run_packed_flash = lambda _: False  # type: ignore[method-assign]
    if args.layer_body == "source":
        _delegate_layer_forwards(candidate_decoder)

    dependencies = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "model_path": args.model_path,
        "candidate_attention": args.candidate_attention,
        "layer_body": args.layer_body,
    }
    results = []
    probe_inputs = []
    if not args.skip_synthetic:
        for probe_index, (batch_size, frames) in enumerate(args.probes):
            probe_inputs.append(
                (
                    f"{batch_size}x{frames}",
                    _make_synthetic_codes(
                        batch_size=batch_size,
                        frames=frames,
                        codebooks=args.codebooks,
                        vocab_size=args.vocab_size,
                        seed=args.seed + probe_index,
                    ),
                )
            )
    for path in args.codes_pt:
        codes_list = _load_codes_file(Path(path))
        max_frames = max(int(codes.shape[0]) for codes in codes_list)
        probe_inputs.append(
            (f"{Path(path).stem}:{len(codes_list)}x{max_frames}", codes_list)
        )

    for probe_label, codes_list in probe_inputs:
        reference_wavs, reference_trace = _decode_with_trace(
            processor,
            codes_list,
            decoder=codec.decoder,
            trace_level=args.trace_level,
            max_store_elements=args.max_store_elements,
        )

        oracle_rows: list[dict[str, Any]] = []
        oracle_originals = _patch_attention_oracles(
            candidate_decoder,
            oracle_rows,
            attention_cls=MossTTSLocalAttention,
        )
        try:
            with _decoder_swap(codec, candidate_decoder):
                candidate_wavs, candidate_trace = _decode_with_trace(
                    processor,
                    codes_list,
                    decoder=candidate_decoder,
                    trace_level=args.trace_level,
                    max_store_elements=args.max_store_elements,
                )
        finally:
            _restore_attention_oracles(oracle_originals)

        waveform_rows = [
            _compare_tensors(ref, cand)
            for ref, cand in zip(reference_wavs, candidate_wavs, strict=True)
        ]
        waveform_summary = {
            "max_abs": max(float(row.get("max_abs", 0.0)) for row in waveform_rows),
            "mean_abs": max(float(row.get("mean_abs", 0.0)) for row in waveform_rows),
            "items": waveform_rows,
        }
        record_rows, record_summary = _compare_records(reference_trace, candidate_trace)
        results.append(
            {
                "probe": probe_label,
                "batch_size": len(codes_list),
                "frames": max(int(codes.shape[0]) for codes in codes_list),
                "waveform_comparison": waveform_summary,
                "record_comparisons": record_rows,
                "record_summary": record_summary,
                "oracle_summary": _summarize_oracles(oracle_rows),
            }
        )
        print(
            f"{probe_label}: waveform max_abs={waveform_summary['max_abs']:.6g} "
            f"oracle_calls={len(oracle_rows)}"
        )
    _write_report(
        out_dir=Path(args.out_dir), results=results, dependencies=dependencies
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--probe",
        dest="probes",
        type=_parse_probe,
        action="append",
        default=None,
        help="Probe shape as BxT, for example 1x25. Repeatable.",
    )
    parser.add_argument("--codebooks", type=int, default=12)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--codes-pt",
        action="append",
        default=[],
        help=(
            "Optional torch-saved tensor/list/dict of [T,C] code rows or a "
            "batch of rows."
        ),
    )
    parser.add_argument(
        "--skip-synthetic",
        action="store_true",
        help="Only run --codes-pt probes.",
    )
    parser.add_argument(
        "--trace-level",
        choices=("stage", "layer"),
        default="layer",
    )
    parser.add_argument(
        "--candidate-attention",
        choices=("packed", "sdpa"),
        default="packed",
    )
    parser.add_argument(
        "--layer-body",
        choices=("owned", "source"),
        default="owned",
        help=(
            "Debug-only isolation control. 'source' delegates transformer "
            "layer bodies back to the upstream MOSS layer and requires "
            "--candidate-attention sdpa."
        ),
    )
    parser.add_argument(
        "--max-store-elements",
        type=int,
        default=2_000_000,
        help=(
            "Only tensors with at most this many elements are stored for "
            "pairwise comparison."
        ),
    )
    args = parser.parse_args()
    if args.skip_synthetic and not args.codes_pt:
        parser.error("--skip-synthetic requires at least one --codes-pt file")
    if args.layer_body == "source" and args.candidate_attention != "sdpa":
        parser.error("--layer-body source requires --candidate-attention sdpa")
    if args.probes is None:
        args.probes = [
            (1, 25),
            (1, 100),
            (1, 300),
            (8, 100),
            (8, 300),
        ]
    return args


if __name__ == "__main__":
    run(parse_args())
