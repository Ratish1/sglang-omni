#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize Nsight Systems SQLite for MOSS-TTS Local codec profiling.

The script is intentionally dependency-free. It targets traces collected with
``SGLANG_OMNI_NVTX_RANGES=1`` and
``SGLANG_MOSS_TTS_LOCAL_VOCODER_DEEP_PROFILE=1`` and answers the non-streaming
codec question:

    inside processor.decode_audio_codes, what owns the time?

It handles common Nsight SQLite schema variants by introspecting table columns
instead of assuming one exact export version.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROCESSOR_LABEL = "moss_tts_local.vocoder.processor.decode_audio_codes"
_DEFAULT_LABELS = (
    "moss_tts_local.vocoder.processor.decode_audio_codes",
    "moss_tts_local.vocoder.audio_tokenizer.batch_decode",
    "moss_tts_local.vocoder.audio_tokenizer._decode_frame",
    "moss_tts_local.vocoder.audio_tokenizer.quantizer.decode_codes",
    "self_attn.forward",
    "self_attn._project_qkv",
    "self_attn.in_proj",
    "self_attn.rope",
    "self_attn._build_streaming_kv",
    "self_attn._build_streaming_sdpa_bias",
    "self_attn._forward_streaming_sdpa",
    "self_attn._run_flash_attention",
    "self_attn.out_proj",
    "self_attn._update_streaming_cache",
    "ffn.forward",
)
_RUNTIME_TABLE = "CUPTI_ACTIVITY_KIND_RUNTIME"
_KERNEL_TABLE = "CUPTI_ACTIVITY_KIND_KERNEL"
_NVTX_TABLE = "NVTX_EVENTS"
_STRINGS_TABLE = "StringIds"


@dataclass(frozen=True)
class Slice:
    name: str
    start: int
    end: int
    tid: int | None = None

    @property
    def dur_ns(self) -> int:
        return max(0, self.end - self.start)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    index = int(round((len(values) - 1) * pct))
    return values[min(max(index, 0), len(values) - 1)]


def _summary(slices: list[Slice]) -> dict[str, Any]:
    vals = [s.dur_ns / 1e6 for s in slices if s.dur_ns > 0]
    total = sum(vals)
    return {
        "count": len(vals),
        "total_ms": total,
        "avg_ms": total / len(vals) if vals else 0.0,
        "p50_ms": _percentile(vals, 0.50),
        "p95_ms": _percentile(vals, 0.95),
        "max_ms": max(vals) if vals else 0.0,
    }


def _duration_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [float(row[key]) for row in rows if float(row[key]) > 0]
    total = sum(vals)
    return {
        "count": len(vals),
        "total_ms": total,
        "avg_ms": total / len(vals) if vals else 0.0,
        "p50_ms": _percentile(vals, 0.50),
        "p95_ms": _percentile(vals, 0.95),
        "max_ms": max(vals) if vals else 0.0,
    }


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _string_map(conn: sqlite3.Connection) -> dict[int, str]:
    if _STRINGS_TABLE not in _table_names(conn):
        return {}
    cols = _columns(conn, _STRINGS_TABLE)
    id_col = "id" if "id" in cols else None
    value_col = "value" if "value" in cols else "string" if "string" in cols else None
    if id_col is None or value_col is None:
        return {}
    return {
        int(row[0]): str(row[1])
        for row in conn.execute(f"SELECT {id_col}, {value_col} FROM {_STRINGS_TABLE}")
    }


def _name_from_row(
    row: sqlite3.Row,
    *,
    string_ids: dict[int, str],
    text_cols: tuple[str, ...],
    id_cols: tuple[str, ...],
) -> str:
    keys = row.keys()
    for col in text_cols:
        if col in keys and row[col] not in (None, ""):
            value = row[col]
            if isinstance(value, int):
                return string_ids.get(value, str(value))
            if isinstance(value, str) and value.isdigit():
                return string_ids.get(int(value), value)
            return str(value)
    for col in id_cols:
        if col in keys and row[col] is not None:
            value = string_ids.get(int(row[col]))
            if value:
                return value
    if "cbid" in keys and row["cbid"] is not None:
        return f"cuda_runtime_cbid_{row['cbid']}"
    return "<unknown>"


def _load_slices(
    conn: sqlite3.Connection,
    table: str,
    *,
    string_ids: dict[int, str],
    text_cols: tuple[str, ...],
    id_cols: tuple[str, ...],
) -> list[Slice]:
    if table not in _table_names(conn):
        return []
    cols = _columns(conn, table)
    if "start" not in cols or "end" not in cols:
        return []

    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"SELECT * FROM {table}")
    slices: list[Slice] = []
    for row in rows:
        start = row["start"]
        end = row["end"]
        if start is None or end is None:
            continue
        tid = None
        for col in ("globalTid", "globalTidStart", "tid", "threadId"):
            if col in row.keys() and row[col] is not None:
                tid = int(row[col])
                break
        slices.append(
            Slice(
                name=_name_from_row(
                    row,
                    string_ids=string_ids,
                    text_cols=text_cols,
                    id_cols=id_cols,
                ),
                start=int(start),
                end=int(end),
                tid=tid,
            )
        )
    return slices


def _by_name(slices: list[Slice]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Slice]] = defaultdict(list)
    for item in slices:
        if item.dur_ns > 0:
            grouped[item.name].append(item)
    rows = []
    for name, group in grouped.items():
        rows.append({"name": name, **_summary(group)})
    rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    return rows


def _overlap_ns(a: Slice, b: Slice) -> int:
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _overlap_by_name(
    parents: list[Slice], children: list[Slice]
) -> list[dict[str, Any]]:
    if not parents or not children:
        return []
    parents = sorted(parents, key=lambda item: item.start)
    children = sorted(children, key=lambda item: item.start)
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "overlap_ns": 0, "duration_ns": 0}
    )
    parent_index = 0
    for child in children:
        while parent_index < len(parents) and parents[parent_index].end < child.start:
            parent_index += 1
        scan = parent_index
        while scan < len(parents) and parents[scan].start <= child.end:
            overlap = _overlap_ns(parents[scan], child)
            if overlap > 0:
                row = grouped[child.name]
                row["count"] += 1
                row["overlap_ns"] += overlap
                row["duration_ns"] += child.dur_ns
            scan += 1

    rows = []
    for name, data in grouped.items():
        rows.append(
            {
                "name": name,
                "count": int(data["count"]),
                "overlap_ms": float(data["overlap_ns"]) / 1e6,
                "duration_ms": float(data["duration_ns"]) / 1e6,
            }
        )
    rows.sort(key=lambda row: float(row["overlap_ms"]), reverse=True)
    return rows


def _filter_contains(slices: list[Slice], needle: str) -> list[Slice]:
    return [item for item in slices if needle in item.name]


def _label_totals(nvtx: list[Slice], labels: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for label in labels:
        matches = _filter_contains(nvtx, label)
        if matches:
            rows.append({"label": label, **_summary(matches)})
    rows.sort(key=lambda row: float(row["total_ms"]), reverse=True)
    return rows


def _children_by_parent_name(
    parents: list[Slice],
    children: list[Slice],
    *,
    parent_contains: str,
    child_contains: tuple[str, ...],
) -> list[dict[str, Any]]:
    matched_parents = _filter_contains(parents, parent_contains)
    rows = []
    for child_label in child_contains:
        child_matches = _filter_contains(children, child_label)
        for overlap in _overlap_by_name(matched_parents, child_matches):
            rows.append({"label": child_label, **overlap})
    rows.sort(key=lambda row: float(row["overlap_ms"]), reverse=True)
    return rows


def _top_scope_child_overlap(
    parents: list[Slice],
    children: list[Slice],
    *,
    scope_names: list[str],
    children_per_scope: int,
) -> list[dict[str, Any]]:
    rows = []
    for scope_name in scope_names:
        scope_slices = [item for item in parents if item.name == scope_name]
        if not scope_slices:
            continue
        scope_summary = _summary(scope_slices)
        for child in _overlap_by_name(scope_slices, children)[:children_per_scope]:
            rows.append(
                {
                    "scope": scope_name,
                    "scope_total_ms": scope_summary["total_ms"],
                    **child,
                }
            )
    return rows


def _kernel_category(name: str) -> str:
    lowered = name.lower()
    if "sdpa" in lowered or "flashattn" in lowered or "flash_attn" in lowered:
        return "sdpa_or_flash_attention"
    if "layer_norm" in lowered or "layernorm" in lowered:
        return "layer_norm"
    if (
        "direct_copy" in lowered
        or "copy_kernel" in lowered
        or "catarraybatchedcopy" in lowered
        or "bfloat16_copy" in lowered
    ):
        return "copy_or_layout"
    if (
        "arange" in lowered
        or "remainder" in lowered
        or "compare" in lowered
        or "where" in lowered
        or "bitwise" in lowered
        or "index" in lowered
        or "fill" in lowered
    ):
        return "index_mask_or_fill"
    if "cos_kernel" in lowered or "sin_kernel" in lowered:
        return "rope_trig"
    if (
        "nvjet" in lowered
        or "gemm" in lowered
        or "wgmma" in lowered
        or "cutlass" in lowered
    ):
        return "gemm_or_matmul"
    if (
        "elementwise" in lowered
        or "cudafunctor_add" in lowered
        or "binaryfunctor" in lowered
        or "gelu" in lowered
        or "exp_kernel" in lowered
    ):
        return "elementwise"
    return "other"


def _kernel_category_rows(kernel_overlap: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "overlap_ms": 0.0, "duration_ms": 0.0}
    )
    for row in kernel_overlap:
        category = _kernel_category(row["name"])
        target = grouped[category]
        target["count"] += int(row["count"])
        target["overlap_ms"] += float(row["overlap_ms"])
        target["duration_ms"] += float(row["duration_ms"])
    rows = [{"category": key, **value} for key, value in grouped.items()]
    rows.sort(key=lambda row: float(row["overlap_ms"]), reverse=True)
    return rows


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    def fmt_ms(value: Any) -> str:
        return f"{float(value):.3f}"

    lines = [
        "# MOSS Local Nsight Codec Summary",
        "",
        f"SQLite: `{report['sqlite']}`",
        "",
        "## Label Totals",
        "",
        "| label | count | total ms | avg ms | p50 ms | p95 ms | max ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["label_totals"]:
        lines.append(
            "| {label} | {count} | {total} | {avg} | {p50} | {p95} | {maxv} |".format(
                label=row["label"],
                count=row["count"],
                total=fmt_ms(row["total_ms"]),
                avg=fmt_ms(row["avg_ms"]),
                p50=fmt_ms(row["p50_ms"]),
                p95=fmt_ms(row["p95_ms"]),
                maxv=fmt_ms(row["max_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## Decoder Subscope Overlap Under processor.decode_audio_codes",
            "",
            "| label | name | count | overlap ms | duration ms |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["decoder_subscope_overlap"][:60]:
        lines.append(
            "| `{label}` | `{name}` | {count} | {overlap} | {duration} |".format(
                label=row["label"],
                name=row["name"],
                count=row["count"],
                overlap=fmt_ms(row["overlap_ms"]),
                duration=fmt_ms(row["duration_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## CUDA Runtime Overlap Under processor.decode_audio_codes",
            "",
            "| runtime | count | overlap ms | duration ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["runtime_overlap"][:30]:
        lines.append(
            f"| `{row['name']}` | {row['count']} | {fmt_ms(row['overlap_ms'])} | {fmt_ms(row['duration_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## Kernel Category Overlap Under processor.decode_audio_codes",
            "",
            "| category | count | overlap ms | duration ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["kernel_category_overlap"]:
        lines.append(
            f"| `{row['category']}` | {row['count']} | {fmt_ms(row['overlap_ms'])} | {fmt_ms(row['duration_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## Top Kernel Overlap Under processor.decode_audio_codes",
            "",
            "| kernel | count | overlap ms | duration ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["kernel_overlap"][:30]:
        lines.append(
            f"| `{row['name']}` | {row['count']} | {fmt_ms(row['overlap_ms'])} | {fmt_ms(row['duration_ms'])} |"
        )

    lines.extend(
        [
            "",
            "## Top Runtime Overlap By Hot Decoder Scope",
            "",
            "| scope | runtime | count | overlap ms | scope total ms |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["hot_scope_runtime_overlap"][:80]:
        lines.append(
            "| `{scope}` | `{name}` | {count} | {overlap} | {scope_total} |".format(
                scope=row["scope"],
                name=row["name"],
                count=row["count"],
                overlap=fmt_ms(row["overlap_ms"]),
                scope_total=fmt_ms(row["scope_total_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## Top Kernel Overlap By Hot Decoder Scope",
            "",
            "| scope | kernel | count | overlap ms | scope total ms |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in report["hot_scope_kernel_overlap"][:80]:
        lines.append(
            "| `{scope}` | `{name}` | {count} | {overlap} | {scope_total} |".format(
                scope=row["scope"],
                name=row["name"],
                count=row["count"],
                overlap=fmt_ms(row["overlap_ms"]),
                scope_total=fmt_ms(row["scope_total_ms"]),
            )
        )

    lines.extend(
        [
            "",
            "## Top MOSS NVTX Labels",
            "",
            "| label | count | total ms | avg ms | p95 ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["top_moss_nvtx"][:50]:
        lines.append(
            f"| `{row['name']}` | {row['count']} | {fmt_ms(row['total_ms'])} | {fmt_ms(row['avg_ms'])} | {fmt_ms(row['p95_ms'])} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(sqlite_path: Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        string_ids = _string_map(conn)
        nvtx = _load_slices(
            conn,
            _NVTX_TABLE,
            string_ids=string_ids,
            text_cols=("text", "message", "name"),
            id_cols=("textId", "messageId", "nameId"),
        )
        runtime = _load_slices(
            conn,
            _RUNTIME_TABLE,
            string_ids=string_ids,
            text_cols=("name",),
            id_cols=("nameId",),
        )
        kernels = _load_slices(
            conn,
            _KERNEL_TABLE,
            string_ids=string_ids,
            text_cols=("demangledName", "shortName", "mangledName", "name"),
            id_cols=("demangledNameId", "shortNameId", "mangledNameId", "nameId"),
        )
    finally:
        conn.close()

    processor_ranges = _filter_contains(nvtx, _PROCESSOR_LABEL)
    decoder_labels = (
        "ProjectedTransformer.forward",
        ".self_attn.forward",
        ".self_attn._forward_streaming_sdpa",
        ".self_attn._forward_streaming_flash",
        ".self_attn.rope",
        ".self_attn._build_streaming_sdpa_bias",
        ".self_attn._update_streaming_cache",
        ".ffn.forward",
        ".quantizer.decode_codes",
    )
    decoder_subscope_overlap = _children_by_parent_name(
        processor_ranges,
        nvtx,
        parent_contains=_PROCESSOR_LABEL,
        child_contains=decoder_labels,
    )
    hot_scope_names = list(
        dict.fromkeys(row["name"] for row in decoder_subscope_overlap[:16])
    )
    runtime_overlap = _overlap_by_name(processor_ranges, runtime)
    kernel_overlap = _overlap_by_name(processor_ranges, kernels)
    top_moss_nvtx = [
        row for row in _by_name(nvtx) if "moss_tts_local.vocoder" in row["name"]
    ]
    return {
        "sqlite": str(sqlite_path),
        "counts": {
            "nvtx": len(nvtx),
            "runtime": len(runtime),
            "kernels": len(kernels),
            "processor_ranges": len(processor_ranges),
        },
        "label_totals": _label_totals(nvtx, _DEFAULT_LABELS),
        "decoder_subscope_overlap": decoder_subscope_overlap,
        "decoder_subscope_summary": _duration_summary(
            decoder_subscope_overlap, "overlap_ms"
        ),
        "runtime_overlap": runtime_overlap,
        "kernel_overlap": kernel_overlap,
        "kernel_category_overlap": _kernel_category_rows(kernel_overlap),
        "hot_scope_runtime_overlap": _top_scope_child_overlap(
            nvtx,
            runtime,
            scope_names=hot_scope_names,
            children_per_scope=4,
        ),
        "hot_scope_kernel_overlap": _top_scope_child_overlap(
            nvtx,
            kernels,
            scope_names=hot_scope_names,
            children_per_scope=6,
        ),
        "top_moss_nvtx": top_moss_nvtx,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.sqlite.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    report = summarize(args.sqlite)
    (out_dir / "nsys_moss_codec_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_markdown(out_dir / "nsys_moss_codec_summary.md", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
