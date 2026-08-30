# SPDX-License-Identifier: Apache-2.0
"""Compare two H100 protocol result trees in place and write one report.

Usage:
    python compare_runs.py OLD NEW [--out DIR] [--workers N] [--skip-traces] [--top N]

OLD and NEW are the OUT directories the h100_runs.sh steps wrote on the
previous image (A) and the current image (B). The report goes to DIR/report.md
(default NEW/compare) with its tables as JSON next to it. Everything comes from
the small files in the two trees except section 6, which streams the torch
profiler traces once through trace_kernels.py and caches the per-file
statistics as <trace>.stats.json, so the traces never have to leave the
machine. Sections:

  1 inventory and package versions (env/pip_freeze.txt from capture_env)
  2 server logs: backend choices, kernel config warnings, decode and prefill
    throughput lines, error counts
  3 benchmark results: summary metrics per run family and per-sample paired
    deltas (latency, audio duration, tokens, rtf, answer flips)
  4 CPU preprocessing A/B files
  5 request profiler events: per-stage intervals, hops, per-request rows paired
    by admission order
  6 torch traces: per stage process kernel time, launches, families, kernels

Run it from the omni checkout with the server venv. Only the standard library
and trace_kernels.py are needed.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trace_kernels  # noqa: E402

RESULT_NAMES = {
    "videomme_results.json",
    "videoamme_results.json",
    "speed_results.json",
    "eval_results.json",
    "mmsu_results.json",
}
SKIP_DIRS = {"compare", "venv_old_cpu", "audio", "__pycache__", "site-packages"}
PACKAGES = re.compile(
    r"^(torch|torchvision|torchaudio|torchcodec|triton|sglang|sgl[-_]kernel|"
    r"flashinfer[-_a-z0-9]*|deep[-_]gemm|transformers|tokenizers|numpy|av|"
    r"qwen[-_]vl[-_]utils|huggingface[-_]hub|nvidia-[a-z0-9-]+|cuda-[a-z-]+|"
    r"xformers|accelerate|safetensors|pillow|librosa|soundfile)$",
    re.I,
)
SUMMARY_KEYS = (
    "completed_requests",
    "failed_requests",
    "accuracy",
    "throughput_qps",
    "latency_mean_s",
    "latency_median_s",
    "latency_p95_s",
    "rtf_mean",
    "rtf_p95",
    "audio_duration_mean_s",
    "audio_throughput_s_per_s",
    "output_tokens_mean",
    "prompt_tokens_mean",
    "output_tok_per_req_s",
)
INTERVALS = (
    ("stage_input_received", "stage_complete"),
    ("encoder_start", "encoder_end"),
    ("preprocess_start", "preprocess_end"),
    ("scheduler_request_build_start", "scheduler_request_build_end"),
    ("scheduler_request_build_end", "scheduler_queue_enter"),
    ("scheduler_queue_enter", "scheduler_prefill_start"),
    ("scheduler_prefill_start", "scheduler_prefill_end"),
    ("scheduler_prefill_end", "stage_complete"),
    ("scheduler_prefill_start", "stage_first_stream_chunk_sent"),
    ("scheduler_prefill_start", "scheduler_first_emit"),
    ("scheduler_first_emit", "stage_complete"),
)
DECODE_RE = re.compile(
    r"Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+), "
    r"cuda graph: (\w+), gen throughput \(token/s\): ([\d.]+)"
)
PREFILL_RE = re.compile(
    r"Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+),"
    r".*?input throughput \(token/s\): ([\d.]+)"
)
LINE_PATTERNS = (
    ("backend policy", re.compile(r"Configured SGLang backend policy: (.*)")),
    ("runtime configuration", re.compile(r"SGLang runtime configuration: (.*)")),
    ("decode cuda graph", re.compile(r"Capture target decode CUDA graph begin\. (.*)")),
    (
        "prefill cuda graph",
        re.compile(r"Capture target prefill CUDA graph begin\. (.*)"),
    ),
    ("audio layer graphs", re.compile(r"audio layer CUDA graphs captured for (.*)")),
    ("code2wav graphs", re.compile(r"Code2Wav CUDA graph runner (.*)")),
    (
        "moe deferred finalize",
        re.compile(r"(FlashInfer TRTLLM MoE deferred finalize is \w+ \(.*?\))"),
    ),
    (
        "moe config fallback",
        re.compile(
            r"[Cc]onfig file not found at \S*/triton_utils/configs/(.+?\.json)[.;,]?\s*"
            r"(Fallback to triton version [\d.]+|reusing the tuned up-projection config without TMA|you can create them)?"
        ),
    ),
    (
        "fp8 block gemm default config",
        re.compile(
            r"Using default W8A8 Block FP8 kernel config.*?quantization/configs/(.+?\.json)"
        ),
    ),
    (
        "flashinfer untuned shape",
        re.compile(r"No tuned config covers (\S+) input_shapes=(\(.*?\)\))"),
    ),
    (
        "flashinfer autotune cache",
        re.compile(r"\[Autotuner\]: Loaded (\d+) configs from (\S+)"),
    ),
    (
        "flashinfer autotune run",
        re.compile(r"\[Autotuner\]: (Autotuning process (?:starts|ends))"),
    ),
    ("deepgemm", re.compile(r"(DeepGEMM[^\n]{0,140})")),
    ("staying eager", re.compile(r"(.{0,80}staying eager.{0,80})")),
    ("stage env defaults", re.compile(r"Configured stage process env defaults: (.*)")),
)
COUNT_PATTERNS = (
    ("Traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("[ERROR]", re.compile(r"\[ERROR\]")),
    ("[WARNING]", re.compile(r"\[WARNING\]|- WARNING -")),
    ("out of memory", re.compile(r"out of memory|OutOfMemory|OOM", re.I)),
    ("retract", re.compile(r"retract", re.I)),
    ("jit compile", re.compile(r"\bjit\b.*compil|compil.*\bjit\b", re.I)),
    ("Decode batch lines", DECODE_RE),
    ("Prefill batch lines", PREFILL_RE),
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fmt(v, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def pct(a, b) -> str:
    if a is None or b is None or not isinstance(a, (int, float)) or a == 0:
        return "-"
    return f"{100.0 * (b - a) / a:+.1f}%"


def delta(a, b, nd: int = 3) -> str:
    if a is None or b is None:
        return "-"
    return f"{b - a:+.{nd}f}"


def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def mean(vals):
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def pctl(vals, q: float):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def cell(v) -> str:
    return str(v).replace("|", "\\|").replace("\n", " ")


def table(headers: list[str], rows: list[list]) -> str:
    if not rows:
        return "(none)\n\n"
    out = ["| " + " | ".join(cell(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(out) + "\n\n"


class Report:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def h(self, level: int, text: str) -> None:
        self.parts.append(f"\n{'#' * level} {text}\n\n")

    def p(self, text: str) -> None:
        self.parts.append(text + "\n\n")

    def table(self, headers, rows) -> None:
        self.parts.append(table(headers, rows))

    def pre(self, text: str) -> None:
        self.parts.append("```\n" + text.rstrip() + "\n```\n\n")

    def text(self) -> str:
        return "".join(self.parts)


def walk_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def du(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for dirpath, _, filenames in os.walk(path, followlinks=True):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# 1 inventory and versions
# ---------------------------------------------------------------------------


def inventory(root: Path) -> dict:
    entries = []
    for child in sorted(root.iterdir()):
        entries.append([child.name, du(child), child.is_dir()])
    traces = []
    bases: dict[tuple, list] = collections.defaultdict(list)
    for p in walk_files(root):
        if ".trace.json" in p.name and not p.name.endswith(".stats.json"):
            base = p.name.replace(".trace.json.gz", "").replace(".trace.json", "")
            bases[(str(p.parent.relative_to(root)), base)].append(p.name)
            traces.append([str(p.relative_to(root)), p.stat().st_size])
    raw_and_gz = sorted(f"{d}/{b}" for (d, b), names in bases.items() if len(names) > 1)
    events = sorted(
        {
            str(p.parent.relative_to(root))
            for p in walk_files(root)
            if re.match(r"events_.*\.jsonl$", p.name)
        }
    )
    return {
        "entries": entries,
        "traces": sorted(traces),
        "raw_and_gz": raw_and_gz,
        "events_dirs": events,
        "logs": (
            sorted(p.name for p in (root / "logs").glob("*.log"))
            if (root / "logs").is_dir()
            else []
        ),
    }


def read_freeze(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    path = root / "env" / "pip_freeze.txt"
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
        elif " @ " in line:
            name, _, ref = line.partition(" @ ")
            out[name.strip().lower().replace("_", "-")] = ref.strip()[-60:]
    return out


def section_inventory(rep: Report, old: Path, new: Path) -> dict:
    inv = {"A": inventory(old), "B": inventory(new)}
    rep.h(2, "1. Inventory and versions")
    rep.p(f"A (previous image): `{old}`  \nB (current image): `{new}`")
    for label, root in (("A", old), ("B", new)):
        rows = [
            [name, human(size), "dir" if is_dir else "file"]
            for name, size, is_dir in inv[label]["entries"]
        ]
        rep.h(
            3, f"{label} top level ({human(sum(e[1] for e in inv[label]['entries']))})"
        )
        rep.table(["entry", "size", "kind"], rows)
        tr = inv[label]["traces"]
        if tr:
            rep.p(
                f"{len(tr)} trace files, {human(sum(t[1] for t in tr))}: "
                + ", ".join(f"{t[0]} ({human(t[1])})" for t in tr)
            )
        if inv[label]["raw_and_gz"]:
            rep.p(
                "Both raw and gz present (gzip was interrupted, the raw file is used): "
                + ", ".join(inv[label]["raw_and_gz"])
            )
        rep.p(
            f"event dirs: {', '.join(inv[label]['events_dirs']) or 'none'}  \nlogs: {', '.join(inv[label]['logs']) or 'none'}"
        )
    fa, fb = read_freeze(old), read_freeze(new)
    if fa or fb:
        names = sorted(n for n in set(fa) | set(fb) if PACKAGES.match(n))
        rows = [
            [
                n,
                fa.get(n, "-"),
                fb.get(n, "-"),
                "" if fa.get(n) == fb.get(n) else "changed",
            ]
            for n in names
        ]
        rep.h(3, "Package versions (env/pip_freeze.txt)")
        rep.table(["package", "A", "B", ""], rows)
        others = sorted(
            n for n in set(fa) & set(fb) if fa[n] != fb[n] and not PACKAGES.match(n)
        )
        if others:
            rep.p(
                "Other packages that differ: "
                + ", ".join(f"{n} {fa[n]} to {fb[n]}" for n in others[:60])
            )
    else:
        rep.p(
            "No env/pip_freeze.txt in either tree (run capture_env). Versions from preprocess_ab files appear in section 4."
        )
    for label, root in (("A", old), ("B", new)):
        for name in ("torch.txt", "git_head.txt", "nvidia_smi.txt", "jit_caches.txt"):
            path = root / "env" / name
            if path.exists():
                rep.p(f"{label} env/{name}:")
                rep.pre(path.read_text(encoding="utf-8", errors="replace")[:3000])
    return inv


# ---------------------------------------------------------------------------
# 2 server logs
# ---------------------------------------------------------------------------


def _bucket_req(n: int) -> str:
    if n <= 1:
        return "1"
    if n <= 4:
        return "2-4"
    if n <= 8:
        return "5-8"
    if n <= 16:
        return "9-16"
    return ">16"


def _bucket_tok(n: int) -> str:
    if n <= 1024:
        return "<=1k"
    if n <= 4096:
        return "1k-4k"
    if n <= 8192:
        return "4k-8k"
    return ">8k"


def scan_log(path: Path) -> dict:
    found: dict[str, collections.Counter] = {
        label: collections.Counter() for label, _ in LINE_PATTERNS
    }
    counts: dict[str, int] = {label: 0 for label, _ in COUNT_PATTERNS}
    decode: dict[str, list] = collections.defaultdict(list)
    prefill: dict[str, list] = collections.defaultdict(list)
    with open(path, encoding="utf-8", errors="replace") as fp:
        for line in fp:
            for label, pattern in LINE_PATTERNS:
                m = pattern.search(line)
                if m:
                    found[label][" | ".join(g for g in m.groups() if g)] += 1
            for label, pattern in COUNT_PATTERNS:
                if pattern.search(line):
                    counts[label] += 1
            m = DECODE_RE.search(line)
            if m:
                key = f"running {_bucket_req(int(m.group(1)))}, graph {m.group(4)}"
                decode[key].append(float(m.group(5)))
                continue
            m = PREFILL_RE.search(line)
            if m:
                prefill[f"new-token {_bucket_tok(int(m.group(2)))}"].append(
                    float(m.group(4))
                )
    return {
        "found": {k: dict(v) for k, v in found.items()},
        "counts": counts,
        "decode": {k: v for k, v in decode.items()},
        "prefill": {k: v for k, v in prefill.items()},
    }


def section_logs(rep: Report, old: Path, new: Path) -> dict:
    rep.h(2, "2. Server logs")
    logs_a = (
        {p.name: p for p in (old / "logs").glob("*.log")}
        if (old / "logs").is_dir()
        else {}
    )
    logs_b = (
        {p.name: p for p in (new / "logs").glob("*.log")}
        if (new / "logs").is_dir()
        else {}
    )
    out = {}
    if not logs_a and not logs_b:
        rep.p("No logs/*.log in either tree.")
        return out
    rep.p(
        "Lines are matched by log file name. The disaggregated speech server "
        "writes all stage processes to one log, so decode and prefill throughput "
        "lines there mix the thinker and the talker."
    )
    for name in sorted(set(logs_a) | set(logs_b)):
        sa = scan_log(logs_a[name]) if name in logs_a else None
        sb = scan_log(logs_b[name]) if name in logs_b else None
        out[name] = {"A": sa, "B": sb}
        rep.h(3, f"{name} ({'A and B' if sa and sb else 'A only' if sa else 'B only'})")
        rows = []
        for label, _ in LINE_PATTERNS:
            va = (sa or {}).get("found", {}).get(label, {})
            vb = (sb or {}).get("found", {}).get(label, {})
            for value in sorted(set(va) | set(vb)):
                flag = (
                    ""
                    if (value in va) == (value in vb)
                    else ("A only" if value in va else "B only")
                )
                rows.append(
                    [label, value[:200], va.get(value, 0), vb.get(value, 0), flag]
                )
        rep.table(["what", "value", "A count", "B count", ""], rows)
        rows = []
        for label, _ in COUNT_PATTERNS:
            rows.append(
                [
                    label,
                    (sa or {}).get("counts", {}).get(label, "-"),
                    (sb or {}).get("counts", {}).get(label, "-"),
                ]
            )
        rep.table(["count", "A", "B"], rows)
        rows = []
        keys = sorted(
            set((sa or {}).get("decode", {})) | set((sb or {}).get("decode", {}))
        )
        for key in keys:
            va = (sa or {}).get("decode", {}).get(key, [])
            vb = (sb or {}).get("decode", {}).get(key, [])
            rows.append(
                [
                    "decode gen tok/s",
                    key,
                    f"{fmt(med(va), 1)} (n={len(va)})",
                    f"{fmt(med(vb), 1)} (n={len(vb)})",
                    pct(med(va), med(vb)),
                ]
            )
        keys = sorted(
            set((sa or {}).get("prefill", {})) | set((sb or {}).get("prefill", {}))
        )
        for key in keys:
            va = (sa or {}).get("prefill", {}).get(key, [])
            vb = (sb or {}).get("prefill", {}).get(key, [])
            rows.append(
                [
                    "prefill input tok/s",
                    key,
                    f"{fmt(med(va), 0)} (n={len(va)})",
                    f"{fmt(med(vb), 0)} (n={len(vb)})",
                    pct(med(va), med(vb)),
                ]
            )
        if rows:
            rep.table(["metric", "bucket", "A median", "B median", "B vs A"], rows)
    return out


# ---------------------------------------------------------------------------
# 3 benchmark results
# ---------------------------------------------------------------------------


def find_results(root: Path) -> dict[str, Path]:
    out = {}
    for p in walk_files(root):
        if p.name in RESULT_NAMES:
            out[str(p.relative_to(root))] = p
    return out


def load_result(path: Path):
    d = load_json(path)
    if not isinstance(d, dict):
        return None, []
    speed = d.get("speed") if isinstance(d.get("speed"), dict) else {}
    summ = d.get("summary") if isinstance(d.get("summary"), dict) else {}
    metrics = {}
    for k in SUMMARY_KEYS:
        v = speed.get(k, summ.get(k))
        if v is None and k == "accuracy":
            v = (
                (d.get("accuracy") or {}).get("overall_accuracy")
                if isinstance(d.get("accuracy"), dict)
                else None
            )
        if isinstance(v, (int, float)):
            metrics[k] = v
    records = d.get("per_sample") or d.get("per_request") or []
    if not metrics and not records:
        return None, []
    return metrics, records


def rec_id(r: dict):
    return (
        r.get("sample_id") or r.get("question_id") or r.get("id") or r.get("request_id")
    )


def family_key(rel: str) -> str:
    return re.sub(r"_\d+(?=/|$)", "", rel)


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def per_sample_side(runs: list[list[dict]]) -> dict:
    """id -> medians of latency, audio, tokens, rtf and the correctness majority."""
    acc: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for records in runs:
        for r in records:
            rid = rec_id(r)
            if rid is None:
                continue
            a = acc[rid]
            a["latency_s"].append(_num(r.get("latency_s")))
            a["audio_duration_s"].append(_num(r.get("audio_duration_s")))
            a["completion_tokens"].append(_num(r.get("completion_tokens")))
            a["prompt_tokens"].append(_num(r.get("prompt_tokens")))
            a["rtf"].append(_num(r.get("rtf")))
            if "is_correct" in r:
                a["is_correct"].append(1 if r.get("is_correct") else 0)
            if r.get("predicted") is not None:
                a["predicted"].append(str(r.get("predicted")))
            if r.get("expected") is not None:
                a["expected"] = r.get("expected")
            if r.get("is_success") is False:
                a["failed"].append(1)
    out = {}
    for rid, a in acc.items():
        out[rid] = {
            "latency_s": med(a["latency_s"]),
            "audio_duration_s": med(a["audio_duration_s"]),
            "completion_tokens": med(a["completion_tokens"]),
            "prompt_tokens": med(a["prompt_tokens"]),
            "rtf": med(a["rtf"]),
            "correct": (
                (sum(a["is_correct"]) / len(a["is_correct"]))
                if a["is_correct"]
                else None
            ),
            "predicted": (
                collections.Counter(a["predicted"]).most_common(1)[0][0]
                if a["predicted"]
                else None
            ),
            "expected": a.get("expected"),
            "failed": len(a["failed"]),
            "runs": len(a["latency_s"]),
        }
    return out


def section_bench(rep: Report, old: Path, new: Path, top: int) -> dict:
    rep.h(2, "3. Benchmark results")
    ra, rb = find_results(old), find_results(new)
    families: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: {"A": [], "B": []}
    )
    for label, found in (("A", ra), ("B", rb)):
        for rel, path in found.items():
            metrics, records = load_result(path)
            if metrics is None:
                continue
            families[family_key(rel)][label].append((rel, metrics, records))
    if not families:
        rep.p("No result files found.")
        return {}
    rep.p(
        "Runs are grouped by path with trailing _N run numbers removed. A run "
        "family present on both sides is compared by the median over its runs. "
        "Per-sample rows pair the same sample id on both sides."
    )
    out = {}
    for fam in sorted(families):
        side = families[fam]
        rep.h(3, f"{fam} (A runs {len(side['A'])}, B runs {len(side['B'])})")
        keys = [
            k for k in SUMMARY_KEYS if any(k in m for _, m, _ in side["A"] + side["B"])
        ]
        rows = []
        for k in keys:
            nd = 0 if k in ("completed_requests", "failed_requests") else 4
            va = [m.get(k) for _, m, _ in side["A"] if k in m]
            vb = [m.get(k) for _, m, _ in side["B"] if k in m]

            def _show(vals):
                if not vals:
                    return "-"
                text = fmt(med(vals), nd)
                if len(vals) > 1:
                    text += " [" + ", ".join(fmt(v, nd) for v in vals) + "]"
                return text

            rows.append(
                [k, _show(va), _show(vb), pct(med(va), med(vb)) if va and vb else "-"]
            )
        rep.table(["metric", "A median [runs]", "B median [runs]", "B vs A"], rows)
        out[fam] = {
            "A": [(rel, m) for rel, m, _ in side["A"]],
            "B": [(rel, m) for rel, m, _ in side["B"]],
        }
        if not side["A"] or not side["B"]:
            continue
        pa = per_sample_side([r for _, _, r in side["A"]])
        pb = per_sample_side([r for _, _, r in side["B"]])
        ids = [i for i in pa if i in pb]
        if not ids:
            rep.p("No shared sample ids.")
            continue
        n = len(ids)
        has_rtf = any(
            pa[i]["rtf"] is not None and pb[i]["rtf"] is not None for i in ids
        )
        lat_ratio = [
            pb[i]["latency_s"] / pa[i]["latency_s"]
            for i in ids
            if pa[i]["latency_s"] and pb[i]["latency_s"]
        ]
        rep.p(
            f"{n} shared samples. Latency ratio B/A per sample: median {fmt(med(lat_ratio))}, "
            f"p10 {fmt(pctl(lat_ratio, 0.1))}, p90 {fmt(pctl(lat_ratio, 0.9))}. "
            f"Sum of latency A {fmt(sum(pa[i]['latency_s'] or 0 for i in ids), 1)} s, B {fmt(sum(pb[i]['latency_s'] or 0 for i in ids), 1)} s. "
            f"Slowest sample A {max(ids, key=lambda i: pa[i]['latency_s'] or 0)}, B {max(ids, key=lambda i: pb[i]['latency_s'] or 0)}."
        )
        if has_rtf:
            contrib = {i: ((pb[i]["rtf"] or 0) - (pa[i]["rtf"] or 0)) / n for i in ids}
            total = sum(contrib.values())
            ordered = sorted(ids, key=lambda i: -abs(contrib[i]))
            top3 = ordered[:3]
            rep.p(
                f"rtf_mean over shared samples: A {fmt(mean([pa[i]['rtf'] for i in ids]), 4)}, "
                f"B {fmt(mean([pb[i]['rtf'] for i in ids]), 4)}, difference {total:+.4f}. "
                f"Largest contributions: "
                + ", ".join(f"{i} {contrib[i]:+.4f}" for i in top3)
                + f" (together {sum(contrib[i] for i in top3):+.4f}). "
                f"Audio seconds A {fmt(sum(pa[i]['audio_duration_s'] or 0 for i in ids), 1)}, B {fmt(sum(pb[i]['audio_duration_s'] or 0 for i in ids), 1)}."
            )
        rows = []
        for i in sorted(
            ids,
            key=lambda i: -abs((pb[i]["latency_s"] or 0) - (pa[i]["latency_s"] or 0)),
        )[:top]:
            a, b = pa[i], pb[i]
            rows.append(
                [
                    i,
                    fmt(a["latency_s"], 2),
                    fmt(b["latency_s"], 2),
                    delta(a["latency_s"], b["latency_s"], 2),
                    fmt(a["audio_duration_s"], 2),
                    fmt(b["audio_duration_s"], 2),
                    f"{fmt(a['completion_tokens'], 0)}/{fmt(b['completion_tokens'], 0)}",
                    f"{fmt(a['prompt_tokens'], 0)}/{fmt(b['prompt_tokens'], 0)}",
                    fmt(a["rtf"], 2),
                    fmt(b["rtf"], 2),
                    (
                        f"{fmt(a['correct'], 2)}/{fmt(b['correct'], 2)}"
                        if a["correct"] is not None
                        else "-"
                    ),
                ]
            )
        rep.p(f"Top {min(top, n)} samples by absolute latency change:")
        rep.table(
            [
                "sample",
                "lat A",
                "lat B",
                "d lat",
                "audio A",
                "audio B",
                "out tok A/B",
                "prompt tok A/B",
                "rtf A",
                "rtf B",
                "correct A/B",
            ],
            rows,
        )
        flips = [
            i
            for i in ids
            if pa[i]["correct"] is not None
            and pb[i]["correct"] is not None
            and pa[i]["correct"] != pb[i]["correct"]
        ]
        if flips:
            rows = [
                [
                    i,
                    pa[i]["expected"],
                    pa[i]["predicted"],
                    pb[i]["predicted"],
                    fmt(pa[i]["correct"], 2),
                    fmt(pb[i]["correct"], 2),
                ]
                for i in flips
            ]
            rep.p("Answer flips (correctness share over runs differs):")
            rep.table(
                ["sample", "expected", "pred A", "pred B", "correct A", "correct B"],
                rows,
            )
        failed = [
            (i, pa[i]["failed"], pb[i]["failed"])
            for i in ids
            if pa[i]["failed"] or pb[i]["failed"]
        ]
        if failed:
            rep.p(
                "Failed requests: "
                + ", ".join(f"{i} A {fa} B {fb}" for i, fa, fb in failed)
            )
    return out


# ---------------------------------------------------------------------------
# 4 preprocessing A/B
# ---------------------------------------------------------------------------


def section_preprocess(rep: Report, old: Path, new: Path) -> dict:
    rep.h(2, "4. CPU preprocessing A/B")
    files = []
    for label, root in (("A", old), ("B", new)):
        for p in sorted(root.glob("preprocess_ab_*.json")):
            d = load_json(p)
            if isinstance(d, dict) and "per_video" in d:
                files.append((f"{label}:{p.name}", d))
    if not files:
        rep.p("No preprocess_ab_*.json files.")
        return {}
    rows = []
    for name, d in files:
        env = d.get("env", {})
        s = d.get("summary", {})
        rows.append(
            [
                name,
                env.get("torch", "-"),
                env.get("torchvision", "-"),
                env.get("torchcodec", "-"),
                env.get("transformers", "-"),
                s.get("videos", "-"),
                fmt(s.get("load_s_median")),
                fmt(s.get("processor_s_median")),
                fmt(s.get("total_s_median")),
                fmt(s.get("full_s_mean")),
                (s.get("full_error") or "")[:60],
            ]
        )
    rep.table(
        [
            "file",
            "torch",
            "torchvision",
            "torchcodec",
            "transformers",
            "videos",
            "load med s",
            "processor med s",
            "total med s",
            "full mean s",
            "full error",
        ],
        rows,
    )
    ids = []
    per = {}
    for name, d in files:
        per[name] = {r["sample_id"]: r for r in d.get("per_video", [])}
        for r in d.get("per_video", []):
            if r["sample_id"] not in ids:
                ids.append(r["sample_id"])
    rows = []
    for i in ids:
        row = [i]
        for name, _ in files:
            r = per[name].get(i)
            row.append(
                f"{fmt(r['load_s'], 2)}/{fmt(r['processor_s'], 2)}/{fmt(r['total_s'], 2)}"
                + (f"/{fmt(r['full_s'], 2)}" if r and "full_s" in r else "")
                if r
                else "-"
            )
        rows.append(row)
    rep.p(
        "Per video: load/processor/total seconds, plus full preprocessor seconds when measured."
    )
    rep.table(["video"] + [name for name, _ in files], rows)
    return {name: d.get("summary") for name, d in files}


# ---------------------------------------------------------------------------
# 5 request profiler events
# ---------------------------------------------------------------------------


def load_timelines(event_dir: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = collections.defaultdict(list)
    for path in sorted(event_dir.glob("events_*.jsonl")):
        with open(path, encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                rid = ev.get("request_id")
                if rid:
                    grouped[rid].append(ev)
    for evs in grouped.values():
        evs.sort(key=lambda e: e.get("timestamp_ns", 0))
    return grouped


def intervals_for(evs: list[dict]) -> list[tuple[str, str, float]]:
    opens: dict[str, list] = collections.defaultdict(list)
    closes: dict[str, list] = collections.defaultdict(list)
    for opener, closer in INTERVALS:
        opens[opener].append((opener, closer))
        closes[closer].append((opener, closer))
    pending: dict[tuple, list[int]] = collections.defaultdict(list)
    out = []
    for ev in evs:
        name = ev.get("event_name")
        stage = ev.get("stage", "unknown")
        ts = int(ev.get("timestamp_ns", 0))
        if name in opens:
            for opener, closer in opens[name]:
                pending[(stage, opener, closer)].append(ts)
        if name in closes:
            for opener, closer in closes[name]:
                stack = pending.get((stage, opener, closer))
                if stack:
                    out.append(
                        (stage, f"{opener}->{closer}", (ts - stack.pop(0)) / 1e6)
                    )
    return out


def hops_for(evs: list[dict]) -> list[tuple[str, str, str, float]]:
    pending: dict[tuple, list[int]] = collections.defaultdict(list)
    out = []
    for ev in evs:
        name = ev.get("event_name")
        md = ev.get("metadata") or {}
        ts = int(ev.get("timestamp_ns", 0))
        stage = ev.get("stage", "unknown")
        if name == "stage_hop_sent":
            pending[(stage, md.get("to_stage", "?"), "payload", None)].append(ts)
        elif name == "stage_stream_chunk_sent":
            pending[
                (stage, md.get("to_stage", "?"), "stream_chunk", md.get("chunk_id"))
            ].append(ts)
        elif name == "stage_input_received":
            src = md.get("from_stage")
            if src and src != "coordinator":
                stack = pending.get((src, stage, "payload", None))
                if stack:
                    out.append((src, stage, "payload", (ts - stack.pop(0)) / 1e6))
        elif name == "stage_stream_chunk_received":
            src = md.get("from_stage")
            if src:
                stack = pending.get((src, stage, "stream_chunk", md.get("chunk_id")))
                if stack:
                    out.append((src, stage, "stream_chunk", (ts - stack.pop(0)) / 1e6))
    return out


def request_row(rid: str, evs: list[dict]) -> dict:
    t0 = evs[0]["timestamp_ns"]
    admission = next(
        (e["timestamp_ns"] for e in evs if e.get("event_name") == "request_admission"),
        t0,
    )
    terminal = next(
        (e["timestamp_ns"] for e in evs if e.get("event_name") == "terminal_response"),
        evs[-1]["timestamp_ns"],
    )
    pre_start = next(
        (e["timestamp_ns"] for e in evs if e.get("event_name") == "preprocess_start"),
        None,
    )
    cols: dict[str, float] = collections.defaultdict(float)
    if pre_start is not None:
        cols["wait.preprocess"] = (pre_start - t0) / 1e9
    for stage, name, ms in intervals_for(evs):
        if name == "preprocess_start->preprocess_end":
            cols["preprocess"] += ms / 1e3
        elif name == "encoder_start->encoder_end":
            cols[f"{stage}.encoder"] += ms / 1e3
        elif name == "stage_input_received->stage_complete":
            cols[stage] += ms / 1e3
        elif name == "scheduler_queue_enter->scheduler_prefill_start":
            cols[f"{stage}.wait"] += ms / 1e3
        elif name == "scheduler_prefill_start->scheduler_prefill_end":
            cols[f"{stage}.prefill"] += ms / 1e3
        elif name == "scheduler_prefill_end->stage_complete":
            cols[f"{stage}.decode"] += ms / 1e3
    return {
        "request_id": rid,
        "t0_ns": t0,
        "admission_ns": admission,
        "total_s": (terminal - t0) / 1e9,
        "events": len(evs),
        "cols": dict(cols),
    }


def events_view(event_dir: Path) -> dict:
    tl = load_timelines(event_dir)
    stage_iv: dict[tuple, list[float]] = collections.defaultdict(list)
    hop_iv: dict[tuple, list[float]] = collections.defaultdict(list)
    rows = []
    for rid, evs in tl.items():
        for stage, name, ms in intervals_for(evs):
            stage_iv[(stage, name)].append(ms)
        for src, dst, kind, ms in hops_for(evs):
            hop_iv[(src, dst, kind)].append(ms)
        rows.append(request_row(rid, evs))
    rows.sort(key=lambda r: r["admission_ns"])
    return {"requests": len(tl), "stage": stage_iv, "hops": hop_iv, "rows": rows}


def _iv_rows(a: dict, b: dict, keyfmt, top: int) -> list[list]:
    rows = []
    for key in set(a) | set(b):
        va, vb = a.get(key, []), b.get(key, [])
        ma, mb = mean(va), mean(vb)
        weight = (
            abs((mb or 0) - (ma or 0)) * min(len(va), len(vb))
            if va and vb
            else (sum(va) + sum(vb))
        )
        rows.append(
            (
                weight,
                keyfmt(key)
                + [
                    len(va),
                    len(vb),
                    fmt(ma, 1),
                    fmt(mb, 1),
                    delta(ma, mb, 1),
                    pct(ma, mb),
                    fmt(med(va), 1),
                    fmt(med(vb), 1),
                    fmt(pctl(va, 0.95), 1),
                    fmt(pctl(vb, 0.95), 1),
                    fmt(sum(va), 0),
                    fmt(sum(vb), 0),
                ],
            )
        )
    rows.sort(key=lambda r: -r[0])
    return [r[1] for r in rows[:top]]


def section_events(rep: Report, old: Path, new: Path, top: int) -> dict:
    rep.h(2, "5. Request profiler events")
    dirs_a = {
        str(p.parent.relative_to(old))
        for p in walk_files(old)
        if re.match(r"events_.*\.jsonl$", p.name)
    }
    dirs_b = {
        str(p.parent.relative_to(new))
        for p in walk_files(new)
        if re.match(r"events_.*\.jsonl$", p.name)
    }
    out = {}
    if not dirs_a and not dirs_b:
        rep.p("No event directories.")
        return out
    rep.p(
        "Intervals are milliseconds inside one stage process, hops are the "
        "transfer time between stages. Per-request rows pair the i-th admitted "
        "request on A with the i-th on B, which is the sample order the "
        "benchmark submits in. Columns are seconds."
    )
    for rel in sorted(dirs_a | dirs_b):
        va = events_view(old / rel) if rel in dirs_a else None
        vb = events_view(new / rel) if rel in dirs_b else None
        rep.h(
            3,
            f"{rel} (A requests {va['requests'] if va else '-'}, B requests {vb['requests'] if vb else '-'})",
        )
        if not va or not vb:
            v = va or vb
            rows = _iv_rows(v["stage"], {}, lambda k: [k[0], k[1]], top)
            rep.table(
                [
                    "stage",
                    "interval",
                    "n A",
                    "n B",
                    "mean A",
                    "mean B",
                    "d mean",
                    "%",
                    "p50 A",
                    "p50 B",
                    "p95 A",
                    "p95 B",
                    "sum A",
                    "sum B",
                ],
                rows,
            )
            continue
        rows = _iv_rows(va["stage"], vb["stage"], lambda k: [k[0], k[1]], top)
        rep.p("Stage intervals sorted by absolute change in mean times count:")
        rep.table(
            [
                "stage",
                "interval",
                "n A",
                "n B",
                "mean A",
                "mean B",
                "d mean",
                "%",
                "p50 A",
                "p50 B",
                "p95 A",
                "p95 B",
                "sum A",
                "sum B",
            ],
            rows,
        )
        rows = _iv_rows(
            va["hops"], vb["hops"], lambda k: [f"{k[0]} to {k[1]}", k[2]], top
        )
        rep.p("Hops:")
        rep.table(
            [
                "hop",
                "kind",
                "n A",
                "n B",
                "mean A",
                "mean B",
                "d mean",
                "%",
                "p50 A",
                "p50 B",
                "p95 A",
                "p95 B",
                "sum A",
                "sum B",
            ],
            rows,
        )
        ra, rb = va["rows"], vb["rows"]
        cols = []
        for r in ra + rb:
            for c in r["cols"]:
                if c not in cols:
                    cols.append(c)
        cols.sort(key=lambda c: (c.split(".")[0], c))
        n = min(len(ra), len(rb))
        pairs = list(zip(ra[:n], rb[:n]))
        ta = [r["total_s"] for r in ra]
        tb = [r["total_s"] for r in rb]
        span_a = (
            (ra[-1]["admission_ns"] - ra[0]["admission_ns"]) / 1e9 if len(ra) > 1 else 0
        )
        span_b = (
            (rb[-1]["admission_ns"] - rb[0]["admission_ns"]) / 1e9 if len(rb) > 1 else 0
        )
        rep.p(
            f"End to end seconds per request: A median {fmt(med(ta), 2)} max {fmt(max(ta), 2)} sum {fmt(sum(ta), 1)}, "
            f"B median {fmt(med(tb), 2)} max {fmt(max(tb), 2)} sum {fmt(sum(tb), 1)}. "
            f"Admission span A {fmt(span_a, 1)} s, B {fmt(span_b, 1)} s."
        )
        rows = []
        for c in cols:
            xa = [r["cols"].get(c) for r in ra if c in r["cols"]]
            xb = [r["cols"].get(c) for r in rb if c in r["cols"]]
            rows.append(
                [
                    c,
                    len(xa),
                    len(xb),
                    fmt(med(xa), 3),
                    fmt(med(xb), 3),
                    delta(med(xa), med(xb), 3),
                    pct(med(xa), med(xb)),
                    fmt(sum(xa), 1),
                    fmt(sum(xb), 1),
                    delta(sum(xa), sum(xb), 1),
                ]
            )
        rep.p("Per-request components (seconds), medians and sums over all requests:")
        rep.table(
            [
                "component",
                "n A",
                "n B",
                "median A",
                "median B",
                "d median",
                "%",
                "sum A",
                "sum B",
                "d sum",
            ],
            rows,
        )
        rows = []
        for idx, (a, b) in enumerate(
            sorted(pairs, key=lambda p: -abs(p[1]["total_s"] - p[0]["total_s"]))[:top]
        ):
            rank = pairs.index((a, b)) + 1
            row = [
                rank,
                fmt(a["total_s"], 2),
                fmt(b["total_s"], 2),
                delta(a["total_s"], b["total_s"], 2),
            ]
            for c in cols:
                row.append(f"{fmt(a['cols'].get(c), 2)}/{fmt(b['cols'].get(c), 2)}")
            rows.append(row)
        rep.p(
            f"Per-request rows with the largest end to end change (A/B seconds per component), rank is admission order:"
        )
        rep.table(["rank", "total A", "total B", "d total"] + cols, rows)
        last_a = (
            max(range(len(ra)), key=lambda i: ra[i]["t0_ns"] + ra[i]["total_s"] * 1e9)
            + 1
        )
        last_b = (
            max(range(len(rb)), key=lambda i: rb[i]["t0_ns"] + rb[i]["total_s"] * 1e9)
            + 1
        )
        rep.p(f"Last request to finish: admission rank {last_a} on A, {last_b} on B.")
        out[rel] = {"A_requests": va["requests"], "B_requests": vb["requests"]}
    return out


# ---------------------------------------------------------------------------
# 6 torch traces
# ---------------------------------------------------------------------------


def section_traces(rep: Report, old: Path, new: Path, top: int, workers: int) -> None:
    rep.h(2, "6. Torch traces")
    dirs_a = {
        p.name: p for p in old.iterdir() if p.is_dir() and p.name.startswith("traces_")
    }
    dirs_b = {
        p.name: p for p in new.iterdir() if p.is_dir() and p.name.startswith("traces_")
    }
    if not dirs_a and not dirs_b:
        rep.p("No traces_* directories.")
        return
    files = []
    for d in list(dirs_a.values()) + list(dirs_b.values()):
        files.extend(trace_kernels.trace_files(d))
    print(
        f"trace files: {len(files)}, computing missing stats with {workers} workers",
        file=sys.stderr,
    )
    trace_kernels.ensure_stats(files, workers)
    rep.p(
        "Kernel time per stage process over the whole profile window (8 samples "
        "at c1 then 16 at c16). A_n and B_n are launch counts, A_avg and B_avg "
        "microseconds per launch. Kernels only on one side mark a backend or "
        "kernel selection change. Busy is kernel time over the GPU span."
    )
    for name in sorted(set(dirs_a) | set(dirs_b)):
        rep.h(3, name)
        if name in dirs_a and name in dirs_b:
            rep.pre(
                trace_kernels.format_diff(
                    trace_kernels.dir_stats(dirs_a[name]),
                    trace_kernels.dir_stats(dirs_b[name]),
                    top,
                )
            )
        else:
            side = "A" if name in dirs_a else "B"
            rep.p(f"Only in {side}.")
            rep.pre(
                trace_kernels.format_summary(
                    trace_kernels.dir_stats((dirs_a or dirs_b)[name]), top
                )
            )


# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("old")
    p.add_argument("new")
    p.add_argument("--out", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-traces", action="store_true")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args(argv)
    old, new = Path(args.old).resolve(), Path(args.new).resolve()
    out_dir = Path(args.out) if args.out else new / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = Report()
    rep.h(1, "Previous image (A) versus current image (B)")
    data = {}
    data["inventory"] = section_inventory(rep, old, new)
    data["logs"] = section_logs(rep, old, new)
    data["bench"] = section_bench(rep, old, new, args.top)
    data["preprocess"] = section_preprocess(rep, old, new)
    data["events"] = section_events(rep, old, new, args.top)
    if args.skip_traces:
        rep.h(2, "6. Torch traces")
        rep.p("Skipped (--skip-traces).")
    else:
        section_traces(rep, old, new, args.top, args.workers)
    (out_dir / "report.md").write_text(rep.text(), encoding="utf-8")
    with open(out_dir / "tables.json", "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=1, default=str)
    print(
        f"report: {out_dir / 'report.md'} ({(out_dir / 'report.md').stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
