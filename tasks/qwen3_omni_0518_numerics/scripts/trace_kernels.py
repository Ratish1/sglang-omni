# SPDX-License-Identifier: Apache-2.0
"""Summarise and diff CUDA kernel time per stage process from omni torch traces.

Usage:
    python trace_kernels.py stats TRACE_DIR [--workers N]
    python trace_kernels.py summary TRACE_DIR [--top 25] [--workers N]
    python trace_kernels.py diff TRACE_DIR_A TRACE_DIR_B [--top 40] [--workers N]

TRACE_DIR holds the files a /start_profile run with enable_torch true wrote
through the template TRACE_DIR/{stage}, one <stage>_pid<pid>_rank<r>.trace.json
(or .gz) per stage process. The traces are streamed line by line, so a
multi-GB raw trace needs no more memory than a small one, and the per-file
statistics are cached next to the trace as <trace>.stats.json. stats only
builds those caches. summary prints, per stage process, the total CUDA kernel
time, the GPU span and busy fraction, the CUDA runtime launch counts, the time
per kernel family and the top kernels. diff aligns two runs by stage process
and kernel name and prints what moved, which is the per-component attribution
of a speed change between two stacks when both runs served the same requests.
When both <base>.trace.json and <base>.trace.json.gz exist the raw file is
read, because gzip -f removes the raw file only after a complete compression.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import multiprocessing
import re
import sys
from pathlib import Path

FAMILIES = (
    (
        "attention",
        re.compile(r"flash|attn|fmha|sdpa|softmax|decode_kernel|paged", re.I),
    ),
    ("moe", re.compile(r"moe|expert|topk|grouped_gemm|silu_and_mul|fused_moe", re.I)),
    (
        "gemm",
        re.compile(
            r"gemm|cutlass|matmul|nvjet|ampere_|sm90|sm80|fp8|deep_gemm|wgmma", re.I
        ),
    ),
    ("norm", re.compile(r"norm|layer_norm|rms", re.I)),
    ("rope", re.compile(r"rope|rotary", re.I)),
    ("sampling", re.compile(r"sampl|argmax|logprob|top_p|top_k|multinomial", re.I)),
    (
        "copy",
        re.compile(
            r"copy|memcpy|memset|cat_|elementwise|vectorized|fill|index|gather|scatter",
            re.I,
        ),
    ),
    ("reduce", re.compile(r"reduce|all_reduce|nccl|allgather|all_gather", re.I)),
)

GPU_CATS = ("kernel", "gpu_memcpy", "gpu_memset")
RUNTIME_CATS = ("cuda_runtime", "cuda_driver")
ANNOTATION_CATS = ("user_annotation", "gpu_user_annotation")
LAUNCH_NAMES = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cudaGraphLaunch",
    "cudaMemcpyAsync",
    "cudaMemcpy",
    "cudaMemsetAsync",
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cudaStreamWaitEvent",
)
CPU_OPS_KEPT = 400

_STR = re.compile(r'"(?:[^"\\]|\\.)*"')
_KV = re.compile(r'"(ph|cat|pid|tid|ts|dur)":\s*(?:"([^"]*)"|([-+0-9.eE]+))')
_NAME = re.compile(r'"name":\s*"((?:[^"\\]|\\.)*)"')


def _family(name: str) -> str:
    for family, pattern in FAMILIES:
        if pattern.search(name):
            return family
    return "other"


def short_name(name: str) -> str:
    """Kernel name without template arguments and parameter lists."""
    out = []
    templates = 0
    params = 0
    for ch in name:
        if ch == "<":
            templates += 1
        elif ch == ">":
            templates = max(0, templates - 1)
        elif ch == "(":
            params += 1
        elif ch == ")":
            params = max(0, params - 1)
        elif templates == 0 and params == 0:
            out.append(ch)
    text = "".join(out).strip()
    if text.startswith("void "):
        text = text[5:]
    return text.split()[-1] if text else name


def _parse_kv(event: dict, line: str) -> dict:
    for m in _KV.finditer(line):
        event[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    if '"name"' in line:
        m = _NAME.search(line)
        if m:
            raw = m.group(1)
            event["name"] = json.loads('"' + raw + '"') if "\\" in raw else raw
    return event


def iter_events(path: Path):
    """Yield complete-duration (ph X) events from a chrome trace, streaming.

    Handles the kineto layout (one field per line, args on their own lines),
    the older layout with several fields per line, and compact one-line
    events. Only ph, cat, name, pid, tid, ts and dur are extracted.
    """
    opener = gzip.open if path.name.endswith(".gz") else open
    depth = 0
    current: dict | None = None
    with opener(path, "rt", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            text = line.strip()
            if not text:
                continue
            if "{" in text or "}" in text:
                bare = _STR.sub("", text)
                delta = bare.count("{") - bare.count("}")
            else:
                delta = 0
            if depth == 1:
                if '"ph"' in text:
                    event = _parse_kv({}, text)
                    if delta <= 0:
                        if event.get("ph") == "X":
                            yield event
                    else:
                        current = event
                elif delta > 0:
                    current = {}
            elif depth == 2 and current is not None:
                _parse_kv(current, text)
                if depth + delta <= 1:
                    if current.get("ph") == "X":
                        yield current
                    current = None
            depth += delta


def key_of(path: Path) -> str:
    """Stage process key: file name without pid, so runs align across images."""
    name = path.name
    for suffix in (".trace.json.gz", ".trace.json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"_pid\d+", "", name)


def _span(span: list, start: float, end: float) -> None:
    if span[0] is None or start < span[0]:
        span[0] = start
    if span[1] is None or end > span[1]:
        span[1] = end


def compute_stats(path: Path) -> dict:
    stats = {
        "file": path.name,
        "key": key_of(path),
        "bytes": path.stat().st_size,
        "events": 0,
        "cats": {},
        "kernels": {},
        "streams": {},
        "runtime": {},
        "cpu_ops": {},
        "annotations": {},
        "span": [None, None],
        "gpu_span": [None, None],
        "error": None,
    }
    cats = stats["cats"]
    kernels = stats["kernels"]
    streams = stats["streams"]
    runtime = stats["runtime"]
    cpu_ops: dict[str, list] = {}
    annotations = stats["annotations"]
    try:
        for ev in iter_events(path):
            cat = ev.get("cat", "?")
            name = ev.get("name", "?")
            try:
                ts = float(ev.get("ts", 0))
                dur = float(ev.get("dur", 0))
            except (TypeError, ValueError):
                continue
            stats["events"] += 1
            c = cats.setdefault(cat, [0, 0.0])
            c[0] += 1
            c[1] += dur
            _span(stats["span"], ts, ts + dur)
            if cat in GPU_CATS:
                k = kernels.setdefault(name, [0.0, 0])
                k[0] += dur
                k[1] += 1
                stream = f"{ev.get('pid', '?')}/{ev.get('tid', '?')}"
                s = streams.setdefault(stream, [0.0, 0, None, None])
                s[0] += dur
                s[1] += 1
                if s[2] is None or ts < s[2]:
                    s[2] = ts
                if s[3] is None or ts + dur > s[3]:
                    s[3] = ts + dur
                _span(stats["gpu_span"], ts, ts + dur)
            elif cat in RUNTIME_CATS:
                r = runtime.setdefault(name, [0.0, 0])
                r[0] += dur
                r[1] += 1
            elif cat == "cpu_op":
                o = cpu_ops.setdefault(name, [0.0, 0])
                o[0] += dur
                o[1] += 1
            elif cat in ANNOTATION_CATS:
                a = annotations.setdefault(name, [0.0, 0])
                a[0] += dur
                a[1] += 1
    except (EOFError, OSError, gzip.BadGzipFile) as exc:
        stats["error"] = f"{type(exc).__name__}: {exc}"
    stats["cpu_ops"] = dict(
        sorted(cpu_ops.items(), key=lambda kv: -kv[1][0])[:CPU_OPS_KEPT]
    )
    stats["cpu_ops_distinct"] = len(cpu_ops)
    return stats


def cache_path(path: Path) -> Path:
    return path.with_name(path.name + ".stats.json")


def _cache_fresh(path: Path) -> bool:
    cache = cache_path(path)
    return cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime


def file_stats(path: Path, refresh: bool = False) -> dict:
    if not refresh and _cache_fresh(path):
        try:
            with open(cache_path(path), encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, ValueError):
            pass
    stats = compute_stats(path)
    try:
        with open(cache_path(path), "w", encoding="utf-8") as fp:
            json.dump(stats, fp)
    except OSError:
        pass
    return stats


def _compute_and_cache(path: Path) -> str:
    file_stats(path, refresh=True)
    return path.name


def trace_files(trace_dir: Path) -> list[Path]:
    """Trace files of a directory, the raw file preferred over its .gz twin."""
    files: dict[str, Path] = {}
    for p in sorted(Path(trace_dir).iterdir()):
        if p.name.endswith(".trace.json"):
            files[p.name] = p
        elif p.name.endswith(".trace.json.gz"):
            files.setdefault(p.name[:-3], p)
    return list(files.values())


def ensure_stats(paths: list[Path], workers: int = 1) -> None:
    todo = [p for p in paths if not _cache_fresh(p)]
    if not todo:
        return
    if workers > 1 and len(todo) > 1:
        with multiprocessing.Pool(min(workers, len(todo))) as pool:
            for name in pool.imap_unordered(_compute_and_cache, todo):
                print(f"  stats: {name}", file=sys.stderr)
    else:
        for p in todo:
            _compute_and_cache(p)
            print(f"  stats: {p.name}", file=sys.stderr)


def dir_stats(trace_dir: str | Path, workers: int = 1) -> dict[str, dict]:
    files = trace_files(Path(trace_dir))
    if not files:
        raise SystemExit(f"no *.trace.json files in {trace_dir}")
    ensure_stats(files, workers)
    out: dict[str, dict] = {}
    for p in files:
        st = file_stats(p)
        key = st["key"]
        n = 2
        while key in out:
            key = f"{st['key']}#{n}"
            n += 1
        out[key] = st
    return out


def _family_totals(kernels: dict) -> collections.Counter:
    fam: collections.Counter = collections.Counter()
    for name, (us, _) in kernels.items():
        fam[_family(name)] += us
    return fam


def _short_totals(kernels: dict) -> dict[str, list]:
    out: dict[str, list] = {}
    for name, (us, n) in kernels.items():
        s = out.setdefault(short_name(name), [0.0, 0])
        s[0] += us
        s[1] += n
    return out


def _kernel_total(st: dict) -> float:
    return sum(v[0] for v in st["kernels"].values())


def _kernel_count(st: dict) -> int:
    return sum(v[1] for v in st["kernels"].values())


def _gpu_span_s(st: dict) -> float | None:
    a, b = st["gpu_span"]
    return (b - a) / 1e6 if a is not None and b is not None else None


def _busy(st: dict) -> str:
    span = _gpu_span_s(st)
    if not span:
        return "-"
    return f"{100 * _kernel_total(st) / 1e6 / span:.1f}%"


def _launches(st: dict) -> str:
    parts = []
    for name in LAUNCH_NAMES:
        if name in st["runtime"]:
            us, n = st["runtime"][name]
            parts.append(f"{name} {n} ({us / 1e3:.0f} ms)")
    return ", ".join(parts) if parts else "-"


def _cats(st: dict) -> str:
    return ", ".join(
        f"{cat} {n}"
        for cat, (n, _) in sorted(st["cats"].items(), key=lambda kv: -kv[1][0])
    )


def format_summary(stats: dict[str, dict], top: int) -> str:
    lines = []
    for key in sorted(stats):
        st = stats[key]
        kernels = st["kernels"]
        total = _kernel_total(st)
        fam = _family_totals(kernels)
        lines.append(
            f"\n== {key}: CUDA kernel time {total / 1e3:.1f} ms in {_kernel_count(st)} launches"
            + (f"  [parse error: {st['error']}]" if st.get("error") else "")
        )
        span = _gpu_span_s(st)
        lines.append(
            f"  gpu span {span:.1f} s, busy {_busy(st)}, file {st['bytes'] / 1e6:.0f} MB, events {st['events']}"
            if span
            else f"  no gpu events, file {st['bytes'] / 1e6:.0f} MB, events {st['events']}"
        )
        lines.append(f"  runtime: {_launches(st)}")
        lines.append(f"  categories: {_cats(st)}")
        if total:
            lines.append(
                "  by family: "
                + ", ".join(
                    f"{k} {v / 1e3:.1f} ms ({100 * v / total:.1f}%)"
                    for k, v in fam.most_common()
                )
            )
        lines.append(f"  {'total_ms':>9} {'count':>7} {'avg_us':>8}  kernel")
        for name, (us, n) in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:top]:
            lines.append(f"  {us / 1e3:9.2f} {n:7d} {us / n:8.1f}  {name[:110]}")
    return "\n".join(lines) + "\n"


def format_diff(a: dict[str, dict], b: dict[str, dict], top: int) -> str:
    lines = []
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            side = "A" if key in a else "B"
            st = a.get(key) or b.get(key)
            lines.append(
                f"\n== {key}: only in {side}, {_kernel_total(st) / 1e3:.1f} ms in {_kernel_count(st)} launches"
            )
            continue
        sa, sb = a[key], b[key]
        ka, kb = sa["kernels"], sb["kernels"]
        ta, tb = _kernel_total(sa), _kernel_total(sb)
        lines.append(
            f"\n== {key}: A {ta / 1e3:.1f} ms in {_kernel_count(sa)} launches, "
            f"B {tb / 1e3:.1f} ms in {_kernel_count(sb)} launches, "
            f"B-A {(tb - ta) / 1e3:+.1f} ms ({100 * (tb - ta) / ta if ta else 0:+.1f}%)"
        )
        for label, st in (("A", sa), ("B", sb)):
            span = _gpu_span_s(st)
            lines.append(
                f"  {label}: gpu span {span:.1f} s, busy {_busy(st)}, events {st['events']}, "
                f"file {st['bytes'] / 1e6:.0f} MB"
                + (f", parse error {st['error']}" if st.get("error") else "")
                if span
                else f"  {label}: no gpu events"
            )
            lines.append(f"  {label} runtime: {_launches(st)}")
            lines.append(f"  {label} categories: {_cats(st)}")
        fa, fb = _family_totals(ka), _family_totals(kb)
        lines.append(f"  {'family':10s} {'A_ms':>9} {'B_ms':>9} {'B-A_ms':>9}")
        for family in sorted(set(fa) | set(fb), key=lambda f: -abs(fb[f] - fa[f])):
            lines.append(
                f"  {family:10s} {fa[family] / 1e3:9.1f} {fb[family] / 1e3:9.1f} {(fb[family] - fa[family]) / 1e3:+9.1f}"
            )
        rows = []
        for name in set(ka) | set(kb):
            ua, na = ka.get(name, [0.0, 0])
            ub, nb = kb.get(name, [0.0, 0])
            rows.append((ub - ua, ua, ub, na, nb, name))
        rows.sort(key=lambda r: -abs(r[0]))
        lines.append(
            f"  {'B-A_ms':>8} {'A_ms':>8} {'B_ms':>8} {'A_n':>6} {'B_n':>6} {'A_avg':>7} {'B_avg':>7}  kernel"
        )
        for d, ua, ub, na, nb, name in rows[:top]:
            lines.append(
                f"  {d / 1e3:+8.2f} {ua / 1e3:8.2f} {ub / 1e3:8.2f} {na:6d} {nb:6d} "
                f"{(ua / na if na else 0):7.1f} {(ub / nb if nb else 0):7.1f}  {name[:96]}"
            )
        only_a = sorted((r for r in rows if r[4] == 0), key=lambda r: -r[1])[:8]
        only_b = sorted((r for r in rows if r[3] == 0), key=lambda r: -r[2])[:8]
        if only_a:
            lines.append("  kernels only in A (top by A_ms):")
            for _, ua, _, na, _, name in only_a:
                lines.append(f"    {ua / 1e3:8.2f} ms {na:6d}  {name[:100]}")
        if only_b:
            lines.append("  kernels only in B (top by B_ms):")
            for _, _, ub, _, nb, name in only_b:
                lines.append(f"    {ub / 1e3:8.2f} ms {nb:6d}  {name[:100]}")
        sa_short, sb_short = _short_totals(ka), _short_totals(kb)
        srows = []
        for name in set(sa_short) | set(sb_short):
            ua, na = sa_short.get(name, [0.0, 0])
            ub, nb = sb_short.get(name, [0.0, 0])
            srows.append((ub - ua, ua, ub, na, nb, name))
        srows.sort(key=lambda r: -abs(r[0]))
        lines.append(
            f"  by short name: {'B-A_ms':>8} {'A_ms':>8} {'B_ms':>8} {'A_n':>6} {'B_n':>6}  name"
        )
        for d, ua, ub, na, nb, name in srows[: max(15, top // 2)]:
            lines.append(
                f"                 {d / 1e3:+8.2f} {ua / 1e3:8.2f} {ub / 1e3:8.2f} {na:6d} {nb:6d}  {name[:90]}"
            )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stats")
    s.add_argument("trace_dir")
    s.add_argument("--workers", type=int, default=1)
    s = sub.add_parser("summary")
    s.add_argument("trace_dir")
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--workers", type=int, default=1)
    d = sub.add_parser("diff")
    d.add_argument("dir_a")
    d.add_argument("dir_b")
    d.add_argument("--top", type=int, default=40)
    d.add_argument("--workers", type=int, default=1)
    args = p.parse_args(argv)
    if args.cmd == "stats":
        ensure_stats(trace_files(Path(args.trace_dir)), args.workers)
    elif args.cmd == "summary":
        sys.stdout.write(
            format_summary(dir_stats(args.trace_dir, args.workers), args.top)
        )
    else:
        files = trace_files(Path(args.dir_a)) + trace_files(Path(args.dir_b))
        ensure_stats(files, args.workers)
        sys.stdout.write(
            format_diff(dir_stats(args.dir_a), dir_stats(args.dir_b), args.top)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
