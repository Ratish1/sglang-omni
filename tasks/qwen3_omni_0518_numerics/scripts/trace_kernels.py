# SPDX-License-Identifier: Apache-2.0
"""Summarise CUDA kernel time per stage from omni torch profiler traces.

Usage:
    python trace_kernels.py summary TRACE_DIR [--top 25]
    python trace_kernels.py diff TRACE_DIR_A TRACE_DIR_B [--top 40]

TRACE_DIR holds the files a /start_profile run with enable_torch true wrote
through the template TRACE_DIR/{stage}, one <stage>_pid<pid>_rank<r>.trace.json
(or .gz) per stage process. summary prints, per stage, the total CUDA kernel
time, the time per kernel family (attention, moe, gemm, norm, rope, sampling,
copy, other) and the top kernels by total time. diff aligns two runs by stage
and kernel name and prints the kernels whose total time moved the most, which
is the per-component attribution of a speed change between two stacks when
both runs served the same requests.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
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


def _family(name: str) -> str:
    for family, pattern in FAMILIES:
        if pattern.search(name):
            return family
    return "other"


def _load(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as fp:
            return json.load(fp)
    return json.load(open(path, encoding="utf-8"))


def _stage_of(path: Path) -> str:
    name = path.name
    return re.sub(r"_pid\d+.*$", "", name)


def kernels_by_stage(trace_dir: str) -> dict[str, dict[str, list[float]]]:
    """stage -> kernel name -> [total_us, count]."""
    out: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: [0.0, 0])
    )
    files = sorted(p for p in Path(trace_dir).iterdir() if ".trace.json" in p.name)
    if not files:
        raise SystemExit(f"no *.trace.json files in {trace_dir}")
    for path in files:
        stage = _stage_of(path)
        events = _load(path).get("traceEvents", [])
        for ev in events:
            if ev.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            dur = float(ev.get("dur", 0.0))
            entry = out[stage][ev.get("name", "?")]
            entry[0] += dur
            entry[1] += 1
    return out


def summary(trace_dir: str, top: int) -> None:
    data = kernels_by_stage(trace_dir)
    for stage in sorted(data):
        kernels = data[stage]
        total = sum(v[0] for v in kernels.values())
        fam = collections.Counter()
        for name, (us, _) in kernels.items():
            fam[_family(name)] += us
        print(
            f"\n== {stage}: CUDA kernel time {total / 1e3:.1f} ms in {sum(v[1] for v in kernels.values())} launches"
        )
        print(
            "  by family: "
            + ", ".join(
                f"{k} {v / 1e3:.1f} ms ({100 * v / total:.1f}%)"
                for k, v in fam.most_common()
            )
        )
        print(f"  {'total_ms':>9} {'count':>7} {'avg_us':>8}  kernel")
        for name, (us, n) in sorted(kernels.items(), key=lambda kv: -kv[1][0])[:top]:
            print(f"  {us / 1e3:9.2f} {n:7d} {us / n:8.1f}  {name[:110]}")


def diff(dir_a: str, dir_b: str, top: int) -> None:
    a = kernels_by_stage(dir_a)
    b = kernels_by_stage(dir_b)
    for stage in sorted(set(a) | set(b)):
        ka = a.get(stage, {})
        kb = b.get(stage, {})
        ta = sum(v[0] for v in ka.values())
        tb = sum(v[0] for v in kb.values())
        fa = collections.Counter()
        fb = collections.Counter()
        for name, (us, _) in ka.items():
            fa[_family(name)] += us
        for name, (us, _) in kb.items():
            fb[_family(name)] += us
        print(
            f"\n== {stage}: A {ta / 1e3:.1f} ms, B {tb / 1e3:.1f} ms, B-A {(tb - ta) / 1e3:+.1f} ms ({100 * (tb - ta) / ta if ta else 0:+.1f}%)"
        )
        for family in sorted(set(fa) | set(fb), key=lambda f: -(abs(fb[f] - fa[f]))):
            print(
                f"  {family:10s} A {fa[family] / 1e3:8.1f} ms  B {fb[family] / 1e3:8.1f} ms  B-A {(fb[family] - fa[family]) / 1e3:+8.1f} ms"
            )
        rows = []
        for name in set(ka) | set(kb):
            ua, na = ka.get(name, [0.0, 0])
            ub, nb = kb.get(name, [0.0, 0])
            rows.append((ub - ua, ua, ub, na, nb, name))
        rows.sort(key=lambda r: -abs(r[0]))
        print(f"  {'B-A_ms':>8} {'A_ms':>8} {'B_ms':>8} {'A_n':>6} {'B_n':>6}  kernel")
        for d, ua, ub, na, nb, name in rows[:top]:
            print(
                f"  {d / 1e3:+8.2f} {ua / 1e3:8.2f} {ub / 1e3:8.2f} {na:6d} {nb:6d}  {name[:100]}"
            )


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("trace_dir")
    s.add_argument("--top", type=int, default=25)
    d = sub.add_parser("diff")
    d.add_argument("dir_a")
    d.add_argument("dir_b")
    d.add_argument("--top", type=int, default=40)
    args = p.parse_args(argv)
    if args.cmd == "summary":
        summary(args.trace_dir, args.top)
    else:
        diff(args.dir_a, args.dir_b, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
