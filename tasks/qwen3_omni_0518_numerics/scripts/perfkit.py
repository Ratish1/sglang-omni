"""Kernel census of decode steps from a torch profiler trace of one omni stage.

The trace is the chrome JSON written by /start_profile with enable_torch, one
file per stage process. Only device events and the scheduler thread's CUDA
runtime calls are needed, so the profiler's thread local op observer does not
matter: kernels launched by a CUDA graph replay carry the correlation id of the
cudaGraphLaunch call and the graph id, which is enough to attribute them.

    perfkit.py ingest TRACE.json[.gz] -o TRACE.pkl
    perfkit.py steps TRACE.pkl [--rows N] [--json OUT]
    perfkit.py census TRACE.pkl [--rows N] [--json OUT] [--top 40]
    perfkit.py diff A.json B.json

A step is the span from one backbone graph launch to the next on the scheduler
thread. Its phases in launch order are the backbone replay, the eager launches
before the predictor replay (layer 0 sampling and predictor prep), the
predictor replay, and the launches after it (token staging and collect).
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

CHUNK = 32 << 20

FAMILIES = (
    (
        "sampling",
        re.compile(r"sampl|argmax|multinomial|top_?k|top_?p|categorical", re.I),
    ),
    ("embedding", re.compile(r"gather_codec|embedding|index_select|indexSelect", re.I)),
    ("norm", re.compile(r"rms|norm", re.I)),
    ("rope", re.compile(r"rope|rotary", re.I)),
    ("activation", re.compile(r"silu|gelu|swiglu|act_and_mul", re.I)),
    ("attention", re.compile(r"fmha|flash|attn|sdpa|mha|scaled_dot", re.I)),
    (
        "gemm",
        re.compile(r"gemm|cutlass|nvjet|matmul|xmma|ampere_|wgmma|cublas|Sm90", re.I),
    ),
    ("reduce", re.compile(r"reduce", re.I)),
    (
        "elementwise",
        re.compile(
            r"elementwise|vectorized|unrolled|fill|cat_|copy|where|clamp|masked|index|scatter|arange",
            re.I,
        ),
    ),
)

GRAPH_LAUNCH = "cudaGraphLaunch"
D2H = "Memcpy DtoH"


def family(name: str) -> str:
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "other"


def short(name: str) -> str:
    name = name.replace("void ", "").replace("(anonymous namespace)::", "")
    cut = min((i for i in (name.find("<"), name.find("(")) if i > 0), default=len(name))
    return name[:cut][:90]


# ---------------------------------------------------------------- ingest


def iter_events(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    decoder = json.JSONDecoder()
    with opener(path, "rt") as handle:
        buf = handle.read(CHUNK)
        start = buf.find('"traceEvents"')
        if start < 0:
            raise SystemExit("no traceEvents array in the file head")
        pos = buf.index("[", start) + 1
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \n\r\t,":
                    pos += 1
                if pos < len(buf) and buf[pos] == "]":
                    return
                try:
                    obj, end = decoder.raw_decode(buf, pos)
                except ValueError:
                    break
                yield obj
                pos = end
            more = handle.read(CHUNK)
            if not more:
                return
            buf = buf[pos:] + more
            pos = 0


def ingest(path: str, out: str) -> None:
    kernels, runtime, memops = [], [], []
    names: dict[str, int] = {}
    for ev in iter_events(path):
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat")
        args = ev.get("args") or {}
        if cat == "kernel":
            nid = names.setdefault(ev["name"], len(names))
            kernels.append(
                (
                    ev["ts"],
                    ev["dur"],
                    nid,
                    args.get("correlation", -1),
                    args.get("stream", -1),
                    args.get("graph id", 0),
                )
            )
        elif cat in ("gpu_memcpy", "gpu_memset"):
            nid = names.setdefault(ev["name"], len(names))
            memops.append(
                (
                    ev["ts"],
                    ev["dur"],
                    nid,
                    args.get("correlation", -1),
                    args.get("bytes", 0),
                )
            )
        elif cat in ("cuda_runtime", "cuda_driver"):
            nid = names.setdefault(ev["name"], len(names))
            runtime.append(
                (ev["ts"], ev["dur"], nid, args.get("correlation", -1), ev.get("tid"))
            )
    kernels.sort()
    memops.sort()
    runtime.sort()
    with open(out, "wb") as handle:
        pickle.dump(
            {"names": names, "kernels": kernels, "runtime": runtime, "memops": memops},
            handle,
            protocol=5,
        )
    print(
        f"kernels={len(kernels)} runtime={len(runtime)} memops={len(memops)} names={len(names)} -> {out}"
    )


# ---------------------------------------------------------------- model


@dataclass
class Launch:
    ts: float
    dur: float
    name: str
    corr: int
    kernels: list = field(default_factory=list)  # (ts, dur, name)
    memops: list = field(default_factory=list)  # (ts, dur, name, bytes)
    graph: int = 0

    @property
    def busy(self) -> float:
        return sum(k[1] for k in self.kernels)

    @property
    def wall(self) -> float:
        if not self.kernels:
            return 0.0
        return self.kernels[-1][0] + self.kernels[-1][1] - self.kernels[0][0]


@dataclass
class Step:
    ts: float
    rows: int
    backbone: Launch
    predictor: Launch | None
    before: list  # launches between backbone and predictor
    after: list  # launches after the predictor until the next backbone
    next_ts: float

    def phase_busy(self, launches) -> float:
        return sum(l.busy + sum(m[1] for m in l.memops) for l in launches)


def load(pkl: str) -> dict:
    with open(pkl, "rb") as handle:
        d = pickle.load(handle)
    d["inv"] = {v: k for k, v in d["names"].items()}
    return d


def build_launches(d: dict, predictor_marker: re.Pattern):
    inv = d["inv"]
    by_corr_k = defaultdict(list)
    for ts, dur, nid, corr, stream, graph in d["kernels"]:
        by_corr_k[corr].append((ts, dur, inv[nid], graph))
    by_corr_m = defaultdict(list)
    for ts, dur, nid, corr, nbytes in d["memops"]:
        by_corr_m[corr].append((ts, dur, inv[nid], nbytes))
    # the scheduler thread launches the graphs
    tid_counts = Counter(
        tid for ts, dur, nid, corr, tid in d["runtime"] if inv[nid] == GRAPH_LAUNCH
    )
    if not tid_counts:
        raise SystemExit("no cudaGraphLaunch in the trace")
    sched_tid = tid_counts.most_common(1)[0][0]
    launches = []
    for ts, dur, nid, corr, tid in d["runtime"]:
        if tid != sched_tid:
            continue
        name = inv[nid]
        ks = by_corr_k.get(corr, [])
        ms = by_corr_m.get(corr, [])
        if not ks and not ms:
            continue
        launch = Launch(
            ts,
            dur,
            name,
            corr,
            [(k[0], k[1], k[2]) for k in ks],
            ms,
            ks[0][3] if ks else 0,
        )
        launches.append(launch)
    kinds = {}
    for launch in launches:
        if launch.name != GRAPH_LAUNCH:
            continue
        kinds[launch.corr] = (
            "predictor"
            if any(predictor_marker.search(k[2]) for k in launch.kernels)
            else "backbone"
        )
    return sched_tid, launches, kinds


def build_steps(launches, kinds) -> list[Step]:
    steps: list[Step] = []
    i = 0
    n = len(launches)
    while i < n:
        launch = launches[i]
        if kinds.get(launch.corr) != "backbone":
            i += 1
            continue
        j = i + 1
        before, after = [], []
        predictor = None
        while j < n and kinds.get(launches[j].corr) != "backbone":
            cur = launches[j]
            if kinds.get(cur.corr) == "predictor" and predictor is None:
                predictor = cur
            elif predictor is None:
                before.append(cur)
            else:
                after.append(cur)
            j += 1
        next_ts = launches[j].ts if j < n else launch.ts + 1e9
        # note: the first device to host copy after the predictor replay is the
        # int32 token staging copy, one element per row
        rows = 0
        for cur in after:
            for ts, dur, name, nbytes in cur.memops:
                if name.startswith(D2H) and nbytes:
                    rows = nbytes // 4
                    break
            if rows:
                break
        steps.append(Step(launch.ts, rows, launch, predictor, before, after, next_ts))
        i = j
    return steps


def p50(values):
    return statistics.median(values) if values else 0.0


def p90(values):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(round(0.9 * (len(values) - 1))))]


# ---------------------------------------------------------------- steps


def steps_report(
    d: dict, rows_filter: int | None, predictor_marker: re.Pattern, json_out: str | None
):
    sched_tid, launches, kinds = build_launches(d, predictor_marker)
    steps = build_steps(launches, kinds)
    decode = [
        s
        for s in steps
        if s.predictor is not None and (rows_filter is None or s.rows == rows_filter)
    ]
    print(
        f"scheduler tid {sched_tid}, launches {len(launches)}, steps {len(steps)}, decode steps with a predictor replay {len(decode)}"
    )
    groups = defaultdict(list)
    for s in decode:
        groups[s.rows].append(s)
    print()
    print(
        "| rows | steps | step wall p50 ms | backbone busy | sampling busy | predictor busy | predictor wall | predictor kernels | staging busy | idle in step | idle in predictor |"
    )
    print(
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    out = {}
    for rows in sorted(groups):
        ss = groups[rows]
        wall = [(s.next_ts - s.ts) / 1000 for s in ss]
        bb = [s.backbone.busy / 1000 for s in ss]
        smp = [s.phase_busy(s.before) / 1000 for s in ss]
        pb = [s.predictor.busy / 1000 for s in ss]
        pw = [s.predictor.wall / 1000 for s in ss]
        pk = [len(s.predictor.kernels) for s in ss]
        st = [s.phase_busy(s.after) / 1000 for s in ss]
        busy = [
            (
                s.backbone.busy
                + s.phase_busy(s.before)
                + s.predictor.busy
                + s.phase_busy(s.after)
            )
            / 1000
            for s in ss
        ]
        idle = [w - b for w, b in zip(wall, busy)]
        pidle = [w - b for w, b in zip(pw, pb)]
        out[rows] = {
            "steps": len(ss),
            "step_wall_ms": p50(wall),
            "backbone_busy_ms": p50(bb),
            "sampling_busy_ms": p50(smp),
            "predictor_busy_ms": p50(pb),
            "predictor_wall_ms": p50(pw),
            "predictor_kernels": p50(pk),
            "staging_busy_ms": p50(st),
            "idle_in_step_ms": p50(idle),
            "idle_in_predictor_ms": p50(pidle),
        }
        o = out[rows]
        print(
            f"| {rows} | {len(ss)} | {o['step_wall_ms']:.3f} | {o['backbone_busy_ms']:.3f} | {o['sampling_busy_ms']:.3f} | {o['predictor_busy_ms']:.3f} | {o['predictor_wall_ms']:.3f} | {o['predictor_kernels']:.0f} | {o['staging_busy_ms']:.3f} | {o['idle_in_step_ms']:.3f} | {o['idle_in_predictor_ms']:.3f} |"
        )
    if json_out:
        with open(json_out, "w") as handle:
            json.dump(out, handle, indent=2)
    return steps


# ---------------------------------------------------------------- census


def split_substeps(kernels, marker: re.Pattern):
    """Split one replay's kernel list at each sampler kernel (end of a sub-step)."""
    segments, cur = [], []
    for k in kernels:
        cur.append(k)
        if marker.search(k[2]):
            segments.append(cur)
            cur = []
    if cur:
        segments.append(cur)
    return segments


def census_report(
    d: dict,
    rows_filter: int | None,
    predictor_marker: re.Pattern,
    substep_marker: re.Pattern,
    top: int,
    json_out: str | None,
):
    sched_tid, launches, kinds = build_launches(d, predictor_marker)
    steps = build_steps(launches, kinds)
    decode = [s for s in steps if s.predictor is not None]
    groups = defaultdict(list)
    for s in decode:
        groups[s.rows].append(s)
    rows_list = [rows_filter] if rows_filter is not None else sorted(groups)
    result = {}
    for rows in rows_list:
        ss = groups.get(rows, [])
        if not ss:
            continue
        counts = Counter(len(s.predictor.kernels) for s in ss)
        n_common = counts.most_common(1)[0][0]
        replays = [s.predictor for s in ss if len(s.predictor.kernels) == n_common]
        print(
            f"\n## rows {rows}: {len(ss)} replays, {n_common} kernels per replay in {len(replays)} of them"
        )
        busy = [r.busy / 1000 for r in replays]
        wall = [r.wall / 1000 for r in replays]
        gaps = []
        for r in replays:
            ks = r.kernels
            gaps.extend(
                ks[i + 1][0] - (ks[i][0] + ks[i][1]) for i in range(len(ks) - 1)
            )
        durs = [k[1] for r in replays for k in r.kernels]
        print(
            f"replay busy p50 {p50(busy):.3f} ms, wall p50 {p50(wall):.3f} ms, kernel dur p50 {p50(durs):.2f} us p90 {p90(durs):.2f} us, gap between kernels p50 {p50(gaps):.2f} us p90 {p90(gaps):.2f} us"
        )
        # per family over the whole replay, medians over replays
        fam_time = defaultdict(list)
        fam_count = defaultdict(list)
        for r in replays:
            ft, fc = defaultdict(float), Counter()
            for ts, dur, name in r.kernels:
                ft[family(name)] += dur
                fc[family(name)] += 1
            for f in set(ft) | set(fc):
                fam_time[f].append(ft[f] / 1000)
                fam_count[f].append(fc[f])
        print("\n| family | kernels per replay | busy p50 ms | share |")
        print("| --- | ---: | ---: | ---: |")
        total = p50(busy) or 1.0
        fam_rows = {}
        for f in sorted(fam_time, key=lambda x: -p50(fam_time[x])):
            fam_rows[f] = {"kernels": p50(fam_count[f]), "busy_ms": p50(fam_time[f])}
            print(
                f"| {f} | {p50(fam_count[f]):.0f} | {p50(fam_time[f]):.3f} | {100 * p50(fam_time[f]) / total:.1f}% |"
            )
        # sub-steps
        seg_counts = Counter(
            len(split_substeps(r.kernels, substep_marker)) for r in replays
        )
        n_sub = seg_counts.most_common(1)[0][0]
        print(
            f"\nsub-steps per replay: {n_sub} (by the sampler kernel), kernels per sub-step:",
            end=" ",
        )
        per_sub = defaultdict(list)
        for r in replays:
            segs = split_substeps(r.kernels, substep_marker)
            if len(segs) != n_sub:
                continue
            for i, seg in enumerate(segs):
                per_sub[i].append(seg)
        print([int(p50([len(s) for s in per_sub[i]])) for i in range(n_sub)])
        # anatomy of one middle sub-step: kernel sequence with median dur
        mid = n_sub // 2
        seqs = per_sub[mid]
        seq_len = Counter(len(s) for s in seqs).most_common(1)[0][0]
        seqs = [s for s in seqs if len(s) == seq_len]
        print(
            f"\nsub-step {mid} anatomy ({seq_len} kernels, busy p50 {p50([sum(k[1] for k in s) for s in seqs]) / 1000:.3f} ms):"
        )
        print("| # | family | kernel | dur p50 us |")
        print("| ---: | --- | --- | ---: |")
        anatomy = []
        for i in range(seq_len):
            name = seqs[0][i][2]
            dur = p50([s[i][1] for s in seqs])
            anatomy.append(
                {"family": family(name), "kernel": short(name), "dur_us": dur}
            )
            print(f"| {i} | {family(name)} | {short(name)} | {dur:.2f} |")
        # top kernels by time over the replay
        by_name_t, by_name_c = defaultdict(list), defaultdict(list)
        for r in replays:
            t, c = defaultdict(float), Counter()
            for ts, dur, name in r.kernels:
                t[name] += dur
                c[name] += 1
            for name in t:
                by_name_t[name].append(t[name])
                by_name_c[name].append(c[name])
        print(f"\ntop {top} kernels per replay by time:")
        print("| kernel | family | count | busy p50 us |")
        print("| --- | --- | ---: | ---: |")
        top_rows = []
        for name in sorted(by_name_t, key=lambda x: -p50(by_name_t[x]))[:top]:
            top_rows.append(
                {
                    "kernel": short(name),
                    "family": family(name),
                    "count": p50(by_name_c[name]),
                    "busy_us": p50(by_name_t[name]),
                }
            )
            print(
                f"| {short(name)} | {family(name)} | {p50(by_name_c[name]):.0f} | {p50(by_name_t[name]):.1f} |"
            )
        result[rows] = {
            "replays": len(replays),
            "kernels_per_replay": n_common,
            "busy_ms": p50(busy),
            "wall_ms": p50(wall),
            "kernel_dur_p50_us": p50(durs),
            "gap_p50_us": p50(gaps),
            "gap_p90_us": p90(gaps),
            "families": fam_rows,
            "substeps": n_sub,
            "substep_anatomy": anatomy,
            "top": top_rows,
        }
    if json_out:
        with open(json_out, "w") as handle:
            json.dump(result, handle, indent=2)


def timeline_report(
    d: dict, rows_filter: int, predictor_marker: re.Pattern, index: int | None
):
    """Host launches on the scheduler thread against the device spans of one
    step, so the gaps where the GPU waits for the host are visible next to
    what the host was issuing. The step is the median wall step of the row
    count unless an index is given."""
    sched_tid, launches, kinds = build_launches(d, predictor_marker)
    steps = build_steps(launches, kinds)
    group = [s for s in steps if s.predictor is not None and s.rows == rows_filter]
    if not group:
        raise SystemExit(f"no decode step with rows {rows_filter}")
    if index is None:
        walls = sorted(group, key=lambda s: s.next_ts - s.ts)
        step = walls[len(walls) // 2]
    else:
        step = group[index]
    t0 = step.ts
    phases = [
        ("backbone", [step.backbone]),
        ("before predictor", step.before),
        ("predictor", [step.predictor]),
        ("after predictor", step.after),
    ]
    print(
        f"rows {rows_filter}, step at {t0:.0f} us, wall {(step.next_ts - t0) / 1000:.3f} ms to the next backbone launch\n"
    )
    print(
        "| phase | host call | host t0 ms | host dur us | device t0 ms | device end ms | kernels | device busy us |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    gpu_end = t0
    for label, group_launches in phases:
        for l in group_launches:
            spans = [(k[0], k[0] + k[1]) for k in l.kernels] + [
                (m[0], m[0] + m[1]) for m in l.memops
            ]
            if not spans:
                print(
                    f"| {label} | {l.name} | {(l.ts - t0) / 1000:.3f} | {l.dur:.1f} | | | 0 | 0 |"
                )
                continue
            d0, d1 = min(x[0] for x in spans), max(x[1] for x in spans)
            busy = sum(x[1] - x[0] for x in spans)
            wait = d0 - gpu_end
            gpu_end = max(gpu_end, d1)
            note = f" (device idle {wait / 1000:.3f} ms before)" if wait > 5 else ""
            print(
                f"| {label} | {l.name}{note} | {(l.ts - t0) / 1000:.3f} | {l.dur:.1f} | {(d0 - t0) / 1000:.3f} | {(d1 - t0) / 1000:.3f} | {len(l.kernels)} | {busy:.1f} |"
            )
    print(
        f"\ndevice end of the step {(gpu_end - t0) / 1000:.3f} ms, next backbone launch at {(step.next_ts - t0) / 1000:.3f} ms, host only tail {(step.next_ts - gpu_end) / 1000:.3f} ms"
    )


def diff_report(a_path: str, b_path: str):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    for rows in sorted(set(a) | set(b), key=int):
        if rows not in a or rows not in b:
            continue
        ra, rb = a[rows], b[rows]
        print(
            f"\n## rows {rows}: kernels per replay {ra['kernels_per_replay']} -> {rb['kernels_per_replay']}, busy {ra['busy_ms']:.3f} -> {rb['busy_ms']:.3f} ms ({rb['busy_ms'] - ra['busy_ms']:+.3f}), wall {ra['wall_ms']:.3f} -> {rb['wall_ms']:.3f} ms"
        )
        print("| family | kernels A | kernels B | busy A ms | busy B ms | delta ms |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for f in sorted(set(ra["families"]) | set(rb["families"])):
            fa = ra["families"].get(f, {"kernels": 0, "busy_ms": 0.0})
            fb = rb["families"].get(f, {"kernels": 0, "busy_ms": 0.0})
            print(
                f"| {f} | {fa['kernels']:.0f} | {fb['kernels']:.0f} | {fa['busy_ms']:.3f} | {fb['busy_ms']:.3f} | {fb['busy_ms'] - fa['busy_ms']:+.3f} |"
            )


# ---------------------------------------------------------------- cli


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("ingest")
    s.add_argument("trace")
    s.add_argument("-o", "--out", required=True)
    for name in ("steps", "census"):
        s = sub.add_parser(name)
        s.add_argument("pkl")
        s.add_argument("--rows", type=int, default=None)
        s.add_argument("--json", default=None)
        s.add_argument(
            "--predictor-marker",
            default=r"gather_codec_embedding|seeded_top_k_top_p|seeded_gumbel",
            help="regex on a kernel name that only the predictor graph contains",
        )
        if name == "census":
            s.add_argument(
                "--substep-marker",
                default=r"seeded_top_k_top_p_sample|seeded_gumbel_sample",
                help="regex on the kernel that ends each predictor sub-step",
            )
            s.add_argument("--top", type=int, default=40)
    s = sub.add_parser("timeline")
    s.add_argument("pkl")
    s.add_argument("--rows", type=int, required=True)
    s.add_argument("--index", type=int, default=None)
    s.add_argument(
        "--predictor-marker",
        default=r"gather_codec_embedding|seeded_top_k_top_p|seeded_gumbel",
    )
    s = sub.add_parser("diff")
    s.add_argument("a")
    s.add_argument("b")
    args = parser.parse_args()
    if args.cmd == "ingest":
        ingest(args.trace, args.out)
    elif args.cmd == "steps":
        steps_report(
            load(args.pkl),
            args.rows,
            re.compile(args.predictor_marker, re.I),
            args.json,
        )
    elif args.cmd == "census":
        census_report(
            load(args.pkl),
            args.rows,
            re.compile(args.predictor_marker, re.I),
            re.compile(args.substep_marker, re.I),
            args.top,
            args.json,
        )
    elif args.cmd == "timeline":
        timeline_report(
            load(args.pkl),
            args.rows,
            re.compile(args.predictor_marker, re.I),
            args.index,
        )
    else:
        diff_report(args.a, args.b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
