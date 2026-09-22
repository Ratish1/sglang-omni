"""Kernel level census of the Qwen3-TTS first chunk path from an nsys sqlite export.

Needs the ttfc_nvtx probe on the profiled server. Every CUDA runtime call is attributed
to the innermost NVTX range covering it on its own thread; its kernel, copy or memset
(by correlation id, graph replays included) goes with it, and to every enclosing range.
Per range kind: count, host wall, kernel busy (union), device span, bubbles inside the
span, launches and syncs per instance, copies, then the kernel name census, the GPU
metric means over the kind's kernels (when the profile carries metrics), the drift of
the first instances against the rest, and the per request critical path joined on the
request id: prepare, queue, prefill to the first codes, bootstrap to the first chunk.

usage: python ttfc_census.py REPORT.sqlite [--bench-log bench.log] [--top 20]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics
from dataclasses import dataclass, field

from nsys_metrics import METRICS, bench_window

LAUNCH_NAMES = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
)
GRAPH_LAUNCH = "cudaGraphLaunch"
SYNC_NAMES = ("cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize")
COPY_KINDS = {1: "h2d", 2: "d2h", 8: "d2d"}
DRIFT_KINDS = (
    "pre.speaker",
    "pre.spk_encoder",
    "pre.mel",
    "sched.batch",
    "voc.initial",
)
NVTX_PUSH_POP = 59
NVTX_MARK = 34


@dataclass
class Range:
    kind: str
    label: str
    start: int
    end: int
    tid: int
    parent: Range | None = None
    apis: int = 0
    launches: int = 0
    graph_launches: int = 0
    syncs: int = 0
    sync_ns: int = 0
    copies: collections.Counter = field(default_factory=collections.Counter)
    copy_bytes: collections.Counter = field(default_factory=collections.Counter)
    memsets: int = 0
    kernels: list[tuple[int, int, str]] = field(default_factory=list)

    @property
    def wall_ns(self) -> int:
        return self.end - self.start

    def rid(self) -> str | None:
        match = re.search(r"rid=(\S+)", self.label)
        return match.group(1) if match else None


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total, cursor = 0, None
    for start, end in sorted(intervals):
        if cursor is None or start > cursor[1]:
            if cursor is not None:
                total += cursor[1] - cursor[0]
            cursor = [start, end]
        else:
            cursor[1] = max(cursor[1], end)
    if cursor is not None:
        total += cursor[1] - cursor[0]
    return total


def load_ranges(db, strings, t0, t1):
    """NVTX push/pop ranges in the window, with parents, per thread."""
    by_thread: dict[int, list[Range]] = collections.defaultdict(list)
    rows = db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS "
        "where eventType = ? and end is not null and end > ? and start < ? order by start",
        (NVTX_PUSH_POP, t0, t1),
    )
    for start, end, tid, text, text_id in rows:
        label = text if text is not None else strings.get(text_id, str(text_id))
        words = label.split(" ")
        # note(ratish): scheduler batches split by mode; the rest by the first word
        kind = " ".join(words[:2]) if words[0].startswith("sched.") else words[0]
        by_thread[tid].append(Range(kind, label, start, end, tid))
    for ranges in by_thread.values():
        stack: list[Range] = []
        for item in ranges:
            while stack and stack[-1].end < item.start:
                stack.pop()
            item.parent = stack[-1] if stack else None
            stack.append(item)
    return by_thread


def innermost(ranges: list[Range], starts: list[int], t: int) -> Range | None:
    index = bisect.bisect_right(starts, t) - 1
    if index < 0:
        return None
    item = ranges[index]
    while item is not None and item.end < t:
        item = item.parent
    return item


def attribute(db, strings, by_thread, t0, t1):
    starts = {tid: [r.start for r in ranges] for tid, ranges in by_thread.items()}
    api_rows = db.execute(
        "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME "
        "where end > ? and start < ?",
        (t0, t1),
    ).fetchall()
    owner: dict[int, Range] = {}
    for start, end, tid, corr, name_id in api_rows:
        ranges = by_thread.get(tid)
        if not ranges:
            continue
        item = innermost(ranges, starts[tid], start)
        if item is None:
            continue
        owner[corr] = item
        name = strings.get(name_id, "")
        is_launch = name.startswith(LAUNCH_NAMES)
        is_graph = name.startswith(GRAPH_LAUNCH)
        is_sync = name.startswith(SYNC_NAMES)
        while item is not None:
            item.apis += 1
            item.launches += is_launch
            item.graph_launches += is_graph
            if is_sync:
                item.syncs += 1
                item.sync_ns += end - start
            item = item.parent
    for start, end, corr, name_id in db.execute(
        "select start, end, correlationId, demangledName from CUPTI_ACTIVITY_KIND_KERNEL "
        "where end > ? and start < ?",
        (t0, t1),
    ):
        item = owner.get(corr)
        name = strings.get(name_id, str(name_id))
        while item is not None:
            item.kernels.append((start, end, name))
            item = item.parent
    for start, end, corr, nbytes, kind in db.execute(
        "select start, end, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY "
        "where end > ? and start < ?",
        (t0, t1),
    ):
        item = owner.get(corr)
        label = COPY_KINDS.get(kind, str(kind))
        while item is not None:
            item.copies[label] += 1
            item.copy_bytes[label] += nbytes
            item = item.parent
    for (corr,) in db.execute(
        "select correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end > ? and start < ?",
        (t0, t1),
    ):
        item = owner.get(corr)
        while item is not None:
            item.memsets += 1
            item = item.parent
    return owner


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def ms(ns):
    return ns / 1e6


def kind_table(kinds: dict[str, list[Range]]):
    print(
        f"{'range kind':<22}{'n':>6}{'wall ms':>9}{'p50':>8}{'p95':>8}{'kern ms':>9}"
        f"{'span ms':>9}{'bubble ms':>10}{'kernels':>9}{'launch':>8}{'graphs':>8}"
        f"{'syncs':>7}{'sync ms':>9}{'copies':>8}{'h2d':>6}{'d2h':>6}"
    )
    print(
        "  (per instance means; kern = union of the kernels launched inside the range;"
    )
    print("   span = first kernel start to last kernel end; bubble = span minus kern)")
    for kind, ranges in sorted(kinds.items()):
        walls = [r.wall_ns for r in ranges]
        busy = [union_ns([(s, e) for s, e, _ in r.kernels]) for r in ranges]
        spans = [
            (
                (max(e for _, e, _ in r.kernels) - min(s for s, _, _ in r.kernels))
                if r.kernels
                else 0
            )
            for r in ranges
        ]
        n = len(ranges)
        copies = sum(sum(r.copies.values()) for r in ranges) / n
        print(
            f"{kind:<22}{n:>6}{ms(statistics.fmean(walls)):>9.2f}{ms(pct(walls, 0.5)):>8.2f}"
            f"{ms(pct(walls, 0.95)):>8.2f}{ms(statistics.fmean(busy)):>9.2f}"
            f"{ms(statistics.fmean(spans)):>9.2f}"
            f"{ms(statistics.fmean(spans) - statistics.fmean(busy)):>10.2f}"
            f"{statistics.fmean(len(r.kernels) for r in ranges):>9.1f}"
            f"{statistics.fmean(r.launches for r in ranges):>8.1f}"
            f"{statistics.fmean(r.graph_launches for r in ranges):>8.1f}"
            f"{statistics.fmean(r.syncs for r in ranges):>7.1f}"
            f"{ms(statistics.fmean(r.sync_ns for r in ranges)):>9.2f}"
            f"{copies:>8.1f}"
            f"{statistics.fmean(r.copies['h2d'] for r in ranges):>6.1f}"
            f"{statistics.fmean(r.copies['d2h'] for r in ranges):>6.1f}"
        )


def kernel_census(kinds: dict[str, list[Range]], top: int, only: tuple[str, ...]):
    for kind in only:
        ranges = kinds.get(kind)
        if not ranges:
            continue
        totals: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
        for r in ranges:
            for s, e, name in r.kernels:
                totals[name][0] += 1
                totals[name][1] += e - s
        n = len(ranges)
        all_ns = sum(v[1] for v in totals.values())
        print(
            f"\n{kind}: {len(totals)} distinct kernels, {sum(v[0] for v in totals.values()) / n:.1f} per instance, {ms(all_ns / n):.2f} ms device per instance"
        )
        print(
            f"  {'kernel':<96}{'per inst':>9}{'us each':>9}{'ms/inst':>9}{'share':>7}"
        )
        for name, (count, total) in sorted(totals.items(), key=lambda kv: -kv[1][1])[
            :top
        ]:
            print(
                f"  {name[:96]:<96}{count / n:>9.1f}{total / count / 1e3:>9.1f}"
                f"{ms(total / n):>9.3f}{100 * total / all_ns:>6.1f}%"
            )


def metric_means(db, kinds: dict[str, list[Range]], t0, t1):
    ids = {}
    try:
        rows = db.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS")
    except sqlite3.OperationalError:
        return
    for metric_id, name in rows:
        for metric in METRICS[:3]:
            if str(name).startswith(metric):
                ids[metric] = metric_id
    if not ids:
        return
    samples = {
        metric: db.execute(
            "select timestamp, value from GPU_METRICS where metricId = ? and timestamp between ? and ? order by timestamp",
            (metric_id, t0, t1),
        ).fetchall()
        for metric, metric_id in ids.items()
    }
    print(
        "\nGPU metric means over each kind's own kernels (samples inside its kernels)"
    )
    print(
        f"  {'range kind':<22}"
        + "".join(f"{m:>16}" for m in samples)
        + f"{'samples':>9}"
    )
    for kind, ranges in sorted(kinds.items()):
        intervals = sorted((s, e) for r in ranges for s, e, _ in r.kernels)
        if not intervals:
            continue
        starts = [s for s, _ in intervals]
        cells, count = [], 0
        for metric, rows in samples.items():
            values = []
            for ts, value in rows:
                index = bisect.bisect_right(starts, ts) - 1
                if index >= 0 and intervals[index][1] >= ts:
                    values.append(value)
            count = len(values)
            cells.append(
                f"{statistics.fmean(values):>15.1f}%" if values else f"{'-':>16}"
            )
        print(f"  {kind:<22}" + "".join(cells) + f"{count:>9}")


def drift(kinds: dict[str, list[Range]]):
    print(
        "\nfirst instances against the rest (wall ms | kernel union ms), startup effects"
    )
    for kind in DRIFT_KINDS:
        ranges = sorted(kinds.get(kind, []), key=lambda r: r.start)
        if len(ranges) < 8:
            continue
        head = ranges[:6]
        rest = ranges[6:]
        cells = " ".join(
            f"{ms(r.wall_ns):.1f}|{ms(union_ns([(s, e) for s, e, _ in r.kernels])):.1f}"
            for r in head
        )
        rest_wall = ms(statistics.median(r.wall_ns for r in rest))
        rest_busy = ms(
            statistics.median(union_ns([(s, e) for s, e, _ in r.kernels]) for r in rest)
        )
        print(
            f"  {kind:<16} first 6: {cells}   rest median: {rest_wall:.1f}|{rest_busy:.1f}"
        )


def critical_path(by_thread, db, strings, t0, t1):
    prepare, build, chunk, commit = {}, {}, {}, {}
    for ranges in by_thread.values():
        for r in ranges:
            rid = r.rid()
            if rid is None:
                continue
            if r.kind == "pre.prepare":
                prepare.setdefault(rid, r)
            elif r.kind == "sched.build":
                build.setdefault(rid, r)
            elif r.kind == "voc.chunk":
                chunk.setdefault(rid, r)
            elif r.kind == "voc.commit":
                commit.setdefault(rid, r)
    prefill, admit = {}, {}
    for start, text, text_id in db.execute(
        "select start, text, textId from NVTX_EVENTS where eventType = ? and start between ? and ?",
        (NVTX_MARK, t0, t1),
    ):
        label = text if text is not None else strings.get(text_id, "")
        match = re.match(r"sched\.(prefill|admit) rid=(\S+)", label)
        if match and match.group(1) == "prefill":
            prefill.setdefault(match.group(2), start)
        elif match:
            admit.setdefault(match.group(2), start)
    joined = [
        rid for rid in prepare if rid in prefill and rid in chunk and rid in commit
    ]
    print(
        f"\nper request critical path: {len(joined)} requests joined "
        f"(prepare {len(prepare)}, builds {len(build)}, admits {len(admit)}, prefill marks {len(prefill)}, "
        f"first chunk {len(chunk)}, commits {len(commit)})"
    )
    if not joined:
        return
    segments = {
        "prepare (preprocessing)": [prepare[r].wall_ns for r in joined],
        "prepare end -> prefill launch": [prefill[r] - prepare[r].end for r in joined],
    }
    split = [r for r in joined if r in build and r in admit]
    if split:
        segments.update(
            {
                "  prepare end -> build start": [
                    build[r].start - prepare[r].end for r in split
                ],
                "  build (scheduler side)": [build[r].wall_ns for r in split],
                "  build end -> admitted": [admit[r] - build[r].end for r in split],
                "  admitted -> prefill launch": [prefill[r] - admit[r] for r in split],
            }
        )
    segments.update(
        {
            "prefill launch -> first codes at vocoder": [
                chunk[r].start - prefill[r] for r in joined
            ],
            "first codes -> initial decode start": [
                (
                    commit[r].parent.start - chunk[r].start
                    if commit[r].parent is not None
                    else 0
                )
                for r in joined
            ],
            "initial decode start -> first chunk committed": [
                commit[r].end
                - (commit[r].parent.start if commit[r].parent else commit[r].start)
                for r in joined
            ],
            "prepare start -> first chunk committed": [
                commit[r].end - prepare[r].start for r in joined
            ],
        }
    )
    print(f"  {'segment':<46}{'mean ms':>9}{'p50':>8}{'p95':>8}")
    for name, values in segments.items():
        print(
            f"  {name:<46}{ms(statistics.fmean(values)):>9.2f}{ms(pct(values, 0.5)):>8.2f}"
            f"{ms(pct(values, 0.95)):>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    if args.bench_log:
        t0, t1 = bench_window(db, args.bench_log)
    else:
        t0, t1 = db.execute(
            "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
    by_thread = load_ranges(db, strings, t0, t1)
    attribute(db, strings, by_thread, t0, t1)
    kinds: dict[str, list[Range]] = collections.defaultdict(list)
    for ranges in by_thread.values():
        for r in ranges:
            if r.start >= t0 and r.end <= t1:
                kinds[r.kind].append(r)
    print(
        f"window {t0 / 1e9:.3f} .. {t1 / 1e9:.3f} s ({(t1 - t0) / 1e9:.1f} s), "
        f"{sum(len(v) for v in kinds.values())} ranges of {len(kinds)} kinds on {len(by_thread)} threads"
    )
    kind_table(kinds)
    kernel_census(
        kinds,
        args.top,
        (
            "pre.spk_encoder",
            "pre.mel",
            "pre.speaker",
            "ref.encode",
            "sched.batch extend",
            "sched.batch decode",
            "voc.cohort",
            "voc.replay",
            "voc.initial",
        ),
    )
    metric_means(db, kinds, t0, t1)
    drift(kinds)
    critical_path(by_thread, db, strings, t0, t1)


if __name__ == "__main__":
    main()
