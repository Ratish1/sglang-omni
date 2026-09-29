"""Whole pipeline census of a Qwen3-TTS serve from one nsys sqlite export.

Needs the pipeline_nvtx probe on the profiled server. Every CUDA runtime call belongs to
the innermost probe range on its thread; its kernels, graph replays, copies and memsets
(by correlation id) belong to that range and to every range enclosing it. Graph node
kernels (node trace) resolve further, through the node creation events, to the capture
time op range that created them: the line that captured the kernel. Host time inside a
range splits into CUDA API calls, GIL waits (python-gil trace), OS runtime blocking and
the rest. Hops pair the probe's queue marks per request in order.

Sections:
  A gates          unknown kernels, graph kernel label coverage, mark pairing counts
  B components     per range kind: wall, host split, device busy, kernels, launches,
                   graphs, syncs, copies
  C hops           per hop: latency and what the consuming thread was doing meanwhile
  D device         busy, idle, per owner exclusive time, pairwise overlap, per kernel
                   slowdown when another owner runs beside it
  E GIL            hold and wait per thread, waiter by holder
  F first chunk    per request critical path
  G attribution    graph and eager kernels by capture site and op call site
  H playback       per chunk margin at the coordinator: audio delivered against time

usage: python pipeline_census.py REPORT.sqlite --bench-log bench.log [--top 25]
       [--sections ABCDEFG]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics
from dataclasses import dataclass, field

from nsys_metrics import bench_window

PUSH_POP = 59
MARK = 34
LAUNCH = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
)
GRAPH_LAUNCH = "cudaGraphLaunch"
SYNC = ("cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize")
COPY_KINDS = {1: "h2d", 2: "d2h", 8: "d2d"}
CAPTURE_KINDS = ("cap", "op", "compiled")
OWNER_RULES = (
    ("mr.predictor", "predictor"),
    ("mr.talker", "talker"),
    ("mr.sample", "sampler"),
    ("sched.build", "build"),
    ("sched.adopt", "build"),
    ("sched", "engine.other"),
    ("mr", "engine.other"),
    ("pre.spk_embed", "speaker"),
    ("pre", "pre.other"),
    ("ref", "ref.codes"),
    ("voc.initial", "voc.initial"),
    ("voc.followup", "voc.followup"),
    ("voc.drain", "voc.followup"),
    ("voc.finish", "voc.followup"),
    ("voc", "voc.other"),
)


@dataclass
class Range:
    kind: str
    label: str
    start: int
    end: int
    tid: int
    parent: "Range | None" = None
    launches: int = 0
    graph_launches: int = 0
    syncs: int = 0
    sync_ns: int = 0
    api_ns: int = 0
    copies: collections.Counter = field(default_factory=collections.Counter)
    kernels: list = field(default_factory=list)

    @property
    def wall(self) -> int:
        return self.end - self.start

    def rid(self) -> str | None:
        match = re.search(r"rid=(\S+)", self.label)
        return match.group(1) if match else None

    def chain(self) -> list[str]:
        kinds, item = [], self
        while item is not None:
            kinds.append(item.kind)
            item = item.parent
        return kinds


def kind_of(label: str) -> str:
    words = label.split(" ")
    if words[0] in ("sched.batch", "sched.result", "voc.collect", "voc.replay"):
        return " ".join(words[:2])
    return words[0]


def owner_of(item: Range | None) -> str:
    if item is None:
        return "unknown"
    kinds = item.chain()
    for prefix, owner in OWNER_RULES:
        for kind in kinds:
            if (
                kind == prefix
                or kind.startswith(prefix + ".")
                or kind.startswith(prefix + " ")
            ):
                return owner
    return "other:" + kinds[-1]


def union_ns(intervals) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def merged(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def clip(sorted_intervals, starts, s, e):
    """Intervals of one thread's non overlapping list that fall inside [s, e], clipped."""
    out = []
    i = max(bisect.bisect_right(starts, s) - 1, 0)
    while i < len(sorted_intervals):
        a, b = sorted_intervals[i][0], sorted_intervals[i][1]
        if a >= e:
            break
        if b > s:
            out.append([max(a, s), min(b, e)])
        i += 1
    return out


def minus_len(intervals, cover) -> int:
    """Length of intervals not covered by cover (both merged and sorted)."""
    total, j = 0, 0
    for s, e in intervals:
        length = e - s
        while j < len(cover) and cover[j][1] <= s:
            j += 1
        k = j
        while k < len(cover) and cover[k][0] < e:
            length -= max(0, min(e, cover[k][1]) - max(s, cover[k][0]))
            k += 1
        total += length
    return total


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else 0


def ms(ns) -> float:
    return ns / 1e6


class Report:
    def __init__(self, path: str, bench_log: str | None):
        self.db = sqlite3.connect(path)
        self.strings = dict(self.db.execute("select id, value from StringIds"))
        self.tables = {
            r[0]
            for r in self.db.execute(
                "select name from sqlite_master where type='table'"
            )
        }
        if bench_log:
            self.t0, self.t1 = bench_window(self.db, bench_log)
        else:
            self.t0, self.t1 = self.db.execute(
                "select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME"
            ).fetchone()
        self.load_nvtx()
        self.load_api()
        self.load_device()
        self.load_host_states()
        self.index_host_states()

    def text(self, text, text_id):
        return text if text is not None else self.strings.get(text_id, "")

    def load_nvtx(self):
        self.ranges: dict[int, list[Range]] = collections.defaultdict(list)
        self.capture: dict[int, list[Range]] = collections.defaultdict(list)
        self.gil_wait: dict[int, list] = collections.defaultdict(list)
        self.gil_hold: dict[int, list] = collections.defaultdict(list)
        self.thread_names: dict[int, str] = {}
        self.marks: list[tuple[int, int, str]] = []
        last_mark: dict[int, tuple[int, str]] = {}
        for start, end, tid, text, text_id, kind in self.db.execute(
            "select start, end, globalTid, text, textId, eventType from NVTX_EVENTS "
            "where eventType in (?, ?) order by start",
            (PUSH_POP, MARK),
        ):
            label = self.text(text, text_id)
            if not label:
                continue
            if kind == MARK:
                if label.startswith("thread name="):
                    self.thread_names[tid] = label.split(" ")[1][5:]
                elif self.t0 <= start <= self.t1:
                    # a probe before 09-29 marked a get_nowait twice, back to back
                    previous = last_mark.get(tid)
                    last_mark[tid] = (start, label)
                    if (
                        previous is not None
                        and previous[1] == label
                        and ".get " in label
                        and start - previous[0] < 20_000
                    ):
                        continue
                    self.marks.append((start, tid, label))
                continue
            if end is None:
                continue
            if label.startswith("Holding GIL"):
                if end > self.t0 and start < self.t1:
                    self.gil_hold[tid].append((start, end))
                continue
            if label.startswith("Waiting for GIL"):
                if end > self.t0 and start < self.t1:
                    self.gil_wait[tid].append((start, end))
                continue
            item = Range(kind_of(label), label, start, end, tid)
            if item.kind in CAPTURE_KINDS:
                self.capture[tid].append(item)
            elif end > self.t0 and start < self.t1:
                self.ranges[tid].append(item)
        for table in (self.ranges, self.capture):
            for items in table.values():
                stack: list[Range] = []
                for item in items:
                    while stack and stack[-1].end < item.start:
                        stack.pop()
                    item.parent = stack[-1] if stack else None
                    stack.append(item)
        self.starts = {
            tid: [r.start for r in items] for tid, items in self.ranges.items()
        }
        self.capture_starts = {
            tid: [r.start for r in items] for tid, items in self.capture.items()
        }
        if "ThreadNames" in self.tables:
            for tid, name in self.db.execute(
                "select t.globalTid, s.value from ThreadNames t join StringIds s on t.nameId = s.id"
            ):
                self.thread_names.setdefault(tid, name)

    def innermost(self, table, starts, tid, t):
        items = table.get(tid)
        if not items:
            return None
        index = bisect.bisect_right(starts[tid], t) - 1
        if index < 0:
            return None
        item = items[index]
        while item is not None and item.end < t:
            item = item.parent
        return item

    def load_api(self):
        self.owner: dict[int, Range | None] = {}
        self.owner_cache: dict[int, str] = {}
        self.api_tid: dict[int, int] = {}
        self.api_start: dict[int, int] = {}
        self.api_by_tid: dict[int, list] = collections.defaultdict(list)
        for start, end, tid, corr, name_id in self.db.execute(
            "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME "
            "where end > ? and start < ? order by start",
            (self.t0, self.t1),
        ):
            name = self.strings.get(name_id, "")
            item = self.innermost(self.ranges, self.starts, tid, start)
            self.owner[corr] = item
            self.api_tid[corr] = tid
            self.api_start[corr] = start
            is_launch = name.startswith(LAUNCH)
            is_graph = name.startswith(GRAPH_LAUNCH)
            is_sync = name.startswith(SYNC)
            self.api_by_tid[tid].append(
                (
                    start,
                    end,
                    (
                        "sync"
                        if is_sync
                        else "launch" if (is_launch or is_graph) else "api"
                    ),
                )
            )
            node = item
            while node is not None:
                node.launches += is_launch
                node.graph_launches += is_graph
                node.api_ns += end - start
                if is_sync:
                    node.syncs += 1
                    node.sync_ns += end - start
                node = node.parent

    def owner_name(self, item: Range | None) -> str:
        key = id(item)
        name = self.owner_cache.get(key)
        if name is None:
            name = owner_of(item)
            self.owner_cache[key] = name
        return name

    def load_device(self):
        """Device intervals: eager and node kernels, whole graph replays, copies."""
        self.device: list[tuple[int, int, str, str, int, object]] = []
        cols = [
            r[1]
            for r in self.db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
        ]
        node_col = "graphNodeId" if "graphNodeId" in cols else "null"
        self.node_mode = False
        for start, end, corr, name_id, node, stream, gx, gy, gz, bx in self.db.execute(
            f"select start, end, correlationId, demangledName, {node_col}, streamId, gridX, gridY, gridZ, blockX "
            "from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?",
            (self.t0, self.t1),
        ):
            item = self.owner.get(corr)
            name = self.strings.get(name_id, str(name_id))
            if node is not None:
                self.node_mode = True
            ident = (name, node) if node is not None else (name, gx, gy, gz, bx)
            self.device.append(
                (start, end, self.owner_name(item), name, stream, ident, corr, node)
            )
            while item is not None:
                item.kernels.append((start, end))
                item = item.parent
        if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in self.tables:
            for start, end, corr, graph_id, stream in self.db.execute(
                "select start, end, correlationId, graphId, streamId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE "
                "where end > ? and start < ?",
                (self.t0, self.t1),
            ):
                item = self.owner.get(corr)
                self.device.append(
                    (
                        start,
                        end,
                        self.owner_name(item),
                        f"graph {graph_id}",
                        stream,
                        ("graph", graph_id),
                        corr,
                        None,
                    )
                )
                while item is not None:
                    item.kernels.append((start, end))
                    item = item.parent
        for start, end, corr, nbytes, kind in self.db.execute(
            "select start, end, correlationId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where end > ? and start < ?",
            (self.t0, self.t1),
        ):
            item = self.owner.get(corr)
            label = COPY_KINDS.get(kind, str(kind))
            while item is not None:
                item.copies[label] += 1
                item = item.parent
        self.device.sort(key=lambda d: (d[0], d[1]))

    def load_host_states(self):
        self.os_wait: dict[int, list] = collections.defaultdict(list)
        if "OSRT_API" not in self.tables:
            return
        for start, end, tid, name_id in self.db.execute(
            "select start, end, globalTid, nameId from OSRT_API where end > ? and start < ? order by start",
            (self.t0, self.t1),
        ):
            self.os_wait[tid].append((start, end, self.strings.get(name_id, "")))

    def index_host_states(self):
        self.api_starts = {tid: [a[0] for a in v] for tid, v in self.api_by_tid.items()}
        for items in self.gil_wait.values():
            items.sort()
        self.gil_starts = {tid: [g[0] for g in v] for tid, v in self.gil_wait.items()}
        self.os_starts = {tid: [o[0] for o in v] for tid, v in self.os_wait.items()}
        self.top_level = {}
        for tid, items in self.ranges.items():
            tops = [item for item in items if item.parent is None]
            self.top_level[tid] = (tops, [item.start for item in tops])

    def host_split(self, tid, s, e):
        """CUDA API, GIL wait, OS blocking (by call), rest; in that priority."""
        api = self.api_by_tid.get(tid, [])
        api_cov = merged(clip(api, self.api_starts[tid], s, e)) if api else []
        api_ns = sum(b - a for a, b in api_cov)
        gil = self.gil_wait.get(tid, [])
        gil_cl = merged(clip(gil, self.gil_starts[tid], s, e)) if gil else []
        gil_ns = minus_len(gil_cl, api_cov)
        cover = merged(api_cov + gil_cl)
        os_calls = self.os_wait.get(tid, [])
        by_call = collections.Counter()
        if os_calls:
            i = max(bisect.bisect_right(self.os_starts[tid], s) - 1, 0)
            while i < len(os_calls) and os_calls[i][0] < e:
                a, b, name = os_calls[i]
                if b > s:
                    by_call[name] += minus_len([[max(a, s), min(b, e)]], cover)
                i += 1
        os_ns = sum(by_call.values())
        return api_ns, gil_ns, os_ns, by_call

    def tname(self, tid) -> str:
        name = self.thread_names.get(tid, str(tid))
        return re.sub(r"_\d+$", "_N", re.sub(r"-\d+$", "-N", name))


def section_a(r: Report):
    print("\n## A. gates")
    unknown = [d for d in r.device if d[2] == "unknown"]
    total = sum(d[1] - d[0] for d in r.device)
    print(
        f"device intervals {len(r.device)}, node trace {r.node_mode}; unknown owner "
        f"{len(unknown)} ({ms(sum(d[1] - d[0] for d in unknown)):.1f} ms of {ms(total):.1f})"
    )
    names = collections.Counter(d[3][:70] for d in unknown)
    for name, count in names.most_common(5):
        print(f"  unknown: {count} x {name}")
    threads = collections.Counter(r.tname(tid) for tid in r.ranges)
    print(f"threads with ranges: {dict(threads)}")
    points = collections.Counter(label.split(" rid=")[0] for _, _, label in r.marks)
    print("marks: " + ", ".join(f"{k} {v}" for k, v in sorted(points.items())))


def section_b(r: Report, top: int):
    print("\n## B. components (per instance means, ms)")
    print("  host split inside the range on its own thread: api = CUDA runtime calls,")
    print(
        "  gil = waiting for the GIL, os = blocking OS calls outside both, py = the rest"
    )
    kinds: dict[str, list[Range]] = collections.defaultdict(list)
    for items in r.ranges.values():
        for item in items:
            if item.start >= r.t0 and item.end <= r.t1:
                kinds[item.kind].append(item)
    head = (
        f"{'kind':<24}{'n':>7}{'wall':>8}{'p50':>7}{'p95':>7}{'api':>7}{'gil':>6}{'os':>6}{'py':>7}"
        f"{'dev':>7}{'kern':>7}{'launch':>7}{'graph':>6}{'sync':>5}{'syncms':>7}{'h2d':>5}{'d2h':>5}{'d2d':>5}  top os call"
    )
    print(head)
    rows = sorted(kinds.items(), key=lambda kv: -sum(x.wall for x in kv[1]))
    for kind, items in rows[:top]:
        n = len(items)
        sample = items if n <= 3000 else items[:: n // 3000 + 1]
        api = gil = os_ns = 0
        calls = collections.Counter()
        for item in sample:
            a, g, o, c = r.host_split(item.tid, item.start, item.end)
            api += a
            gil += g
            os_ns += o
            calls.update(c)
        m = len(sample)
        wall = statistics.fmean(x.wall for x in items)
        dev = statistics.fmean(union_ns(x.kernels) for x in items)
        top_call = ",".join(f"{k}:{ms(v) / m:.2f}" for k, v in calls.most_common(2))
        print(
            f"{kind[:24]:<24}{n:>7}{ms(wall):>8.2f}{ms(pct([x.wall for x in items], .5)):>7.2f}"
            f"{ms(pct([x.wall for x in items], .95)):>7.2f}{ms(api / m):>7.2f}{ms(gil / m):>6.2f}"
            f"{ms(os_ns / m):>6.2f}{ms(wall - (api + gil + os_ns) / m):>7.2f}{ms(dev):>7.2f}"
            f"{statistics.fmean(len(x.kernels) for x in items):>7.0f}"
            f"{statistics.fmean(x.launches for x in items):>7.1f}"
            f"{statistics.fmean(x.graph_launches for x in items):>6.1f}"
            f"{statistics.fmean(x.syncs for x in items):>5.1f}"
            f"{ms(statistics.fmean(x.sync_ns for x in items)):>7.2f}"
            f"{statistics.fmean(x.copies['h2d'] for x in items):>5.1f}"
            f"{statistics.fmean(x.copies['d2h'] for x in items):>5.1f}"
            f"{statistics.fmean(x.copies['d2d'] for x in items):>5.1f}  {top_call}"
        )


HOPS = (
    ("preprocessing.in.put", "preprocessing.in.get", "pre inbox wait"),
    ("preprocessing.in.get", "pre.payload", "pre dispatch to worker"),
    ("pre.payload/end", "preprocessing.out.put", "pre finish to outbox"),
    ("preprocessing.out.put", "preprocessing.out.get", "pre outbox drain wake"),
    ("preprocessing.out.get", "tts_engine.in.put", "loop route pre to engine"),
    ("tts_engine.in.put", "tts_engine.in.get", "engine inbox wait"),
    ("tts_engine.in.get", "sched.build", "engine get to build start"),
    ("sched.build/end", "sched.admit", "build end to admitted"),
    ("sched.admit", "sched.prefill", "admitted to prefill launch"),
    ("sched.prefill", "tts_engine.out.put:stream", "prefill launch to first frame out"),
)
FRAME_HOPS = (
    (
        "tts_engine.out.put:stream",
        "tts_engine.out.get:stream",
        "frame outbox drain wake",
    ),
    (
        "tts_engine.out.get:stream",
        "vocoder.in.put:stream_chunk",
        "loop route frame to vocoder",
    ),
    (
        "vocoder.in.put:stream_chunk",
        "vocoder.in.get:stream_chunk",
        "vocoder inbox wait",
    ),
    ("vocoder.in.get:stream_chunk", "voc.ingest", "vocoder get to ingest"),
)
CHUNK_HOPS = (
    ("voc.q.put", "voc.q.get", "decode queue wait (gather window included)"),
    ("voc.q.get", "voc.commit*", "taken to commit start (plan, launch, gpu, resolve)"),
    ("voc.commit*", "vocoder.out.put:stream", "commit start to outbox put"),
    ("vocoder.out.put:stream", "vocoder.out.get:stream", "chunk outbox drain wake"),
    ("vocoder.out.get:stream", "coord.recv", "loop, zmq, coordinator"),
)


def section_c(r: Report):
    print("\n## C. hops (ms; consumer = the thread that ends the hop)")
    points: dict[str, dict[str, list[tuple[int, int]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    seen_initial: set[tuple[str, str]] = set()
    for t, tid, label in r.marks:
        match = re.match(r"q (\S+) rid=(\S+) t=(\S+)", label)
        if match:
            points[f"{match.group(1)}:{match.group(3)}"][match.group(2)].append(
                (t, tid)
            )
            points[match.group(1)][match.group(2)].append((t, tid))
            if match.group(1).startswith("voc_"):
                queue_op = "voc.q." + match.group(1).rsplit(".", 1)[1]
                points[queue_op][match.group(2)].append((t, tid))
            continue
        # reports before 09-29 12:30 marked voc.put on every schedule_initial call,
        # which returns early on a pending stream: only the first initial put counts
        match = re.match(r"(voc\.put|voc\.take) (\S+) rid=(\S+)", label)
        if match:
            key = "voc.q.put" if match.group(1) == "voc.put" else "voc.q.get"
            rid = match.group(3)
            if match.group(2) == "initial" and (key, rid) in seen_initial:
                continue
            if match.group(2) == "initial":
                seen_initial.add((key, rid))
            points[key][rid].append((t, tid))
            continue
        match = re.match(r"(\S+) rid=(\S+)", label)
        if match:
            points[match.group(1)][match.group(2)].append((t, tid))
    for items in r.ranges.values():
        for item in items:
            rid = item.rid()
            if rid is None:
                continue
            key = (
                "voc.commit*"
                if item.kind in ("voc.commit", "voc.commit_followup")
                else item.kind
            )
            points[key][rid].append((item.start, item.tid))
            points[key + "/end"][rid].append((item.end, item.tid))
    for series in points.values():
        for values in series.values():
            values.sort()

    def report(pairs, first_only):
        print(
            f"  {'hop':<46}{'n':>7}{'mean':>8}{'p50':>8}{'p95':>8}{'max':>8}  consumer busy in"
        )
        for a, b, name in pairs:
            gaps, busy = [], collections.Counter()
            for rid, starts in points.get(a, {}).items():
                ends = points.get(b, {}).get(rid, [])
                count = 1 if first_only else min(len(starts), len(ends))
                for k in range(min(count, len(starts), len(ends))):
                    (ta, _), (tb, tid) = starts[k], ends[k]
                    if tb < ta:
                        continue
                    gaps.append(tb - ta)
                    if len(gaps) <= 3000 and tid in r.top_level:
                        tops, top_starts = r.top_level[tid]
                        i = max(bisect.bisect_right(top_starts, ta) - 1, 0)
                        covered = 0
                        while i < len(tops) and tops[i].start < tb:
                            item = tops[i]
                            if item.end > ta:
                                part = min(tb, item.end) - max(ta, item.start)
                                busy[item.kind] += part
                                covered += part
                            i += 1
                        busy["(no range)"] += (tb - ta) - covered
            if not gaps:
                print(f"  {name:<46}{0:>7}")
                continue
            total = sum(busy.values()) or 1
            share = " ".join(
                f"{k} {100 * v / total:.0f}%" for k, v in busy.most_common(3)
            )
            print(
                f"  {name:<46}{len(gaps):>7}{ms(statistics.fmean(gaps)):>8.2f}{ms(pct(gaps, .5)):>8.2f}"
                f"{ms(pct(gaps, .95)):>8.2f}{ms(max(gaps)):>8.2f}  {share}"
            )

    print(" request, first occurrence per request")
    report(HOPS, True)
    print(" per codec frame, in order per request")
    report(FRAME_HOPS, False)
    print(" per audio chunk, in order per request")
    report(CHUNK_HOPS, False)
    counts = {
        p: sum(len(v) for v in points.get(p, {}).values())
        for p in (
            "tts_engine.out.put:stream",
            "vocoder.in.get:stream_chunk",
            "voc.ingest",
            "voc.q.put",
            "voc.q.get",
            "voc.commit*",
            "vocoder.out.put:stream",
            "coord.recv",
        )
    }
    print("  pairing counts: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


def section_d(r: Report, top: int):
    print("\n## D. device (ms over the window)")
    window = r.t1 - r.t0
    events = []
    for index, (s, e, owner, *_rest) in enumerate(r.device):
        events.append((max(s, r.t0), 1, index))
        events.append((min(e, r.t1), 0, index))
    events.sort()
    active: dict[int, str] = {}
    by_owner_count = collections.Counter()
    last = r.t0
    busy = 0
    exclusive = collections.Counter()
    total = collections.Counter()
    pair = collections.Counter()
    overlapped = collections.defaultdict(int)
    for t, is_start, index in events:
        span = t - last
        if span > 0 and active:
            busy += span
            owners = set(by_owner_count)
            for owner in owners:
                total[owner] += span
            if len(owners) == 1:
                exclusive[next(iter(owners))] += span
            else:
                ordered = sorted(owners)
                for i in range(len(ordered)):
                    for j in range(i + 1, len(ordered)):
                        pair[(ordered[i], ordered[j])] += span
                for other_index, owner in active.items():
                    overlapped[other_index] += span
        last = t
        owner = r.device[index][2]
        if is_start:
            active[index] = owner
            by_owner_count[owner] += 1
        else:
            active.pop(index, None)
            by_owner_count[owner] -= 1
            if by_owner_count[owner] <= 0:
                del by_owner_count[owner]
    print(
        f"window {ms(window):.0f}, device busy {ms(busy):.0f} ({100 * busy / window:.1f}%), empty {ms(window - busy):.0f}"
    )
    print(f"  {'owner':<16}{'busy':>9}{'share':>7}{'alone':>9}{'shared':>9}")
    for owner, value in total.most_common():
        print(
            f"  {owner:<16}{ms(value):>9.0f}{100 * value / busy:>6.1f}%{ms(exclusive[owner]):>9.0f}{ms(value - exclusive[owner]):>9.0f}"
        )
    print("  running at once (ms):")
    for (a, b), value in pair.most_common(10):
        print(f"    {a} + {b}: {ms(value):.0f}")
    # per kernel identity: duration alone against with another owner running beside it
    alone = collections.defaultdict(list)
    shared = collections.defaultdict(list)
    weight = collections.Counter()
    for index, (s, e, owner, name, stream, ident, corr, node) in enumerate(r.device):
        duration = e - s
        if duration <= 0:
            continue
        other = 0
        if index in overlapped:
            other = overlapped[index]
        key = (owner, ident)
        weight[key] += duration
        if other < 0.05 * duration:
            alone[key].append(duration)
        elif other > 0.5 * duration:
            shared[key].append(duration)
    rows = []
    for key, total_ns in weight.most_common(4000):
        a, b = alone.get(key, []), shared.get(key, [])
        if len(a) >= 10 and len(b) >= 10:
            ma, mb = statistics.median(a), statistics.median(b)
            rows.append((total_ns, key, ma, mb, len(a), len(b)))
    print(
        f"  slowdown beside other kernels (identity = name and graph node, or name and grid; {len(rows)} with 10+ each):"
    )
    cost = collections.Counter()
    for total_ns, (owner, ident), ma, mb, na, nb in rows:
        cost[owner] += (mb - ma) * nb
    for owner, value in cost.most_common():
        print(f"    {owner:<16} extra device time from sharing {ms(value):.0f} ms")
    rows.sort(key=lambda row: -(row[3] - row[2]) * row[5])
    for total_ns, (owner, ident), ma, mb, na, nb in rows[:top]:
        print(
            f"    {owner:<14}{str(ident[0])[:60]:<62} alone {ma / 1e3:7.1f} us x{na:<6} shared {mb / 1e3:7.1f} us x{nb:<6} +{100 * (mb - ma) / ma:5.1f}%"
        )


def section_e(r: Report):
    print("\n## E. GIL (ms over the window)")
    if not r.gil_hold:
        print("  no python-gil trace in this report")
        return
    window = r.t1 - r.t0
    names = {}
    hold_lists = {}
    for tid, items in r.gil_hold.items():
        items.sort()
        hold_lists[tid] = (items, [s for s, _ in items])
        names[tid] = r.tname(tid)
    rows = []
    for tid in set(r.gil_hold) | set(r.gil_wait):
        hold = union_ns(r.gil_hold.get(tid, []))
        wait = union_ns(r.gil_wait.get(tid, []))
        rows.append((hold, wait, r.tname(tid)))
    grouped = collections.defaultdict(lambda: [0, 0, 0])
    for hold, wait, name in rows:
        grouped[name][0] += hold
        grouped[name][1] += wait
        grouped[name][2] += 1
    print(f"  {'thread':<36}{'n':>4}{'hold':>9}{'hold %':>8}{'wait':>9}")
    for name, (hold, wait, n) in sorted(grouped.items(), key=lambda kv: -kv[1][0]):
        print(
            f"  {name[:36]:<36}{n:>4}{ms(hold):>9.0f}{100 * hold / window:>7.1f}%{ms(wait):>9.0f}"
        )
    matrix = collections.Counter()
    for tid, waits in r.gil_wait.items():
        for s, e in waits:
            for other, (items, starts) in hold_lists.items():
                if other == tid:
                    continue
                for a, b in clip(items, starts, s, e):
                    matrix[(r.tname(tid), r.tname(other))] += b - a
    print("  waiter <- holder (ms):")
    for (waiter, holder), value in matrix.most_common(15):
        print(f"    {waiter[:30]:<30} <- {holder[:30]:<30} {ms(value):8.0f}")


def section_f(r: Report):
    print("\n## F. first chunk critical path (ms, per request)")
    first = collections.defaultdict(dict)
    for items in r.ranges.values():
        for item in items:
            rid = item.rid()
            if rid and item.kind not in first[rid]:
                first[rid][item.kind] = item
    marks = collections.defaultdict(dict)
    for t, tid, label in r.marks:
        match = re.match(r"(q \S+|\S+)(?: \S+)? rid=(\S+)", label)
        if match:
            key = match.group(1)
            if key.startswith("q "):
                key += ":" + label.rsplit("t=", 1)[-1]
            marks[match.group(2)].setdefault(key, t)
    segments = collections.defaultdict(list)
    for rid, kinds in first.items():
        m = marks.get(rid, {})
        pay = kinds.get("pre.payload")
        commit = kinds.get("voc.commit")
        if pay is None or commit is None:
            continue
        points = [
            ("pre inbox put", m.get("q preprocessing.in.put:new_request")),
            ("pre.payload start", pay.start),
            ("pre.payload end", pay.end),
            ("engine inbox put", m.get("q tts_engine.in.put:new_request")),
            (
                "build start",
                kinds["sched.build"].start if "sched.build" in kinds else None,
            ),
            ("admitted", m.get("sched.admit")),
            ("prefill launch", m.get("sched.prefill")),
            ("first frame out", m.get("q tts_engine.out.put:stream")),
            (
                "vocoder ingest",
                kinds["voc.ingest"].start if "voc.ingest" in kinds else None,
            ),
            ("decode queued", m.get("q voc_initial.put:-", m.get("voc.put"))),
            ("decode taken", m.get("q voc_initial.get:-", m.get("voc.take"))),
            ("first chunk committed", commit.end),
            ("chunk at coordinator", m.get("coord.recv")),
        ]
        known = [(name, t) for name, t in points if t is not None]
        for (na, ta), (nb, tb) in zip(known, known[1:]):
            segments[f"{na} -> {nb}"].append(tb - ta)
        segments["TOTAL pre inbox put -> chunk at coordinator"].append(
            (m.get("coord.recv") or commit.end)
            - (m.get("q preprocessing.in.put:new_request") or pay.start)
        )
    print(f"  {'segment':<56}{'n':>6}{'mean':>8}{'p50':>8}{'p95':>8}")
    for name, values in segments.items():
        print(
            f"  {name:<56}{len(values):>6}{ms(statistics.fmean(values)):>8.2f}{ms(pct(values, .5)):>8.2f}{ms(pct(values, .95)):>8.2f}"
        )


def section_h(r: Report, sample_rate: int = 24000):
    print(
        "\n## H. playback margin at the coordinator (ms; audio delivered minus time since the first chunk)"
    )
    arrivals = collections.defaultdict(list)
    for t, tid, label in r.marks:
        match = re.match(r"coord\.recv rid=(\S+) n=(\d+)", label)
        if match:
            arrivals[match.group(1)].append((t, int(match.group(2))))
    if not arrivals:
        print("  no coord.recv marks with a sample count in this report")
        return
    margins, late, stalls, gaps = [], 0, [], []
    for rid, chunks in arrivals.items():
        chunks.sort()
        first = chunks[0][0]
        delivered = 0.0
        stall = 0.0
        for index, (t, samples) in enumerate(chunks):
            if index:
                margin = first + delivered * 1e9 + stall - t
                margins.append(margin)
                gaps.append(t - chunks[index - 1][0])
                if margin < 0:
                    late += 1
                    stall += -margin
            delivered += samples / sample_rate
        stalls.append(stall)
    print(
        f"  requests {len(arrivals)}, chunks after the first {len(margins)}, late {late} ({100 * late / max(len(margins), 1):.2f}%)"
    )
    print(
        f"  margin p5 {ms(pct(margins, .05)):.1f}  p50 {ms(pct(margins, .5)):.1f}  min {ms(min(margins)) if margins else 0:.1f}"
    )
    print(
        f"  inter chunk gap p50 {ms(pct(gaps, .5)):.1f}  p95 {ms(pct(gaps, .95)):.1f}; stall per request mean {ms(statistics.fmean(stalls)):.2f}  max {ms(max(stalls)):.1f}"
    )


def section_g(r: Report, top: int):
    print("\n## G. attribution: device time by capture site and op call site")
    node_rows = (
        r.db.execute(
            "select start, globalTid, graphNodeId, originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS"
        ).fetchall()
        if "CUDA_GRAPH_NODE_EVENTS" in r.tables
        else []
    )
    original, created = {}, {}
    for start, tid, node, orig in node_rows:
        if orig is not None:
            original[node] = orig
        elif node not in created:
            created[node] = (start, tid)
    graph_rows = (
        r.db.execute(
            "select start, globalTid, graphId from CUDA_GRAPH_EVENTS"
        ).fetchall()
        if "CUDA_GRAPH_EVENTS" in r.tables
        else []
    )
    graph_created = {}
    for start, tid, graph_id in graph_rows:
        graph_created.setdefault(graph_id, (start, tid))

    def capture_labels(start, tid):
        item = r.innermost(r.capture, r.capture_starts, tid, start)
        op = cap = None
        while item is not None:
            if item.kind == "op" and op is None:
                op = item.label
            if item.kind == "cap" and cap is None:
                cap = item.label
            item = item.parent
        return cap, op

    by_site = collections.Counter()
    by_omni = collections.Counter()
    by_cap = collections.Counter()
    labeled = unlabeled = 0
    owner_total = collections.Counter()
    for start, end, owner, name, stream, ident, corr, node in r.device:
        duration = end - start
        owner_total[owner] += duration
        cap = op = None
        if node is not None:
            orig = node
            while orig in original:
                orig = original[orig]
            origin = created.get(orig)
            if origin is not None:
                cap, op = capture_labels(*origin)
        elif name.startswith("graph "):
            origin = graph_created.get(ident[1])
            if origin is not None:
                cap, _ = capture_labels(*origin)
        else:
            tid = r.api_tid.get(corr)
            if tid is not None:
                launched = r.api_start[corr]
                item = r.innermost(r.capture, r.capture_starts, tid, launched)
                if item is not None and item.kind == "op":
                    op = item.label
        if op is not None:
            labeled += duration
            match = re.match(r"op (\S+) @(\S+) <(\S+)>", op)
            if match:
                by_site[(owner, match.group(1), match.group(2), name[:48])] += duration
                by_omni[(owner, match.group(3))] += duration
        else:
            unlabeled += duration
        if cap is not None:
            by_cap[(owner, cap[4:120])] += duration
    total = labeled + unlabeled
    print(
        f"  device time with an op label {ms(labeled):.0f} ms of {ms(total):.0f} ({100 * labeled / max(total, 1):.1f}%)"
    )
    print("  by capture site (graph kernels and whole replays):")
    for (owner, cap), value in by_cap.most_common(top):
        print(
            f"    {owner:<14}{ms(value):>9.0f} ms {100 * value / max(owner_total[owner], 1):5.1f}%  {cap}"
        )
    print("  by omni line (innermost sglang_omni frame of the op):")
    for (owner, site), value in by_omni.most_common(top):
        print(
            f"    {owner:<14}{ms(value):>9.0f} ms {100 * value / max(owner_total[owner], 1):5.1f}%  {site}"
        )
    print("  by op, library line and kernel:")
    for (owner, op, site, name), value in by_site.most_common(top):
        print(
            f"    {owner:<14}{ms(value):>9.0f} ms {100 * value / max(owner_total[owner], 1):5.1f}%  {op:<22} {site:<60} {name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--bench-log")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--sections", default="ABCDEFGH")
    args = parser.parse_args()
    r = Report(args.report, args.bench_log)
    print(
        f"window {r.t0 / 1e9:.3f} .. {r.t1 / 1e9:.3f} s ({(r.t1 - r.t0) / 1e9:.1f} s)"
    )
    steps = {
        "A": lambda: section_a(r),
        "B": lambda: section_b(r, args.top),
        "C": lambda: section_c(r),
        "D": lambda: section_d(r, args.top),
        "E": lambda: section_e(r),
        "F": lambda: section_f(r),
        "G": lambda: section_g(r, args.top),
        "H": lambda: section_h(r),
    }
    for key in args.sections:
        steps[key]()


if __name__ == "__main__":
    main()
