"""Whole pipeline census of a Qwen3-Omni serve from one nsys sqlite export (run on the box).

Needs the omni_pipeline_nvtx probe on the profiled server. Processes are named by their
"proc stage=" marks; threads by their "thread name=" marks. Every CUDA runtime call
belongs to the innermost probe range on its thread; its kernels, graph replays, copies and
memsets belong to that range and every range enclosing it. Correlation ids are unique per
process only, so every join is keyed by (pid, correlation id). Graph node kernels (node
trace) resolve through the node creation events to the capture time op range that created
them. Host time inside a range splits into CUDA API calls, GIL waits (python-gil trace),
OS runtime blocking and the rest. Owner of device work: the stage of the launching thread
(its scheduler thread "scheduler-<stage>", else the process's stages) and the component
of the innermost range chain.

Sections:
  A gates          window, unknown owners, captures or compiles inside the window, marks
  B components     per stage and range kind: wall, host split, device, kernels, launches,
                   graphs, syncs, copies
  C hops           every queue (put to get) and every stage-to-stage send (send start to
                   receive start), per request in order, with what the consumer was doing
  D card           per stage: device busy, alone and contended with another process;
                   context residency and switches (with --gpuctxsw); per kernel duration
                   alone against contended, the extra device time from sharing
  E GIL            per process thread: hold, wait; waiter by holder
  F request path   per request, the first occurrence of every point in the observed order
                   (coordinator submit to last audio chunk), segment by segment
  G attribution    device time by capture site and by op and call site
  H playback       per audio chunk margin at the coordinator, late chunks, stalls
  I engine steps   thinker and talker: per mode, rows, tokens, wall, device, kernels,
                   graph launches, host split, step period
  J table          per completed request: device ms by owner, and the request path's
                   segments ranked; the census's ranking in one place
  K activity       where the window went: a charged kernel, a resident context with no kernel,
                   switching, no context; with --dcgm (H100 host engine) DCGM GR active
                   against the trace's charged kernel time, each field's mean, and per owner
                   the value of each field while its kernels run (fitted over 100 ms windows)

usage: python omni_census.py REPORT.sqlite --window window.txt [--top 25] [--sections ABCDEFGHIJ]
       [--sample-rate 24000] [--dcgm samples.tsv --dcgm-gpu 0]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics
from dataclasses import dataclass, field

import numpy
from nsys_stage_ledger import pid_of, session_window

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
RESTORE_START = 8
SAVE_END = 7
# a gap between two residency slices up to this long is the context switch itself
SWITCH_GAP_NS = 200_000
COMPONENT_RULES = (
    ("mr.predictor", "predictor"),
    ("c2w", "code2wav"),
    ("tk.build", "build"),
    ("sched.build", "build"),
    ("sched.batch decode", "decode"),
    ("sched.launch decode", "decode"),
    ("sched.resolve decode", "decode"),
    ("sched.batch extend", "prefill"),
    ("sched.launch extend", "prefill"),
    ("sched.batch mixed", "mixed"),
    ("sched.launch mixed", "mixed"),
    ("sched", "engine.other"),
    ("mr", "engine.other"),
    ("enc", "encoder"),
    ("pre", "preprocessing"),
    ("io", "io"),
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
    if words[0] in (
        "sched.batch",
        "sched.launch",
        "sched.resolve",
        "sched.result",
        "io.ser",
        "io.deser",
    ):
        return " ".join(words[:2])
    return words[0]


def component_of(item: Range | None) -> str:
    if item is None:
        return "unranged"
    kinds = item.chain()
    for prefix, component in COMPONENT_RULES:
        for kind in kinds:
            if (
                kind == prefix
                or kind.startswith(prefix + ".")
                or kind.startswith(prefix + " ")
            ):
                return component
    return "other:" + kinds[-1]


def union_ns(intervals) -> int:
    total, current_start, current_end = 0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        elif end > current_end:
            current_end = end
    if current_end is not None:
        total += current_end - current_start
    return total


def merged(intervals):
    out = []
    for start, end in sorted(intervals):
        if out and start <= out[-1][1]:
            if end > out[-1][1]:
                out[-1][1] = end
        else:
            out.append([start, end])
    return out


def clip(sorted_intervals, starts, s, e):
    """Intervals of one thread's non overlapping list that fall inside [s, e], clipped."""
    out = []
    index = max(bisect.bisect_right(starts, s) - 1, 0)
    while index < len(sorted_intervals):
        a, b = sorted_intervals[index][0], sorted_intervals[index][1]
        if a >= e:
            break
        if b > s:
            out.append([max(a, s), min(b, e)])
        index += 1
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
    def __init__(self, path: str, window: str | None):
        self.db = sqlite3.connect(path)
        self.strings = dict(self.db.execute("select id, value from StringIds"))
        self.tables = {
            r[0]
            for r in self.db.execute(
                "select name from sqlite_master where type='table'"
            )
        }
        self.t0, self.t1 = session_window(self.db, window)
        self.load_nvtx()
        self.load_api()
        self.load_device()
        self.load_residency()
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
        self.stages_of_pid: dict[int, list[str]] = collections.defaultdict(list)
        self.marks: list[tuple[int, int, str]] = []
        self.window_captures = 0
        for start, end, tid, text, text_id, kind in self.db.execute(
            "select start, end, globalTid, text, textId, eventType from NVTX_EVENTS "
            "where eventType in (?, ?) order by start, end desc",
            (PUSH_POP, MARK),
        ):
            label = self.text(text, text_id)
            if not label:
                continue
            if kind == MARK:
                if label.startswith("thread name="):
                    self.thread_names[tid] = label.split(" ")[1][5:]
                elif label.startswith("proc stage="):
                    stage = label.split("=", 1)[1]
                    if stage not in self.stages_of_pid[pid_of(tid)]:
                        self.stages_of_pid[pid_of(tid)].append(stage)
                elif self.t0 <= start <= self.t1:
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
                if item.kind == "cap" and end > self.t0 and start < self.t1:
                    self.window_captures += 1
            elif end > self.t0 and start < self.t1:
                self.ranges[tid].append(item)
        for table in (self.ranges, self.capture):
            for items in table.values():
                stack: list[Range] = []
                for item in items:
                    while stack and stack[-1].end <= item.start:
                        stack.pop()
                    item.parent = stack[-1] if stack else None
                    stack.append(item)
        self.starts = {
            tid: [r.start for r in items] for tid, items in self.ranges.items()
        }
        self.capture_starts = {
            tid: [r.start for r in items] for tid, items in self.capture.items()
        }

    def stage_of_tid(self, tid: int) -> str:
        name = self.thread_names.get(tid, "")
        if name.startswith("scheduler-"):
            return name[len("scheduler-") :]
        stages = self.stages_of_pid.get(pid_of(tid))
        return "+".join(sorted(stages)) if stages else f"pid{pid_of(tid)}"

    def stage_of_pid(self, pid: int) -> str:
        stages = self.stages_of_pid.get(pid)
        return "+".join(sorted(stages)) if stages else f"pid{pid}"

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
        self.owner: dict[tuple[int, int], Range | None] = {}
        self.api_tid: dict[tuple[int, int], int] = {}
        self.api_start: dict[tuple[int, int], int] = {}
        self.api_end: dict[tuple[int, int], int] = {}
        self.api_by_tid: dict[int, list] = collections.defaultdict(list)
        for start, end, tid, corr, name_id in self.db.execute(
            "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME "
            "where end > ? and start < ? order by start",
            (self.t0, self.t1),
        ):
            name = self.strings.get(name_id, "")
            key = (pid_of(tid), corr)
            item = self.innermost(self.ranges, self.starts, tid, start)
            self.owner[key] = item
            self.api_tid[key] = tid
            self.api_start[key] = start
            self.api_end[key] = end
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

    def owner_name(self, key: tuple[int, int]) -> str:
        item = self.owner.get(key)
        tid = self.api_tid.get(key)
        stage = self.stage_of_tid(tid) if tid is not None else self.stage_of_pid(key[0])
        return f"{stage}:{component_of(item)}"

    def load_device(self):
        """Device intervals: eager and node kernels, whole graph replays, copies."""
        self.device: list[tuple] = []
        columns = [
            r[1]
            for r in self.db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
        ]
        node_column = "graphNodeId" if "graphNodeId" in columns else "null"
        self.node_mode = False
        for (
            start,
            end,
            corr,
            name_id,
            node,
            stream,
            gx,
            gy,
            gz,
            bx,
            global_pid,
        ) in self.db.execute(
            f"select start, end, correlationId, demangledName, {node_column}, streamId, gridX, gridY, "
            "gridZ, blockX, globalPid from CUPTI_ACTIVITY_KIND_KERNEL "
            "where end > ? and start < ?",
            (self.t0, self.t1),
        ):
            pid = pid_of(global_pid)
            key = (pid, corr)
            item = self.owner.get(key)
            name = self.strings.get(name_id, str(name_id))
            if node is not None:
                self.node_mode = True
            identity = (name, node) if node is not None else (name, gx, gy, gz, bx)
            self.device.append(
                (
                    start,
                    end,
                    self.owner_name(key),
                    name,
                    stream,
                    identity,
                    key,
                    node,
                    pid,
                )
            )
            while item is not None:
                item.kernels.append((start, end))
                item = item.parent
        self.copies_by_stream: dict[tuple[int, int], list] = collections.defaultdict(
            list
        )
        for start, end, corr, kind, global_pid, stream in self.db.execute(
            "select start, end, correlationId, copyKind, globalPid, streamId from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where end > ? and start < ?",
            (self.t0, self.t1),
        ):
            self.copies_by_stream[(pid_of(global_pid), stream)].append((start, end))
            item = self.owner.get((pid_of(global_pid), corr))
            label = COPY_KINDS.get(kind, str(kind))
            while item is not None:
                item.copies[label] += 1
                item = item.parent
        self.device.sort(key=lambda d: (d[0], d[1]))

    def load_residency(self):
        """Context residency slices per process on the session GPU (--gpuctxsw).

        Switch records cover every GPU in the container with host context ids and host pids,
        and kernels carry container pids. A kernel's interval includes time its context was
        switched out, so a context is named by the process whose kernel starts fall inside its
        slices (work starts only while its own context is resident); the contexts of one host
        pid (a process can hold several, e.g. streams of different priority) take the process
        most of their starts name."""
        self.resident: dict[int, list] = {}
        self.context_rows: list = []
        self.switch_gaps: list[int] = []
        self.has_residency = "GPU_CONTEXT_SWITCH_EVENTS" in self.tables
        if not self.has_residency:
            return
        open_since, raw, host_of, gaps = (
            {},
            collections.defaultdict(list),
            {},
            collections.defaultdict(list),
        )
        last_save = {}
        for timestamp, tag, context, gpu, host_pid in self.db.execute(
            "select timestamp, tag, contextId, gpuId, (globalPid >> 24) & 16777215 "
            "from GPU_CONTEXT_SWITCH_EVENTS order by gpuId, timestamp, seqNo"
        ):
            key = (gpu, context)
            host_of[key] = host_pid
            if tag == RESTORE_START:
                open_since[key] = timestamp
                if gpu in last_save and self.t0 <= timestamp <= self.t1:
                    gaps[gpu].append(timestamp - last_save[gpu])
            elif tag == SAVE_END and key in open_since:
                start = open_since.pop(key)
                if timestamp > self.t0 and start < self.t1:
                    raw[key].append((max(start, self.t0), min(timestamp, self.t1)))
                last_save[gpu] = timestamp
        starts_of_pid = collections.defaultdict(list)
        for start, *_rest, pid in self.device:
            starts_of_pid[pid].append(start)
        for values in starts_of_pid.values():
            values.sort()
        counts_of_key = {}
        by_gpu = collections.Counter()
        for key, slices in raw.items():
            counts = collections.Counter()
            for pid, values in starts_of_pid.items():
                counts[pid] = sum(
                    bisect.bisect_left(values, e) - bisect.bisect_left(values, s)
                    for s, e in slices
                )
            counts_of_key[key] = counts
            by_gpu[key[0]] += max(counts.values(), default=0)
        if not by_gpu:
            return
        session_gpu = by_gpu.most_common(1)[0][0]
        by_host = collections.defaultdict(collections.Counter)
        for key, counts in counts_of_key.items():
            if key[0] == session_gpu:
                by_host[host_of[key]].update(counts)
        pid_of_host = {
            host: counts.most_common(1)[0][0]
            for host, counts in by_host.items()
            if counts and counts.most_common(1)[0][1] > 0
        }
        slices_of_pid = collections.defaultdict(list)
        for key, slices in raw.items():
            if key[0] != session_gpu:
                continue
            pid = pid_of_host.get(host_of[key])
            self.context_rows.append((key, host_of[key], pid, slices))
            if pid is not None:
                slices_of_pid[pid] += slices
        self.resident = {pid: merged(v) for pid, v in slices_of_pid.items()}
        self.switch_gaps = gaps.get(session_gpu, [])

    def resident_ns(self, intervals, pid: int) -> int:
        """Time of intervals (kernels of pid) during which pid's own context was resident; the
        plain union when the report has no residency."""
        spans = merged(intervals)
        if not self.has_residency or not self.resident:
            return sum(b - a for a, b in spans)
        own = self.resident.get(pid, [])
        return sum(b - a for a, b in spans) - minus_len(spans, own)

    def charged_spans(self) -> list[list]:
        """Per device interval, the spans charged to it: from the later of its start and the
        end of its stream predecessor (a kernel launched with programmatic dependent launch
        starts early and waits for the predecessor inside its own interval), counted only
        while its own context was resident (the rest is another process's turn)."""
        resident_starts = {
            pid: [a for a, _ in spans] for pid, spans in self.resident.items()
        }
        stream_end: dict[tuple[int, int], int] = {}
        charged = []
        for start, end, owner, name, stream, identity, key, node, pid in self.device:
            previous_end = stream_end.get((pid, stream), start)
            begin = min(max(start, previous_end), end)
            stream_end[(pid, stream)] = max(previous_end, end)
            if pid in self.resident:
                charged.append(
                    clip(self.resident[pid], resident_starts[pid], begin, end)
                )
            else:
                charged.append([[begin, end]])
        return charged

    def charged_durations(self) -> list[int]:
        return [sum(b - a for a, b in spans) for spans in self.charged_spans()]

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
        api_cover = merged(clip(api, self.api_starts[tid], s, e)) if api else []
        api_ns = sum(b - a for a, b in api_cover)
        gil = self.gil_wait.get(tid, [])
        gil_clipped = merged(clip(gil, self.gil_starts[tid], s, e)) if gil else []
        gil_ns = minus_len(gil_clipped, api_cover)
        cover = merged(api_cover + gil_clipped)
        os_calls = self.os_wait.get(tid, [])
        by_call = collections.Counter()
        if os_calls:
            index = max(bisect.bisect_right(self.os_starts[tid], s) - 1, 0)
            while index < len(os_calls) and os_calls[index][0] < e:
                a, b, name = os_calls[index]
                if b > s:
                    by_call[name] += minus_len([[max(a, s), min(b, e)]], cover)
                index += 1
        return api_ns, gil_ns, sum(by_call.values()), by_call

    def thread_label(self, tid) -> str:
        name = self.thread_names.get(tid, str(tid))
        name = re.sub(r"_\d+$", "_N", re.sub(r"-\d+$", "-N", name))
        return f"{self.stage_of_pid(pid_of(tid))}/{name}"

    def completed_requests(self) -> set[str]:
        done = set()
        for _, _, label in self.marks:
            match = re.match(r"coord\.done rid=(\S+)", label)
            if match:
                done.add(match.group(1))
        return done


def section_a(r: Report):
    print("\n## A. gates")
    unknown = [d for d in r.device if ":unranged" in d[2] or d[2].startswith("pid")]
    total = sum(d[1] - d[0] for d in r.device)
    print(
        f"device intervals {len(r.device)}, node trace {r.node_mode}; unranged or unnamed "
        f"{len(unknown)} ({ms(sum(d[1] - d[0] for d in unknown)):.1f} ms of {ms(total):.1f})"
    )
    for name, count in collections.Counter(
        (d[2], d[3][:60]) for d in unknown
    ).most_common(6):
        print(f"  {count} x {name}")
    print(f"graph captures inside the window: {r.window_captures} (must be 0)")
    print(
        "processes: "
        + ", ".join(f"{pid}={'+'.join(s)}" for pid, s in r.stages_of_pid.items())
    )
    threads = collections.Counter(r.thread_label(tid) for tid in r.ranges)
    print(f"threads with ranges: {dict(threads)}")
    points = collections.Counter(
        re.split(r" rid=| n=| bs=", label)[0] for _, _, label in r.marks
    )
    print("marks: " + ", ".join(f"{k} {v}" for k, v in sorted(points.items())))
    print(f"completed requests (coord.done): {len(r.completed_requests())}")
    spawned = collections.Counter()
    for tid, calls in r.os_wait.items():
        for start, end, name in calls:
            if name.startswith("waitpid") or name.startswith("wait4"):
                spawned[r.thread_label(tid)] += 1
    print(
        "subprocess waits inside the window (a compile or a tool run; must be 0 in steady state): "
        + (", ".join(f"{k} {v}" for k, v in spawned.most_common()) or "none")
    )


def section_b(r: Report, top: int):
    print("\n## B. components (per instance means, ms)")
    print("  host split inside the range on its own thread: api = CUDA runtime calls,")
    print(
        "  gil = waiting for the GIL, os = blocking OS calls outside both, py = the rest"
    )
    kinds: dict[tuple[str, str], list[Range]] = collections.defaultdict(list)
    for tid, items in r.ranges.items():
        stage = r.stage_of_tid(tid)
        for item in items:
            if item.start >= r.t0 and item.end <= r.t1:
                kinds[(stage, item.kind)].append(item)
    print(
        f"{'stage':<18}{'kind':<24}{'n':>7}{'wall':>8}{'p50':>7}{'p95':>7}{'api':>7}{'gil':>6}"
        f"{'os':>6}{'py':>7}{'dev':>7}{'kern':>6}{'launch':>7}{'graph':>6}{'sync':>5}"
        f"{'syncms':>7}{'h2d':>5}{'d2h':>5}  top os call"
    )
    rows = sorted(kinds.items(), key=lambda kv: -sum(x.wall for x in kv[1]))
    for (stage, kind), items in rows[:top]:
        count = len(items)
        sample = items if count <= 3000 else items[:: count // 3000 + 1]
        api = gil = os_ns = 0
        calls = collections.Counter()
        for item in sample:
            a, g, o, c = r.host_split(item.tid, item.start, item.end)
            api += a
            gil += g
            os_ns += o
            calls.update(c)
        m = len(sample)
        walls = [x.wall for x in items]
        wall = statistics.fmean(walls)
        device = statistics.fmean(
            r.resident_ns(x.kernels, pid_of(x.tid)) for x in items
        )
        top_call = ",".join(f"{k}:{ms(v) / m:.2f}" for k, v in calls.most_common(2))
        print(
            f"{stage[:18]:<18}{kind[:24]:<24}{count:>7}{ms(wall):>8.2f}{ms(pct(walls, .5)):>7.2f}"
            f"{ms(pct(walls, .95)):>7.2f}{ms(api / m):>7.2f}{ms(gil / m):>6.2f}{ms(os_ns / m):>6.2f}"
            f"{ms(wall - (api + gil + os_ns) / m):>7.2f}{ms(device):>7.2f}"
            f"{statistics.fmean(len(x.kernels) for x in items):>6.0f}"
            f"{statistics.fmean(x.launches for x in items):>7.1f}"
            f"{statistics.fmean(x.graph_launches for x in items):>6.1f}"
            f"{statistics.fmean(x.syncs for x in items):>5.1f}"
            f"{ms(statistics.fmean(x.sync_ns for x in items)):>7.2f}"
            f"{statistics.fmean(x.copies['h2d'] for x in items):>5.1f}"
            f"{statistics.fmean(x.copies['d2h'] for x in items):>5.1f}  {top_call}"
        )


def request_points(r: Report) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """point name -> request -> [(time, tid)] in time order, from marks and ranged calls."""
    points: dict[str, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for t, tid, label in r.marks:
        match = re.match(r"q (\S+) rid=(\S+) t=(\S+)", label)
        if match:
            points[f"q {match.group(1)}:{match.group(3)}"][match.group(2)].append(
                (t, tid)
            )
            continue
        match = re.match(r"(io\.\w+(?:\.end)?) rid=(\S+) (\S+)", label)
        if match:
            points[f"{match.group(1)} {match.group(3)}"][match.group(2)].append(
                (t, tid)
            )
            continue
        match = re.match(r"(coord\.recv) rid=(\S+) from=(\S+)", label)
        if match:
            points[f"coord.recv {match.group(3)}"][match.group(2)].append((t, tid))
            continue
        match = re.match(r"(coord\.done) rid=(\S+) from=(\S+)", label)
        if match:
            points[f"coord.done {match.group(3)}"][match.group(2)].append((t, tid))
            continue
        match = re.match(r"(\S+) rid=(\S+)", label)
        if match:
            name = match.group(1)
            if not name.startswith("coord."):
                name = f"{r.stage_of_tid(tid)}:{name}"
            points[name][match.group(2)].append((t, tid))
    for tid, items in r.ranges.items():
        stage = r.stage_of_tid(tid)
        for item in items:
            rid = item.rid()
            if rid is None:
                continue
            points[f"{stage}:{item.kind}"][rid].append((item.start, item.tid))
            points[f"{stage}:{item.kind}/end"][rid].append((item.end, item.tid))
    for series in points.values():
        for values in series.values():
            values.sort()
    return points


def section_c(r: Report, points):
    print(
        "\n## C. hops (ms; in order per request; consumer = the thread that ends the hop)"
    )
    pairs = []
    for name in points:
        if name.startswith("q ") and ".put:" in name:
            pairs.append(
                (name, name.replace(".put:", ".get:"), name[2:].replace(".put", ""))
            )
    for name in points:
        match = re.match(r"io\.(send|stream) (\S+->\S+)$", name)
        if match:
            receive = "recv" if match.group(1) == "send" else "chunk"
            pairs.append(
                (
                    name,
                    f"io.{receive} {match.group(2)}",
                    f"{match.group(1)} {match.group(2)}",
                )
            )
    print(
        f"  {'hop':<58}{'n':>7}{'mean':>8}{'p50':>8}{'p95':>8}{'max':>8}  consumer busy in"
    )
    for a, b, title in sorted(pairs, key=lambda p: p[2]):
        gaps, busy = [], collections.Counter()
        for rid, starts in points.get(a, {}).items():
            ends = points.get(b, {}).get(rid, [])
            for k in range(min(len(starts), len(ends))):
                (ta, _), (tb, tid) = starts[k], ends[k]
                if tb < ta:
                    continue
                gaps.append(tb - ta)
                if len(gaps) <= 3000 and tid in r.top_level:
                    tops, top_starts = r.top_level[tid]
                    index = max(bisect.bisect_right(top_starts, ta) - 1, 0)
                    covered = 0
                    while index < len(tops) and tops[index].start < tb:
                        item = tops[index]
                        if item.end > ta:
                            part = min(tb, item.end) - max(ta, item.start)
                            busy[item.kind] += part
                            covered += part
                        index += 1
                    busy["(no range)"] += (tb - ta) - covered
        if not gaps or len(gaps) < 3:
            continue
        total = sum(busy.values()) or 1
        share = " ".join(f"{k} {100 * v / total:.0f}%" for k, v in busy.most_common(3))
        print(
            f"  {title[:58]:<58}{len(gaps):>7}{ms(statistics.fmean(gaps)):>8.2f}{ms(pct(gaps, .5)):>8.2f}"
            f"{ms(pct(gaps, .95)):>8.2f}{ms(max(gaps)):>8.2f}  {share}"
        )


def section_d(r: Report, top: int):
    print("\n## D. card (ms over the window)")
    window = r.t1 - r.t0
    by_pid = collections.defaultdict(list)
    owners = collections.defaultdict(list)
    for s, e, owner, name, stream, identity, key, node, pid in r.device:
        span = (max(s, r.t0), min(e, r.t1))
        by_pid[pid].append(span)
        owners[(owner, pid)].append(span)
    unions = {pid: merged(v) for pid, v in by_pid.items()}
    everything = merged([iv for v in by_pid.values() for iv in v])
    busy = sum(b - a for a, b in everything)
    print(
        f"window {ms(window):.0f}, device busy {ms(busy):.0f} ({100 * busy / window:.1f}%)"
    )
    print(
        "  kernels = union of the process's kernel intervals (switched-out time included);"
    )
    print(
        "  resident = the part of it while its own context was resident (the card it used);"
    )
    print("  contended = kernels while another process also had a kernel in flight")
    print(
        f"  {'process':<24}{'kernels':>9}{'resident':>10}{'share':>7}{'contended':>11}"
    )
    for pid, spans in sorted(
        unions.items(), key=lambda kv: -sum(b - a for a, b in kv[1])
    ):
        own = sum(b - a for a, b in spans)
        others = merged([iv for other, v in unions.items() if other != pid for iv in v])
        contended = own - minus_len(spans, others)
        resident = r.resident_ns(spans, pid)
        print(
            f"  {r.stage_of_pid(pid)[:24]:<24}{ms(own):>9.0f}{ms(resident):>10.0f}"
            f"{100 * resident / window:>6.1f}%{ms(contended):>11.0f}"
        )
    print("  per owner (stage:component), resident device time:")
    rows = sorted(
        ((owner, r.resident_ns(spans, pid)) for (owner, pid), spans in owners.items()),
        key=lambda row: -row[1],
    )
    for owner, value in rows[:top]:
        print(f"    {owner:<40}{ms(value):>9.0f}")
    if r.has_residency and r.context_rows:
        print("  context residency (--gpuctxsw), session GPU:")
        print(
            f"    {'process':<24}{'host pid':>9}{'slices':>8}{'resident':>10}{'share':>7}"
            f"{'slice p50':>10}{'p90':>8}{'max':>8}"
        )
        for key, host_pid, pid, slices in sorted(
            r.context_rows, key=lambda row: -sum(b - a for a, b in row[3])
        ):
            name = (
                r.stage_of_pid(pid) if pid is not None else f"ctx{key[1]} (no starts)"
            )
            total = sum(b - a for a, b in slices)
            lengths = [b - a for a, b in slices]
            print(
                f"    {name[:24]:<24}{host_pid:>9}{len(slices):>8}{ms(total):>10.0f}"
                f"{100 * total / window:>6.1f}%{ms(pct(lengths, .5)):>10.3f}"
                f"{ms(pct(lengths, .9)):>8.3f}{ms(max(lengths)):>8.3f}"
            )
        if r.switch_gaps:
            print(
                f"    switch gap (save end to next restore start): n {len(r.switch_gaps)}, "
                f"p50 {pct(r.switch_gaps, .5) / 1e3:.1f} us, p90 {pct(r.switch_gaps, .9) / 1e3:.1f} us"
            )
    else:
        print("  no context residency (profile with --gpuctxsw=true)")
    other_by_pid = {
        pid: merged([iv for other, v in unions.items() if other != pid for iv in v])
        for pid in unions
    }
    starts_by_pid = {pid: [a for a, _ in v] for pid, v in other_by_pid.items()}
    alone, contended, weight = (
        collections.defaultdict(list),
        collections.defaultdict(list),
        collections.Counter(),
    )
    for s, e, owner, name, stream, identity, key, node, pid in r.device:
        duration = e - s
        if duration <= 0:
            continue
        overlap = sum(
            b - a for a, b in clip(other_by_pid[pid], starts_by_pid[pid], s, e)
        )
        entry = (owner, identity)
        weight[entry] += duration
        if overlap < 0.05 * duration:
            alone[entry].append(duration)
        elif overlap > 0.5 * duration:
            contended[entry].append(duration)
    stretch_rows = []
    for entry, total in weight.most_common(5000):
        a, b = alone.get(entry, []), contended.get(entry, [])
        if len(a) >= 10 and len(b) >= 10:
            stretch_rows.append(
                (entry, statistics.median(a), statistics.median(b), len(a), len(b))
            )
    cost = collections.Counter()
    for (owner, identity), ma, mb, na, nb in stretch_rows:
        cost[owner.split(":")[0]] += (mb - ma) * nb
    print(
        "  kernel duration alone against with another process's kernel in flight "
        f"({len(stretch_rows)} identities with 10+ each; the excess is time switched out):"
    )
    for owner, value in cost.most_common():
        print(f"    {owner:<24} excess over alone {ms(value):.0f} ms")
    # card wait: a kernel ready to run (its launch call returned and the previous kernel or
    # copy of its stream ended) that has not started while another process's context was
    # resident; unioned per owner so waits on parallel streams count once
    others_resident = {}
    if r.has_residency and r.resident:
        for pid in unions:
            others_resident[pid] = merged(
                [iv for other, v in r.resident.items() if other != pid for iv in v]
            )
    completed = max(len(r.completed_requests()), 1)
    if not others_resident:
        print("  card wait needs context residency")
    else:
        other_starts = {pid: [a for a, _ in v] for pid, v in others_resident.items()}
        by_stream = collections.defaultdict(list)
        for s, e, owner, name, stream, identity, key, node, pid in r.device:
            by_stream[(pid, stream)].append((s, e, owner, key))
        waits = collections.defaultdict(list)
        for (pid, stream), items in by_stream.items():
            copies = sorted(r.copies_by_stream.get((pid, stream), []))
            copy_ends = [e for _, e in copies]
            copy_starts = [s for s, _ in copies]
            items.sort()
            previous_end = None
            for s, e, owner, key in items:
                ready = r.api_end.get(key)
                if ready is not None:
                    if previous_end is not None:
                        ready = max(ready, previous_end)
                    index = bisect.bisect_left(copy_starts, s) - 1
                    if index >= 0:
                        ready = max(ready, min(copy_ends[index], s))
                    if s > ready:
                        waits[owner] += clip(
                            others_resident[pid], other_starts[pid], ready, s
                        )
                previous_end = e if previous_end is None else max(previous_end, e)
        print(
            "  card wait per owner: ready to run but not started while another process's "
            "context was resident (ms total, ms per completed request):"
        )
        for owner, spans in sorted(waits.items(), key=lambda kv: -union_ns(kv[1]))[
            :top
        ]:
            value = union_ns(spans)
            print(f"    {owner:<40}{ms(value):>9.0f}{ms(value) / completed:>9.2f}")
    stretch_rows.sort(key=lambda row: -(row[2] - row[1]) * row[4])
    for (owner, identity), ma, mb, na, nb in stretch_rows[:top]:
        print(
            f"    {owner[:26]:<26}{str(identity[0])[:56]:<58} alone {ma / 1e3:7.1f} us x{na:<6} "
            f"contended {mb / 1e3:7.1f} us x{nb:<6} +{100 * (mb - ma) / ma:5.1f}%"
        )


def section_e(r: Report):
    print("\n## E. GIL (ms over the window)")
    if not r.gil_hold:
        print("  no python-gil trace in this report")
        return
    window = r.t1 - r.t0
    hold_lists = {}
    for tid, items in r.gil_hold.items():
        items.sort()
        hold_lists[tid] = (items, [s for s, _ in items])
    grouped = collections.defaultdict(lambda: [0, 0, 0])
    for tid in set(r.gil_hold) | set(r.gil_wait):
        name = r.thread_label(tid)
        grouped[name][0] += union_ns(r.gil_hold.get(tid, []))
        grouped[name][1] += union_ns(r.gil_wait.get(tid, []))
        grouped[name][2] += 1
    print(f"  {'process/thread':<52}{'n':>4}{'hold':>9}{'hold %':>8}{'wait':>9}")
    for name, (hold, wait, count) in sorted(grouped.items(), key=lambda kv: -kv[1][0])[
        :30
    ]:
        print(
            f"  {name[:52]:<52}{count:>4}{ms(hold):>9.0f}{100 * hold / window:>7.1f}%{ms(wait):>9.0f}"
        )
    matrix = collections.Counter()
    for tid, waits in r.gil_wait.items():
        for s, e in waits:
            for other, (items, starts) in hold_lists.items():
                if other == tid or pid_of(other) != pid_of(tid):
                    continue
                for a, b in clip(items, starts, s, e):
                    matrix[(r.thread_label(tid), r.thread_label(other))] += b - a
    print("  waiter <- holder, same process (ms):")
    for (waiter, holder), value in matrix.most_common(15):
        print(f"    {waiter[:40]:<40} <- {holder[:40]:<40} {ms(value):8.0f}")


def section_f(r: Report, points) -> dict[str, list[int]]:
    print(
        "\n## F. request path (ms; first occurrence per request, points in observed order)"
    )
    by_rid: dict[str, dict[str, int]] = collections.defaultdict(dict)
    for name, series in points.items():
        for rid, values in series.items():
            by_rid[rid][name] = values[0][0]
    offsets = collections.defaultdict(list)
    for rid, named in by_rid.items():
        origin = named.get("coord.submit")
        if origin is None:
            continue
        for name, t in named.items():
            offsets[name].append(t - origin)
    counted = {rid for rid, named in by_rid.items() if "coord.submit" in named}
    common = [
        name
        for name, values in offsets.items()
        if len(values) >= 0.8 * max(len(counted), 1)
    ]
    order = sorted(common, key=lambda name: statistics.median(offsets[name]))
    segments = collections.defaultdict(list)
    for rid in counted:
        named = by_rid[rid]
        known = [(name, named[name]) for name in order if name in named]
        for (name_a, ta), (name_b, tb) in zip(known, known[1:]):
            segments[(name_a, name_b)].append(tb - ta)
    print(f"  requests with a coordinator submit: {len(counted)}")
    print(
        f"  {'point':<48}{'offset p50':>11}{'p90':>9}   segment to next: {'p50':>7}{'p90':>8}"
    )
    for index, name in enumerate(order):
        segment = ""
        if index + 1 < len(order):
            values = segments.get((name, order[index + 1]), [])
            if values:
                segment = f"{ms(pct(values, .5)):>7.2f}{ms(pct(values, .9)):>8.2f}"
        print(
            f"  {name[:48]:<48}{ms(pct(offsets[name], .5)):>11.2f}{ms(pct(offsets[name], .9)):>9.2f}   {'':>17}{segment}"
        )
    chain = (
        "coord.submit",
        "q thinker.in.get:new_request",
        "thinker:sched.prefill",
        "q thinker.out.put:stream",
        "coord.recv decode",
        "q talker_ar.in.put:stream_done",
        "talker_ar:tk.build",
        "talker_ar:tk.build/end",
        "talker_ar:sched.prefill",
        "q talker_ar.out.put:stream",
        "q code2wav.in.get:stream_chunk",
        "code2wav:c2w.decode",
        "code2wav:c2w.decode/end",
        "coord.recv code2wav",
    )
    fixed = collections.defaultdict(list)
    totals = []
    for rid in counted:
        named = by_rid[rid]
        known = [(name, named[name]) for name in chain if name in named]
        for (name_a, ta), (name_b, tb) in zip(known, known[1:]):
            fixed[(name_a, name_b)].append(tb - ta)
        if "coord.recv code2wav" in named:
            totals.append(named["coord.recv code2wav"] - named["coord.submit"])
    print("  first audio chain (first occurrence of each point per request; ms):")
    print(f"    {'segment':<72}{'n':>5}{'mean':>9}{'p50':>9}{'p90':>9}")
    for (name_a, name_b), values in fixed.items():
        print(
            f"    {(name_a + ' -> ' + name_b)[:72]:<72}{len(values):>5}"
            f"{ms(statistics.fmean(values)):>9.2f}{ms(pct(values, .5)):>9.2f}{ms(pct(values, .9)):>9.2f}"
        )
    if totals:
        print(
            f"    {'TOTAL coord.submit -> coord.recv code2wav':<72}{len(totals):>5}"
            f"{ms(statistics.fmean(totals)):>9.2f}{ms(pct(totals, .5)):>9.2f}{ms(pct(totals, .9)):>9.2f}"
        )
    return {f"{a} -> {b}": v for (a, b), v in segments.items()}


def section_h(r: Report, sample_rate: int):
    print(
        "\n## H. playback margin at the coordinator (ms; audio delivered minus time since the first chunk)"
    )
    arrivals = collections.defaultdict(list)
    for t, tid, label in r.marks:
        match = re.match(r"coord\.recv rid=(\S+) from=\S+ n=(\d+)", label)
        if match:
            arrivals[match.group(1)].append((t, int(match.group(2))))
    if not arrivals:
        print("  no audio chunks at the coordinator in this report")
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
        f"  requests {len(arrivals)}, chunks after the first {len(margins)}, late {late} "
        f"({100 * late / max(len(margins), 1):.2f}%)"
    )
    if margins:
        print(
            f"  margin p5 {ms(pct(margins, .05)):.1f}  p50 {ms(pct(margins, .5)):.1f}  min {ms(min(margins)):.1f}"
        )
        print(
            f"  inter chunk gap p50 {ms(pct(gaps, .5)):.1f}  p95 {ms(pct(gaps, .95)):.1f}; stall per request "
            f"mean {ms(statistics.fmean(stalls)):.2f}  max {ms(max(stalls)):.1f}"
        )


def section_g(r: Report, top: int):
    print("\n## G. attribution: resident device time by capture site and op call site")
    node_rows = (
        r.db.execute(
            "select start, globalTid, graphNodeId, originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS"
        ).fetchall()
        if "CUDA_GRAPH_NODE_EVENTS" in r.tables
        else []
    )
    original, created = {}, {}
    for start, tid, node, orig in node_rows:
        pid = pid_of(tid)
        if orig is not None:
            original[(pid, node)] = orig
        elif (pid, node) not in created:
            created[(pid, node)] = (start, tid)

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

    by_site, by_omni, by_cap = (
        collections.Counter(),
        collections.Counter(),
        collections.Counter(),
    )
    labeled = unlabeled = 0
    owner_total = collections.Counter()
    charged = r.charged_durations()
    for index, (start, end, owner, name, stream, identity, key, node, pid) in enumerate(
        r.device
    ):
        duration = charged[index]
        stage = owner.split(":")[0]
        owner_total[stage] += duration
        cap = op = None
        if node is not None:
            orig = node
            while (pid, orig) in original:
                orig = original[(pid, orig)]
            origin = created.get((pid, orig))
            if origin is not None:
                cap, op = capture_labels(*origin)
        else:
            tid = r.api_tid.get(key)
            if tid is not None:
                item = r.innermost(r.capture, r.capture_starts, tid, r.api_start[key])
                if item is not None and item.kind == "op":
                    op = item.label
        if op is not None:
            labeled += duration
            match = re.match(r"op (\S+) @(\S+) <(\S+)>", op)
            if match:
                by_site[(stage, match.group(1), match.group(2), name[:48])] += duration
                by_omni[(stage, match.group(3))] += duration
        else:
            unlabeled += duration
        if cap is not None:
            by_cap[(stage, cap[4:120])] += duration
    total = labeled + unlabeled
    print(
        f"  device time with an op label {ms(labeled):.0f} ms of {ms(total):.0f} ({100 * labeled / max(total, 1):.1f}%)"
    )
    print("  by capture site:")
    for (stage, cap), value in by_cap.most_common(top):
        print(
            f"    {stage[:18]:<18}{ms(value):>9.0f} ms {100 * value / max(owner_total[stage], 1):5.1f}%  {cap}"
        )
    print("  by omni line (innermost sglang_omni frame of the op):")
    for (stage, site), value in by_omni.most_common(top):
        print(
            f"    {stage[:18]:<18}{ms(value):>9.0f} ms {100 * value / max(owner_total[stage], 1):5.1f}%  {site}"
        )
    print("  by op, library line and kernel:")
    for (stage, op, site, name), value in by_site.most_common(top):
        print(
            f"    {stage[:18]:<18}{ms(value):>9.0f} ms {100 * value / max(owner_total[stage], 1):5.1f}%  "
            f"{op:<22} {site:<58} {name}"
        )


def section_i(r: Report):
    print(
        "\n## I. engine steps (per forward range on each engine's scheduler thread; ms)"
    )
    steps = collections.defaultdict(list)
    for tid, items in r.ranges.items():
        stage = r.stage_of_tid(tid)
        for item in items:
            if (
                item.kind.split(" ")[0] in ("sched.batch", "sched.launch")
                and r.t0 <= item.start
                and item.end <= r.t1
            ):
                steps[(stage, item.kind)].append(item)
    print(
        f"  {'stage':<14}{'kind':<22}{'steps':>7}{'rows p50':>9}{'mean':>7}{'toks p50':>9}{'wall':>8}"
        f"{'dev':>7}{'kern':>6}{'graph':>6}{'api':>7}{'gil':>6}{'py':>7}{'period p50':>11}"
    )
    for (stage, kind), items in sorted(steps.items()):
        rows = [int(re.search(r"bs=(\d+)", x.label).group(1)) for x in items]
        tokens = [
            int(m.group(1)) for x in items if (m := re.search(r"toks=(\d+)", x.label))
        ]
        sample = items if len(items) <= 2000 else items[:: len(items) // 2000 + 1]
        api = gil = os_ns = 0
        for item in sample:
            a, g, o, _ = r.host_split(item.tid, item.start, item.end)
            api += a
            gil += g
            os_ns += o
        m = len(sample)
        wall = statistics.fmean(x.wall for x in items)
        starts = sorted(x.start for x in items)
        periods = [b - a for a, b in zip(starts, starts[1:])]
        print(
            f"  {stage[:14]:<14}{kind[:22]:<22}{len(items):>7}{pct(rows, .5):>9}{statistics.fmean(rows):>7.2f}"
            f"{pct(tokens, .5) if tokens else '-':>9}{ms(wall):>8.2f}"
            f"{ms(statistics.fmean(r.resident_ns(x.kernels, pid_of(x.tid)) for x in items)):>7.2f}"
            f"{statistics.fmean(len(x.kernels) for x in items):>6.0f}"
            f"{statistics.fmean(x.graph_launches for x in items):>6.1f}{ms(api / m):>7.2f}{ms(gil / m):>6.2f}"
            f"{ms(wall - (api + gil + os_ns) / m):>7.2f}{ms(pct(periods, .5)):>11.2f}"
        )
    completed = len(r.completed_requests())
    if completed:
        print(f"  per completed request ({completed}):")
        for (stage, kind), items in sorted(steps.items()):
            print(
                f"    {stage[:14]:<14}{kind[:22]:<22}{len(items) / completed:>8.1f} steps, "
                f"{ms(sum(r.resident_ns(x.kernels, pid_of(x.tid)) for x in items)) / completed:>8.1f} ms device, "
                f"{ms(sum(x.wall for x in items)) / completed:>8.1f} ms wall"
            )


def section_j(r: Report, segments: dict[str, list[int]]):
    print("\n## J. table: per completed request")
    completed = len(r.completed_requests())
    if not completed:
        print("  no completed requests in the window")
        return
    owners = collections.defaultdict(list)
    for s, e, owner, name, stream, identity, key, node, pid in r.device:
        owners[(owner, pid)].append((max(s, r.t0), min(e, r.t1)))
    values = {
        owner: r.resident_ns(spans, pid) for (owner, pid), spans in owners.items()
    }
    total = sum(values.values())
    print(
        f"  resident device time per request by owner (card throughput view), {completed} requests:"
    )
    for owner, value in sorted(values.items(), key=lambda kv: -kv[1])[:20]:
        print(
            f"    {owner:<40}{ms(value) / completed:>9.2f} ms {100 * value / max(total, 1):>6.1f}%"
        )
    print("  request path segments ranked by median (latency view):")
    ranked = sorted(segments.items(), key=lambda kv: -pct(kv[1], 0.5))
    for name, values in ranked[:20]:
        print(
            f"    {name[:90]:<90}{ms(pct(values, .5)):>9.2f} p50 {ms(pct(values, .9)):>9.2f} p90"
        )


def load_dcgm(r: Report, samples_path: str, gpu: int, lag: int, period: int):
    """DCGM values of one card keyed by the start of the window each covers, in trace time."""
    session_ns = r.db.execute(
        "select utcEpochNs from TARGET_INFO_SESSION_START_TIME"
    ).fetchone()[0]
    values = collections.defaultdict(dict)
    with open(samples_path) as handle:
        handle.readline()
        for line in handle:
            ts, card, name, value = line.rstrip("\n").split("\t")
            end = int(ts) * 1000 - lag - session_ns
            if int(card) == gpu and end - period >= r.t0 and end <= r.t1:
                values[end - period][name] = float(value)
    return values


def section_k(
    r: Report, samples_path: str | None, gpu: int, lag_ms: float, period_ms: float
):
    print("\n## K. card activity: where the window went, and DCGM GR and SM active")
    period = int(period_ms * 1e6)
    owner_spans = collections.defaultdict(list)
    for row, spans in zip(r.device, r.charged_spans()):
        owner_spans[r.stage_of_pid(row[-1])].extend(spans)
    indexed = {}
    for owner, spans in owner_spans.items():
        spans = merged(spans)
        indexed[owner] = (spans, [a for a, _ in spans])
    owners = sorted(indexed)
    card = merged([span for spans, _ in indexed.values() for span in spans])
    card_starts = [a for a, _ in card]

    def busy(spans, starts, start) -> float:
        return (
            sum(b - a for a, b in clip(spans, starts, start, start + period)) / period
        )

    window = r.t1 - r.t0
    resident = merged([span for spans in r.resident.values() for span in spans])
    gaps, cursor = [], r.t0
    for a, b in resident:
        if a > cursor:
            gaps.append(a - cursor)
        cursor = max(cursor, b)
    if r.t1 > cursor:
        gaps.append(r.t1 - cursor)
    busy_ns = sum(b - a for a, b in card)
    resident_ns = sum(b - a for a, b in resident)
    switching = sum(gap for gap in gaps if gap <= SWITCH_GAP_NS)
    print("  where the window went (trace):")
    for label, value in (
        ("a kernel charged", busy_ns),
        ("a context resident, no kernel (host turn, copies)", resident_ns - busy_ns),
        (f"switching (gaps up to {SWITCH_GAP_NS // 1000} us)", switching),
        ("no context resident (no process had work)", window - resident_ns - switching),
    ):
        print(f"    {label:<52}{ms(value):9.0f} ms {100 * value / window:6.1f} %")
    if samples_path is None:
        print("  no DCGM samples (--dcgm): counters not read")
        return
    else:
        print(f"  DCGM, card {gpu}:")
    # the lag is refitted on this run, the one whose windows best match the trace's busy
    # fraction: it holds DCGM's own delay (the known answer's) and the offset between the
    # trace clock and its session start epoch
    fits = []
    for lag_step in range(-30, 31):
        lag = int((lag_ms + 10 * lag_step) * 1e6)
        values = load_dcgm(r, samples_path, gpu, lag, period)
        windows = sorted(w for w in values if "gr_active" in values[w])
        if len(windows) < 3:
            continue
        measured = numpy.array([values[w]["gr_active"] for w in windows])
        trace = numpy.array([busy(card, card_starts, w) for w in windows])
        fits.append((float(numpy.corrcoef(measured, trace)[0, 1]), lag, values))
    if not fits:
        print(f"  no DCGM sample of card {gpu} inside the window")
        return
    correlation, lag, values = max(fits, key=lambda fit: fit[0])
    # DCGM stamps each field on its own, so each field's samples are its own windows
    windows = sorted(values)
    fields = sorted({name for fields in values.values() for name in fields})
    measured = {
        name: numpy.array([values[w].get(name, numpy.nan) for w in windows])
        for name in fields
    }
    has_gr = ~numpy.isnan(measured["gr_active"])
    trace_gr = numpy.array([busy(card, card_starts, w) for w in windows])[has_gr]
    print(
        f"  card {gpu}, {has_gr.sum()} GR samples of "
        f"{period_ms:.0f} ms; lag {lag / 1e6:.0f} ms "
        f"(known answer {lag_ms:.0f}), per-window correlation of DCGM GR with the trace "
        f"{correlation:.3f}"
    )
    print(
        f"  GR active: DCGM {100 * numpy.nanmean(measured['gr_active']):.1f} %, trace "
        f"(charged kernels) {100 * trace_gr.mean():.1f} %"
    )
    print("  DCGM means over the windows:")
    for name in fields:
        print(f"    {name:<16}{100 * numpy.nanmean(measured[name]):6.1f} %")
    matrix = numpy.array([[busy(*indexed[o], w) for o in owners] for w in windows])
    print(
        "  per owner, fitted over the windows: each field = sum over owners of (the owner's "
        "charged kernel time / window) x its value while its kernels run; gr_active near 1 "
        "for every owner is the fit's own check"
    )
    coefficients, quality = {}, {}
    for name in fields:
        ok = ~numpy.isnan(measured[name])
        solution = numpy.linalg.lstsq(matrix[ok], measured[name][ok], rcond=None)[0]
        fitted = matrix[ok] @ solution
        spread = ((measured[name][ok] - measured[name][ok].mean()) ** 2).sum()
        residual = ((measured[name][ok] - fitted) ** 2).sum()
        coefficients[name] = solution
        quality[name] = 1 - residual / spread if spread > 0 else float("nan")
    print(
        f"    {'owner':<24}{'card share':>11}"
        + "".join(f"{n[:12]:>13}" for n in fields)
    )
    for index, owner in enumerate(owners):
        share = matrix[:, index].mean()
        print(
            f"    {owner:<24}{100 * share:10.1f}%"
            + "".join(f"{coefficients[n][index]:13.3f}" for n in fields)
        )
    print(f"    {'fit R2':<35}" + "".join(f"{quality[n]:13.3f}" for n in fields))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--sections", default="ABCDEFGHIJK")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--dcgm", help="dcgm_sampler.py samples for section K")
    parser.add_argument("--dcgm-gpu", type=int, default=0)
    parser.add_argument("--dcgm-lag-ms", type=float, default=122.0)
    parser.add_argument("--dcgm-period-ms", type=float, default=100.0)
    args = parser.parse_args()
    r = Report(args.report, args.window)
    print(
        f"window {r.t0 / 1e9:.3f} .. {r.t1 / 1e9:.3f} s ({(r.t1 - r.t0) / 1e9:.1f} s)"
    )
    points = request_points(r) if set("CFJ") & set(args.sections) else {}
    segments: dict[str, list[int]] = {}
    for key in args.sections:
        if key == "A":
            section_a(r)
        elif key == "B":
            section_b(r, args.top)
        elif key == "C":
            section_c(r, points)
        elif key == "D":
            section_d(r, args.top)
        elif key == "E":
            section_e(r)
        elif key == "F":
            segments = section_f(r, points)
        elif key == "G":
            section_g(r, args.top)
        elif key == "H":
            section_h(r, args.sample_rate)
        elif key == "I":
            section_i(r)
        elif key == "J":
            section_j(r, segments)
        elif key == "K":
            section_k(
                r, args.dcgm, args.dcgm_gpu, args.dcgm_lag_ms, args.dcgm_period_ms
            )
        else:
            print(f"\n## {key}: unknown section")


if __name__ == "__main__":
    main()
