"""Where a request's first audio goes, per request, from nsys reports taken with the probe (cosy_profile_probe) on
the AR engine and the vocoder, one report per arm, the arms traced at the same time.

Per request, from its marks and ranges (all on the server's clock):
  start    the request's first get from the AR stage's inbox (the stage whose thread runs mr.emit)
  emit     the first mr.emit rid=: the AR hands its first hop's codes on
  ingest   the first voc.ingest rid=: the vocoder takes them in
  ready    the first voc.ready rid=: the first hop can be decoded
  chosen   the voc.steprid rid= of the first step that carries it, inside a voc.step range (its plan and rows)
  out      the first stream put from the vocoder stage's outbox after that step: the first chunk leaves
and the voc.rank marks with slack=u (a stream that has not emitted yet) and chosen=0 before it is chosen: the steps
that passed a ready first hop over. Also: whether a vocoder step was running when the AR emitted (the vocoder takes
its inbox in between steps, so the hand off includes what was left of that step), and the steps that ran between
ready and the chosen step, split into hop steps and final (leftover) steps, by count and by the time they held it.

Segments: AR (emit - start), hand off (ingest - emit), to ready (ready - ingest), wait for a step (step start - ready),
step (its wall time), out (out - step end). Printed for every request, for the slowest 5 % by server first audio
(out - start), with the steps that carried first hops (plan, rows, wall) and every vocoder step by plan.

usage: python3 first_audio_timeline_c20261017.py NAME=REPORT.sqlite [NAME=REPORT.sqlite ...] [--skip-head-s S]
       [--skip-tail-s S]"""

import argparse
import bisect
import re
import sqlite3
import statistics
from collections import Counter, defaultdict

EMIT = re.compile(r"mr\.emit rid=(\S+)")
INGEST = re.compile(r"voc\.ingest rid=(\S+)")
READY = re.compile(r"voc\.ready rid=(\S+) plan=(\S+)")
RANK = re.compile(r"voc\.rank rid=(\S+) plan=\S+ slack=(\S+) chosen=(\d)")
STEPRID = re.compile(r"voc\.steprid rid=(\S+) plan=(\S+)")
STEP = re.compile(r"voc\.step (\S+) rows=(\d+)")
QUEUE = re.compile(r"q (\S+)\.(put|get) rid=(\S+) t=(\S+)")


def load(path):
    db = sqlite3.connect(path)
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    if "textId" in columns:
        query = ("SELECT e.start, e.end, e.globalTid, COALESCE(e.text, s.value) FROM NVTX_EVENTS e "
                 "LEFT JOIN StringIds s ON e.textId = s.id")
    else:
        query = "SELECT start, end, globalTid, text FROM NVTX_EVENTS"
    return sorted((row for row in db.execute(query) if row[3]), key=lambda row: row[0])


def first(table, rid, value):
    if rid not in table:
        table[rid] = value


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def summary(values):
    if not values:
        return "n 0"
    return (f"n {len(values)} mean {statistics.mean(values):7.1f} p50 {quantile(values, 0.5):7.1f} "
            f"p90 {quantile(values, 0.9):7.1f} p99 {quantile(values, 0.99):7.1f}")


def analyse(name, path, skip_head_s, skip_tail_s):
    rows = load(path)
    emit, ingest, ready, chosen, passes = {}, {}, {}, {}, Counter()
    steps, queue_ops, emit_tids = [], [], set()
    for start, end, tid, text in rows:
        if match := EMIT.match(text):
            first(emit, match[1], start)
            emit_tids.add(tid)
        elif match := INGEST.match(text):
            first(ingest, match[1], start)
        elif match := READY.match(text):
            first(ready, match[1], (start, match[2]))
        elif match := RANK.match(text):
            if match[1] not in chosen and match[2] == "u" and match[3] == "0":
                passes[match[1]] += 1
        elif match := STEPRID.match(text):
            first(chosen, match[1], (start, match[2]))
        elif (match := STEP.match(text)) and end is not None:
            steps.append((start, end, tid, match[1], int(match[2])))
        elif match := QUEUE.match(text):
            queue_ops.append((start, tid, match[1], match[2], match[3], match[4]))
    steps.sort()
    vocoder_tids = {step[2] for step in steps}
    labels = Counter((label, op, kind, "ar" if tid in emit_tids else "voc" if tid in vocoder_tids else "other")
                     for _, tid, label, op, _, kind in queue_ops)
    out_labels = {label for (label, op, kind, side), _ in labels.items() if op == "put" and kind == "stream" and side == "voc"}
    ar_inbox = {label for (label, op, kind, side), _ in labels.items() if op == "get" and side == "ar"}
    start_of, outs = {}, defaultdict(list)
    for time, tid, label, op, rid, kind in queue_ops:
        if op == "get" and label in ar_inbox:
            first(start_of, rid, time)
        elif op == "put" and label in out_labels:
            outs[rid].append(time)
    t0, t1 = steps[0][0] + skip_head_s * 1e9, steps[-1][1] - skip_tail_s * 1e9
    step_starts = [step[0] for step in steps]
    records = []
    for rid, (chosen_t, plan) in chosen.items():
        if rid not in emit or rid not in start_of or not (t0 <= start_of[rid] <= t1):
            continue
        index = bisect.bisect_right(step_starts, chosen_t) - 1
        if index < 0 or not steps[index][0] <= chosen_t <= steps[index][1]:
            continue
        step = steps[index]
        later = [time for time in outs[rid] if time >= step[1]]
        if not later or rid not in ingest or rid not in ready:
            continue
        ms = 1e-6
        running = bisect.bisect_right(step_starts, emit[rid]) - 1
        behind = [s for s in steps if ready[rid][0] <= s[1] and s[0] < step[0]]
        records.append({
            "in step at emit": int(running >= 0 and steps[running][1] > emit[rid]),
            "step left at emit": max(0.0, (steps[running][1] - emit[rid]) * ms) if running >= 0 else 0.0,
            "behind hops": sum(1 for s in behind if s[3] != "leftover"),
            "behind finals": sum(1 for s in behind if s[3] == "leftover"),
            "behind hops ms": sum(min(s[1], step[0]) - max(s[0], ready[rid][0]) for s in behind if s[3] != "leftover") * ms,
            "behind finals ms": sum(min(s[1], step[0]) - max(s[0], ready[rid][0]) for s in behind if s[3] == "leftover") * ms,
            "rid": rid, "total": (later[0] - start_of[rid]) * ms, "ar": (emit[rid] - start_of[rid]) * ms,
            "hand off": (ingest[rid] - emit[rid]) * ms, "to ready": (ready[rid][0] - ingest[rid]) * ms,
            "wait": (step[0] - ready[rid][0]) * ms, "step": (step[1] - step[0]) * ms, "out": (later[0] - step[1]) * ms,
            "passes": passes[rid], "plan": step[3], "rows": step[4],
        })
    print(f"\n# {name}: {path}")
    print("queue labels (label op type thread: count): " + ", ".join(
        f"{label} {op} {kind} {side}: {count}" for (label, op, kind, side), count in sorted(labels.items())))
    print(f"AR inbox {sorted(ar_inbox)}, vocoder out {sorted(out_labels)}, requests timed {len(records)}")
    if not records:
        return
    slowest = sorted(records, key=lambda record: record["total"])[-max(1, len(records) // 20):]
    for title, group in (("every request", records), ("slowest 5 % by server first audio", slowest)):
        print(f"\n## {title} (ms)")
        for key in ("total", "ar", "hand off", "to ready", "wait", "step", "out"):
            print(f"{key:9} {summary([record[key] for record in group])}")
        print(f"passes    {summary([record['passes'] for record in group])}")
        for key in ("in step at emit", "step left at emit", "behind hops", "behind finals", "behind hops ms",
                    "behind finals ms"):
            print(f"{key:17} {summary([record[key] for record in group])}")
        print(f"rows      {summary([record['rows'] for record in group])}")
        print("plans     " + ", ".join(f"{plan} {count}" for plan, count in Counter(r["plan"] for r in group).items()))
    window = [step for step in steps if t0 <= step[0] <= t1]
    span = (window[-1][1] - window[0][0]) * 1e-9
    print("\n## vocoder steps in the window")
    for plan in sorted({step[3] for step in window}):
        own = [step for step in window if step[3] == plan]
        print(f"{plan:14} {len(own)} steps, {len(own) / span:.1f}/s, rows {summary([s[4] for s in own])}, "
              f"wall ms {summary([(s[1] - s[0]) * 1e-6 for s in own])}")


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+")
parser.add_argument("--skip-head-s", type=float, default=1.0)
parser.add_argument("--skip-tail-s", type=float, default=1.0)
arguments = parser.parse_args()
for report in arguments.reports:
    name, _, path = report.partition("=")
    analyse(name, path, arguments.skip_head_s, arguments.skip_tail_s)
