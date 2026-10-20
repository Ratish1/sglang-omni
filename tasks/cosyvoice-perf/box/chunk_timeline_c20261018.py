"""Every streamed chunk of every request split into the AR's part and the vocoder's part, from nsys reports taken with
the probe (cosy_profile_probe), one report per arm. The first audio timeline (first_audio_timeline_c20261017.py) for
every chunk, not only the first.

Per request, its marks in order (each list is FIFO per request, so the j-th ingest takes the j-th emit's codes):
  start     the first get from the AR stage's inbox
  emit j    mr.emit rid=: the AR flushes codes, 28 tokens first, then 25 a flush, the rest when it finishes
  ingest j  voc.ingest rid=: the vocoder takes flush j in
  done      voc.done rid=: the vocoder takes the stream's end in
  step      voc.steprid rid= inside a voc.step <plan> range: a step that carries the stream
  out k     the k-th stream put from the vocoder's outbox: chunk k leaves
Chunk k comes from the last step carrying the stream that started before out k. A causal window chunk (hop h, counted
from 1) completes at flush 1, 3, 7, 11, ... (hops of 25, 50, then 100 tokens, 3 of lookahead), a final chunk at the
last flush. The rule is checked against every voc.ready at=ingest mark, which falls inside the completing ingest.

Per chunk: ar (emit of its completing flush minus that of the chunk before; for chunk 1, minus start), hand off
(ingest minus emit), wait (step start minus the latest of the ingest, the stream's end for a final, and the end of the
stream's previous step), step (its wall time), out (out minus step end), and vocoder (out minus emit). So gap k =
ar k + vocoder k - vocoder k-1, printed as the vocoder change.

usage: python3 chunk_timeline_c20261018.py NAME=REPORT.sqlite [NAME=REPORT.sqlite ...] [--skip-head-s S]
       [--skip-tail-s S]"""

import argparse
import bisect
import re
import sqlite3
import statistics
from collections import defaultdict

EMIT = re.compile(r"mr\.emit rid=(\S+)")
INGEST = re.compile(r"voc\.ingest rid=(\S+)")
DONE = re.compile(r"voc\.done rid=(\S+)")
READY = re.compile(r"voc\.ready rid=(\S+) plan=\S+ tokens=\d+ at=ingest")
STEPRID = re.compile(r"voc\.steprid rid=(\S+) plan=(\S+)")
STEP = re.compile(r"voc\.step (\S+) rows=(\d+)")
QUEUE = re.compile(r"q (\S+)\.(put|get) rid=(\S+) t=(\S+)")
SEGMENTS = ("gap", "ar", "vocoder", "vocoder change", "hand off", "wait", "step", "out")
MS = 1e-6


def load(path):
    db = sqlite3.connect(path)
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    if "textId" in columns:
        query = ("SELECT e.start, e.end, e.globalTid, COALESCE(e.text, s.value) FROM NVTX_EVENTS e "
                 "LEFT JOIN StringIds s ON e.textId = s.id")
    else:
        query = "SELECT start, end, globalTid, text FROM NVTX_EVENTS"
    return sorted((row for row in db.execute(query) if row[3]), key=lambda row: row[0])


def completing_flush(hop):
    return 1 if hop == 1 else 3 + 4 * (hop - 2)


HOP_FLUSHES = {completing_flush(hop) for hop in range(1, 64)}


def summary(values):
    ordered = sorted(values)
    return (f"n {len(values):4d} mean {statistics.mean(values):7.1f} p50 {ordered[len(ordered) // 2]:7.1f} "
            f"p90 {ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]:7.1f}")


def analyse(name, path, skip_head_s, skip_tail_s):
    emits, ingests, done, readies, steprids = (defaultdict(list), defaultdict(list), {}, defaultdict(list),
                                               defaultdict(list))
    steps, queue_ops, emit_tids = [], [], set()
    for start, end, tid, text in load(path):
        if match := EMIT.match(text):
            emits[match[1]].append(start)
            emit_tids.add(tid)
        elif (match := INGEST.match(text)) and end is not None:
            ingests[match[1]].append((start, end))
        elif match := DONE.match(text):
            done.setdefault(match[1], start)
        elif match := READY.match(text):
            readies[match[1]].append(start)
        elif match := STEPRID.match(text):
            steprids[match[1]].append((start, match[2]))
        elif (match := STEP.match(text)) and end is not None:
            steps.append((start, end, match[1], int(match[2])))
        elif match := QUEUE.match(text):
            queue_ops.append((start, tid, match[1], match[2], match[3], match[4]))
    steps.sort()
    step_starts = [step[0] for step in steps]
    ar_inbox = {label for _, tid, label, op, _, _ in queue_ops if op == "get" and tid in emit_tids}
    vocoder_out = {label for _, tid, label, op, _, kind in queue_ops
                   if op == "put" and kind == "stream" and tid not in emit_tids and label.startswith("vocoder")}
    start_of, outs = {}, defaultdict(list)
    for time, _, label, op, rid, kind in queue_ops:
        if op == "get" and label in ar_inbox:
            start_of.setdefault(rid, time)
        elif op == "put" and label in vocoder_out and kind == "stream":
            outs[rid].append(time)
    t0, t1 = steps[0][0] + skip_head_s * 1e9, steps[-1][1] - skip_tail_s * 1e9
    rule_checks = rule_agrees = skipped = 0
    by_chunk = defaultdict(lambda: defaultdict(list))
    for rid, chunk_outs in outs.items():
        if rid not in start_of or not t0 <= start_of[rid] <= t1 or len(ingests[rid]) != len(emits[rid]):
            skipped += 1
            continue
        ingest_starts = [start for start, _ in ingests[rid]]
        for ready in readies[rid]:
            flush = bisect.bisect_right(ingest_starts, ready)
            rule_checks += 1
            rule_agrees += flush in HOP_FLUSHES or flush == len(emits[rid])
        carried = sorted(steprids[rid])
        previous_end, previous_emit, previous_vocoder, previous_out, hop = start_of[rid], start_of[rid], None, None, 0
        for k, out in enumerate(chunk_outs, start=1):
            before = [entry for entry in carried if entry[0] < out]
            if not before:
                break
            chosen_t, plan = before[-1]
            index = bisect.bisect_right(step_starts, chosen_t) - 1
            if index < 0 or not steps[index][0] <= chosen_t <= steps[index][1]:
                break
            step = steps[index]
            if plan == "causal_window":
                hop += 1
                flush = completing_flush(hop)
                ready_floor = 0
            else:
                flush = len(emits[rid])
                ready_floor = done.get(rid, 0)
            if flush > len(emits[rid]):
                break
            emit, (ingest_start, ingest_end) = emits[rid][flush - 1], ingests[rid][flush - 1]
            record = by_chunk[(k, "hop" if plan == "causal_window" else "final")]
            vocoder = (out - emit) * MS
            record["ar"].append((emit - previous_emit) * MS)
            record["vocoder"].append(vocoder)
            record["hand off"].append((ingest_start - emit) * MS)
            record["wait"].append(max(0.0, (step[0] - max(ingest_end, previous_end, ready_floor)) * MS))
            record["step"].append((step[1] - step[0]) * MS)
            record["out"].append((out - step[1]) * MS)
            if previous_out is not None:
                record["gap"].append((out - previous_out) * MS)
                record["vocoder change"].append(vocoder - previous_vocoder)
            previous_end, previous_emit, previous_vocoder, previous_out = step[1], emit, vocoder, out
    print(f"\n# {name}: {path}")
    print(f"AR inbox {sorted(ar_inbox)}, vocoder out {sorted(vocoder_out)}, requests {len(outs)}, skipped {skipped}; "
          f"the flush rule against the at=ingest ready marks: {rule_agrees} of {rule_checks} agree")
    for (k, kind), record in sorted(by_chunk.items()):
        if len(record["ar"]) < 10:
            continue
        print(f"\n## chunk {k} ({kind}), ms")
        for key in SEGMENTS:
            if record[key]:
                print(f"{key:14} {summary(record[key])}")


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+")
parser.add_argument("--skip-head-s", type=float, default=1.0)
parser.add_argument("--skip-tail-s", type=float, default=1.0)
arguments = parser.parse_args()
for report in arguments.reports:
    name, _, path = report.partition("=")
    analyse(name, path, arguments.skip_head_s, arguments.skip_tail_s)
