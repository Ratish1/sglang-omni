"""Every request's time split into its states, from one nsys report taken with cosy_profile_probe, and the mean number
of requests in each state over the window.

States, in order (each from the marks the probe records):
  preprocessing   preprocessing.in put -> tts_engine.in get (queue and reference encoding)
  ar build        tts_engine.in get -> sched.admit (the request builder)
  ar slot wait    sched.admit -> the request's first sched.prefill (waiting for a running slot of the AR)
  ar first flush  first sched.prefill -> first mr.emit (prefill and the first 28 tokens)
  vocoder first   first mr.emit -> first vocoder.out stream put (hand off, wait, the first hop's step, out)
  = first audio on the server
  ar rest         first audio -> vocoder.in put of stream_done (the AR still generating while audio streams)
  vocoder tail    stream_done put -> vocoder.out result put (the final and the result)

usage: python3 request_states_c20261019.py NAME=REPORT.sqlite [NAME=REPORT.sqlite ...] [--skip-head-s S] [--skip-tail-s S]
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import statistics
from collections import defaultdict

QUEUE = re.compile(r"q (\S+)\.(put|get) rid=(\S+) t=(\S+)")
ADMIT = re.compile(r"sched\.admit rid=(\S+)")
PREFILL = re.compile(r"sched\.prefill rid=(\S+)")
EMIT = re.compile(r"mr\.emit rid=(\S+)")
STATES = ("preprocessing", "ar build", "ar slot wait", "ar first flush", "vocoder first", "ar rest", "vocoder tail")
MS = 1e-6


def load(path: str) -> list[tuple[int, str]]:
    db = sqlite3.connect(path)
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    if "textId" in columns:
        query = (
            "SELECT e.start, COALESCE(e.text, s.value) FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId = s.id"
        )
    else:
        query = "SELECT start, text FROM NVTX_EVENTS"
    return sorted((row for row in db.execute(query) if row[1]), key=lambda row: row[0])


def analyse(name: str, path: str, skip_head_s: float, skip_tail_s: float) -> None:
    marks: dict[str, dict[str, int]] = defaultdict(dict)
    for start, text in load(path):
        if match := QUEUE.match(text):
            label, op, rid, kind = match[1], match[2], match[3], match[4]
            key = f"{label}.{op}.{kind}"
            if key in ("vocoder.out.put.stream",):
                marks[rid].setdefault("first audio", start)
            elif key in (
                "preprocessing.in.put.new_request",
                "tts_engine.in.get.new_request",
                "vocoder.in.put.stream_done",
                "vocoder.out.put.result",
            ):
                marks[rid].setdefault(key, start)
            else:
                pass
        elif match := ADMIT.match(text):
            marks[match[1]].setdefault("admit", start)
        elif match := PREFILL.match(text):
            marks[match[1]].setdefault("prefill", start)
        elif match := EMIT.match(text):
            marks[match[1]].setdefault("first emit", start)
    order = (
        "preprocessing.in.put.new_request",
        "tts_engine.in.get.new_request",
        "admit",
        "prefill",
        "first emit",
        "first audio",
        "vocoder.in.put.stream_done",
        "vocoder.out.put.result",
    )
    complete = {rid: m for rid, m in marks.items() if all(key in m for key in order)}
    if not complete:
        print(f"\n# {name}: no request carries every mark ({len(marks)} requests seen)")
        return
    t0 = min(m[order[0]] for m in complete.values()) + skip_head_s * 1e9
    t1 = max(m[order[-1]] for m in complete.values()) - skip_tail_s * 1e9
    inside = {rid: m for rid, m in complete.items() if t0 <= m[order[0]] and m[order[-1]] <= t1}
    durations: dict[str, list[float]] = defaultdict(list)
    for m in inside.values():
        for state, (start_key, end_key) in zip(STATES, zip(order, order[1:])):
            durations[state].append((m[end_key] - m[start_key]) * MS)
        durations["first audio on the server"].append((m["first audio"] - m[order[0]]) * MS)
        durations["request on the server"].append((m[order[-1]] - m[order[0]]) * MS)
    window_ms = (t1 - t0) * MS
    print(f"\n# {name}: {path}\nrequests {len(inside)} of {len(marks)}, window {window_ms / 1e3:.1f} s")
    print(f"{'state':28} {'mean ms':>9} {'p50':>9} {'p90':>9} {'requests in it':>15}")
    for state in (*STATES, "first audio on the server", "request on the server"):
        values = sorted(durations[state])
        occupancy = sum(values) / window_ms
        print(
            f"{state:28} {statistics.mean(values):9.1f} {values[len(values) // 2]:9.1f} "
            f"{values[int(0.9 * (len(values) - 1))]:9.1f} {occupancy:15.2f}"
        )
    first_audio = durations["first audio on the server"]
    slowest = sorted(range(len(first_audio)), key=first_audio.__getitem__)[-max(1, len(first_audio) // 100) :]
    print(f"the slowest 1 % by first audio ({len(slowest)} requests), mean ms per state:")
    for state in (*STATES[:5], "first audio on the server"):
        print(f"  {state:26} {statistics.mean(durations[state][index] for index in slowest):9.1f}")


parser = argparse.ArgumentParser()
parser.add_argument("reports", nargs="+")
parser.add_argument("--skip-head-s", type=float, default=2.0)
parser.add_argument("--skip-tail-s", type=float, default=2.0)
arguments = parser.parse_args()
for report in arguments.reports:
    name, _, path = report.partition("=")
    analyse(name, path, arguments.skip_head_s, arguments.skip_tail_s)
