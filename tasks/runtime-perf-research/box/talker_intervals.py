"""Qwen3-Omni talker decode intervals in a trace marks trace, split by what fills them.

usage: python3 talker_intervals.py TRACE.sqlite SERVE.LOG [--long-ms 150]

A decode interval runs from the start of one talker forward that advances the running rows (DECODE or MIXED) to the
start of the next. Per interval:

  rows     the first step's batch size
  ext      EXTEND steps (talker prefill) inside the interval, and their forward wall
  fwd      the decode forward's own wall
  sched    the talker scheduler thread's top-level scheduler.* spans inside the interval, by name, and inside them
           omni.sample, omni.finalize and omni.before_decode (run_batch holds these)
  talker   the talker process's device busy inside the interval (code2wav included, it shares the process)
  thinker  the other processes' device busy inside the interval
  free     the card idle inside the interval
  held     with --gpuctxsw, the card slices the talker and the others held inside the interval

Printed for every interval, the slowest 5 % and the intervals above --long-ms, as means.
"""

import argparse
import re
import sqlite3
import statistics
from collections import defaultdict

from span_table import (
    STARTUP,
    context_slices,
    covered,
    device_records,
    first_at,
    merged,
    pid_of,
    tables,
)

INNER = ("omni.sample", "omni.finalize", "omni.before_decode")
STEP = re.compile(r"omni\.forward step\[(\w+) bs=(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("serve_log")
    parser.add_argument("--long-ms", type=float, default=150.0)
    args = parser.parse_args()

    stages = {}
    with open(args.serve_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := STARTUP.search(line):
                stages[int(match[3])] = match[1]
            else:
                pass
    talker_pid = next(pid for pid, stage in stages.items() if stage == "talker_ar")

    db = sqlite3.connect(args.trace)
    names = tables(db)
    session = db.execute(
        "SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_RUNTIME"
    ).fetchone()
    device_by_pid = defaultdict(list)
    for _, gpid, start, end in device_records(db, names, session[0], session[1]):
        device_by_pid[pid_of(gpid)].append((start, end))
    busy = {pid: merged(spans) for pid, spans in device_by_pid.items()}
    talker_busy = busy.get(talker_pid, [])
    others_busy = merged(
        [span for pid, spans in busy.items() if pid != talker_pid for span in spans]
    )
    card_busy = merged([span for spans in busy.values() for span in spans])
    slices = context_slices(db, names, session[0], session[1], busy)
    talker_held = merged(
        [s for owner, _, spans in slices.values() if owner == talker_pid for s in spans]
    )
    others_held = merged(
        [s for owner, _, spans in slices.values() if owner != talker_pid for s in spans]
    )

    text_column = "coalesce(text, (SELECT value FROM StringIds WHERE id = textId))"
    steps = []
    sched = defaultdict(list)
    for start, end, text, gtid in db.execute(
        f"SELECT start, end, {text_column}, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND end > start"
    ):
        if pid_of(gtid) != talker_pid or not text:
            continue
        elif match := STEP.match(text):
            steps.append((start, end, match[1], int(match[2])))
        elif text.startswith("scheduler.") or text in INNER:
            sched[text].append((start, end))
        else:
            pass
    steps.sort()
    # note (ratish): a step opens omni.forward around the custom forward and again around the standard forward
    # when the custom one returns nothing, so back-to-back spans of one mode are one step
    joined = []
    for step in steps:
        if joined and joined[-1][2] == step[2] and step[0] - joined[-1][1] < 1_000_000:
            joined[-1] = (joined[-1][0], step[1], step[2], max(joined[-1][3], step[3]))
        else:
            joined.append(step)
    steps = joined
    sched = {name: sorted(spans) for name, spans in sched.items()}
    # note (ratish): scheduler.run_batch nests run_batch_launch and the rest, so only spans no other span holds count
    outer = {
        name
        for name in sched
        if name
        in (
            "scheduler.recv_requests",
            "scheduler.process_input_requests",
            "scheduler.get_next_batch_to_run",
            "scheduler.run_batch",
            "scheduler.process_batch_result",
            "scheduler.sleep_during_idle",
            *INNER,
        )
    }

    advancing = [step for step in steps if step[2] in ("DECODE", "MIXED")]
    intervals = []
    for (a_start, a_end, mode, rows), (b_start, *_rest) in zip(
        advancing, advancing[1:]
    ):
        if a_start < session[0] or b_start > session[1]:
            continue
        else:
            pass
        extends = [s for s in steps if a_start < s[0] < b_start and s[2] == "EXTEND"]
        record = {
            "wall": b_start - a_start,
            "rows": rows,
            "mode": mode,
            "ext_n": len(extends),
            "ext_rows": sum(s[3] for s in extends),
            "ext_wall": sum(s[1] - s[0] for s in extends),
            "fwd": a_end - a_start,
            "talker": covered(
                talker_busy, a_start, b_start, first_at(talker_busy, a_start)
            ),
            "thinker": covered(
                others_busy, a_start, b_start, first_at(others_busy, a_start)
            ),
            "free": (b_start - a_start)
            - covered(card_busy, a_start, b_start, first_at(card_busy, a_start)),
            "t_held": (
                covered(talker_held, a_start, b_start, first_at(talker_held, a_start))
                if talker_held
                else 0
            ),
            "o_held": (
                covered(others_held, a_start, b_start, first_at(others_held, a_start))
                if others_held
                else 0
            ),
        }
        for name in outer:
            spans = sched[name]
            record[name] = covered(spans, a_start, b_start, first_at(spans, a_start))
        intervals.append(record)

    window_s = (session[1] - session[0]) / 1e9
    print(
        f"talker pid {talker_pid}, window {window_s:.2f} s, steps {len(steps)} "
        f"({sum(s[2] == 'EXTEND' for s in steps)} EXTEND, {sum(s[2] == 'DECODE' for s in steps)} DECODE, "
        f"{sum(s[2] == 'MIXED' for s in steps)} MIXED), decode intervals {len(intervals)}, "
        f"context switch slices {'yes' if slices else 'no'}"
    )
    walls = sorted(r["wall"] for r in intervals)
    print(
        "interval wall ms p50 {:.1f} p90 {:.1f} p99 {:.1f} max {:.1f}; rows p50 {}".format(
            *(
                walls[min(len(walls) - 1, int(q * len(walls)))] / 1e6
                for q in (0.5, 0.9, 0.99)
            ),
            walls[-1] / 1e6,
            sorted(r["rows"] for r in intervals)[len(intervals) // 2],
        )
    )
    cut = walls[int(0.95 * len(walls))]
    groups = (
        ("every", intervals),
        ("slowest 5 %", [r for r in intervals if r["wall"] >= cut]),
        (
            f"above {args.long_ms:.0f} ms",
            [r for r in intervals if r["wall"] > args.long_ms * 1e6],
        ),
    )
    keys = [
        "wall",
        "fwd",
        "ext_wall",
        "talker",
        "thinker",
        "free",
        "t_held",
        "o_held",
        *sorted(outer),
    ]
    print(
        f"{'group':<16}{'n':>6}{'rows':>6}{'ext n':>7}{'ext rows':>9}"
        + "".join(f"{k.replace('scheduler.', ''):>15}" for k in keys)
    )
    for label, members in groups:
        if not members:
            print(f"{label:<16}{0:>6}")
            continue
        else:
            pass
        mean = lambda key: statistics.mean(r[key] for r in members)
        print(
            f"{label:<16}{len(members):>6}{mean('rows'):>6.0f}{mean('ext_n'):>7.2f}{mean('ext_rows'):>9.2f}"
            + "".join(f"{mean(k) / 1e6:>15.2f}" for k in keys)
        )
    trains = defaultdict(int)
    for r in intervals:
        trains[r["ext_n"]] += 1
    print(
        "intervals by EXTEND steps inside: "
        + ", ".join(f"{n}: {c}" for n, c in sorted(trains.items()))
    )
    for n in sorted(trains):
        members = [r for r in intervals if r["ext_n"] == n]
        print(
            f"  {n} EXTEND: wall p50 {statistics.median(r['wall'] for r in members) / 1e6:.1f} ms, "
            f"extend wall mean {statistics.mean(r['ext_wall'] for r in members) / 1e6:.1f} ms, "
            f"thinker device mean {statistics.mean(r['thinker'] for r in members) / 1e6:.1f} ms"
        )
    ext_walls = [s[1] - s[0] for s in steps if s[2] == "EXTEND"]
    if ext_walls:
        by_rows = defaultdict(list)
        for s in steps:
            if s[2] == "EXTEND":
                by_rows[s[3]].append((s[1] - s[0]) / 1e6)
            else:
                pass
        print(
            "EXTEND forward wall by batch size: "
            + ", ".join(
                f"bs {b}: n {len(v)} p50 {statistics.median(v):.1f} ms"
                for b, v in sorted(by_rows.items())
            )
        )
    else:
        pass
    by_rows = defaultdict(list)
    for s in advancing:
        by_rows[(s[2], (s[3] + 7) // 8 * 8)].append((s[1] - s[0]) / 1e6)
    print(
        "decode forward wall by rows (rounded up to 8): "
        + ", ".join(
            f"{mode} {b}: n {len(v)} p50 {statistics.median(v):.1f} ms"
            for (mode, b), v in sorted(by_rows.items())
        )
    )


if __name__ == "__main__":
    main()
