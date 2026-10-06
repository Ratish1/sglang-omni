"""Talker admissions against the streams they stall, from one probed nsys report (run on the box).

A talker prefill-only step (sched.batch extend on the talker's scheduler thread) gives no running stream a
frame for its duration. For each admission in the window this reads the running rows at that moment (the bs
of the talker's last decode step before it, launch or sync), and prints how many admissions found running
rows, the stall they put on those rows (extend wall times running rows), and the share of the window the
talker spent in admissions: the bound on what riding admissions in a decode step can return (design 05,
slice C, section 6.4).

usage: python talker_admission_bound.py REPORT.sqlite [--window window.txt]
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import statistics

from nsys_stage_ledger import pid_of, session_window

MARK = 34
RANGE = re.compile(r"^sched\.(batch|launch) (extend|decode|mixed) bs=(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    talker_pids = set()
    steps = []
    for start, end, tid, text, text_id, event_type in db.execute(
        "select start, end, globalTid, text, textId, eventType from NVTX_EVENTS order by start"
    ):
        label = text if text is not None else strings.get(text_id, "")
        if not label:
            continue
        if event_type == MARK and label == "proc stage=talker_ar":
            talker_pids.add(pid_of(tid))
        match = RANGE.match(label)
        if match and end is not None:
            steps.append((start, end, pid_of(tid), match.group(2), int(match.group(3))))
    steps = [s for s in steps if s[2] in talker_pids and t0 <= s[0] <= t1]
    window_ms = (t1 - t0) / 1e6
    running = 0
    admissions = []
    for start, end, _, kind, rows in steps:
        if kind == "decode":
            running = rows
        else:
            admissions.append(((end - start) / 1e6, rows, running))
    with_running = [a for a in admissions if a[2] > 0]
    stall_ms = sum(wall * running_rows for wall, _, running_rows in with_running)
    print(
        f"window {window_ms / 1e3:.2f} s, talker steps {len(steps)}, admissions {len(admissions)}"
    )
    if not admissions:
        return
    print(
        f"admission wall mean {statistics.fmean(a[0] for a in admissions):.2f} ms, "
        f"new rows mean {statistics.fmean(a[1] for a in admissions):.2f}, "
        f"talker time in admissions {sum(a[0] for a in admissions):.0f} ms "
        f"({100 * sum(a[0] for a in admissions) / window_ms:.1f} % of the window)"
    )
    print(
        f"admissions that found running rows {len(with_running)} of {len(admissions)}, "
        f"running rows then mean {statistics.fmean(a[2] for a in with_running) if with_running else 0:.2f}, "
        f"stream-ms stalled {stall_ms:.0f} "
        f"({stall_ms / max(1, sum(a[2] for a in with_running)):.2f} ms per stalled row)"
    )


if __name__ == "__main__":
    main()
