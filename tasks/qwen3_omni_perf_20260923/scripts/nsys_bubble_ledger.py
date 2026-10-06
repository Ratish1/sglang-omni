"""Where the card has no kernel, and what every engine loop was doing then (run on the box).

From one probed nsys report: the union of all kernels on the card gives the busy time; every gap between them is a
bubble. Bubbles are bucketed by length (a launch gap inside a step is microseconds, a host turn is milliseconds), and
each bubble's time is charged to the innermost sched.* or c2w.* range each stage's scheduler thread was in at the
bubble's midpoint ("outside" when the thread was in no range: the loop's own code between ranges). Per stage this
gives the bubble time by state; across stages, by the pair of states of the thinker and the talker.

usage: python nsys_bubble_ledger.py REPORT.sqlite [--window window.txt] [--top 12]
"""

from __future__ import annotations

import argparse
import bisect
import collections
import sqlite3

from nsys_stage_ledger import pid_of, session_window

MARK = 34
BUCKETS_US = (20, 200, 1000, 10000)


def stage_ranges(db: sqlite3.Connection, strings: dict[int, str], t0: int, t1: int):
    names_of_pid: dict[int, set[str]] = collections.defaultdict(set)
    ranges: dict[str, list[tuple[int, int, str]]] = collections.defaultdict(list)
    rows = db.execute(
        "select start, end, globalTid, text, textId, eventType from NVTX_EVENTS order by start"
    ).fetchall()
    for start, end, tid, text, text_id, event_type in rows:
        label = text if text is not None else strings.get(text_id, "")
        if event_type == MARK and label and label.startswith("proc stage="):
            names_of_pid[pid_of(tid)].add(label.split("=", 1)[1])
    stage_of_pid = {pid: "+".join(sorted(names)) for pid, names in names_of_pid.items()}
    for start, end, tid, text, text_id, event_type in rows:
        label = text if text is not None else strings.get(text_id, "")
        if end is None or not label or end < t0 or start > t1:
            continue
        if label.startswith(("sched.", "c2w.")):
            stage = stage_of_pid.get(pid_of(tid))
            if stage is not None:
                words = label.split(" ")
                name = (
                    " ".join(words[:2])
                    if len(words) > 1 and "=" not in words[1]
                    else words[0]
                )
                ranges[stage].append((start, end, name))
    for stage in ranges:
        ranges[stage].sort()
    return ranges


def state_at(
    stage_list: list[tuple[int, int, str]], starts: list[int], point: int
) -> str:
    index = bisect.bisect_right(starts, point) - 1
    best = None
    scanned = 0
    while index >= 0 and scanned < 64:
        start, end, name = stage_list[index]
        if start <= point <= end and (best is None or end - start < best[1] - best[0]):
            best = (start, end, name)
        index -= 1
        scanned += 1
    return best[2] if best else "outside"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--skip-head-s", type=float, default=0.0)
    parser.add_argument("--skip-tail-s", type=float, default=0.0)
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    # note (ratish): the window opens before the measured requests arrive and closes over the drain;
    # the head and tail cut leave the steady state.
    t0, t1 = t0 + int(args.skip_head_s * 1e9), t1 - int(args.skip_tail_s * 1e9)
    kernels = db.execute(
        "select start, end, deviceId from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ? order by start",
        (t0, t1),
    ).fetchall()
    device = collections.Counter(k[2] for k in kernels).most_common(1)[0][0]
    busy = []
    for start, end, dev in kernels:
        if dev != device:
            continue
        start, end = max(start, t0), min(end, t1)
        if busy and start <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], end)
        else:
            busy.append([start, end])
    gaps = []
    cursor = t0
    for start, end in busy:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if t1 > cursor:
        gaps.append((cursor, t1))
    window = t1 - t0
    gap_total = sum(e - s for s, e in gaps)
    print(
        f"window {window / 1e9:.2f} s, card busy {100 * (window - gap_total) / window:.1f} %, no kernel {100 * gap_total / window:.1f} %"
    )
    by_bucket = collections.Counter()
    for s, e in gaps:
        length_us = (e - s) / 1e3
        bucket = next(
            (f"<{b} us" for b in BUCKETS_US if length_us < b), f">={BUCKETS_US[-1]} us"
        )
        by_bucket[bucket] += e - s
    for bucket in [f"<{b} us" for b in BUCKETS_US] + [f">={BUCKETS_US[-1]} us"]:
        print(
            f"  bubbles {bucket:>10s}: {100 * by_bucket[bucket] / window:5.1f} % of the window"
        )
    per_second = collections.Counter()
    for s, e in gaps:
        per_second[(s - t0) // 1_000_000_000] += e - s
    print(
        "  no-kernel share per second of the window: "
        + " ".join(
            f"{100 * per_second[sec] / 1e9:.0f}"
            for sec in range(int(window // 1_000_000_000) + 1)
        )
    )
    ranges = stage_ranges(db, strings, t0, t1)
    starts = {stage: [r[0] for r in lst] for stage, lst in ranges.items()}
    per_stage = {stage: collections.Counter() for stage in ranges}
    pairs = collections.Counter()
    for s, e in gaps:
        if e - s < 20_000:
            continue
        mid = (s + e) // 2
        states = {
            stage: state_at(ranges[stage], starts[stage], mid) for stage in ranges
        }
        for stage, state in states.items():
            per_stage[stage][state] += e - s
        talker = next((name for name in states if "talker_ar" in name), None)
        pairs[
            (states.get("thinker", "-"), states.get(talker, "-") if talker else "-")
        ] += (e - s)
    print(
        "\nbubbles of 20 us or more, charged to each stage's scheduler state at the bubble's midpoint (% of the window)"
    )
    for stage, counter in sorted(per_stage.items()):
        top = ", ".join(
            f"{name} {100 * t / window:.1f}"
            for name, t in counter.most_common(args.top)
        )
        print(f"  {stage}: {top}")
    print("\nby thinker state x talker state (% of the window)")
    for (thinker, talker), t in pairs.most_common(args.top):
        print(f"  thinker {thinker:22s} talker {talker:22s} {100 * t / window:5.1f}")


if __name__ == "__main__":
    main()
