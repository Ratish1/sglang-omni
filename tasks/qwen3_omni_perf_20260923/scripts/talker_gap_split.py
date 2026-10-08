"""Where the talker's part of each inter-chunk gap goes, from a probed census with the per-step rows mark.

For each request, the time between two of its window-completing codec messages leaving the talker (as in
chunk_gap_origin.py) is split by the talker scheduler thread's top-level ranges inside it: decode steps that carried
the request (its rid in the step's "sched.rows" mark), decode steps without it, extend steps, the scheduler's other
ranges (result, recv and input, pick, idle) and time in no range. Each carried step's stretch is its wall minus the
median wall of decode steps with the same rows that no thinker prefill step overlapped, so a gap's stretch is the
time the thinker's prefill added to the steps the request rode. Reported by the band of the coordinator gap, as
census section L, then every talker decode step by how much a thinker mixed or extend step overlaps it.

usage: python3 talker_gap_split.py REPORT.sqlite --window window.txt
"""

from __future__ import annotations

import argparse
import bisect
import collections
import re
import sqlite3
import statistics

from nsys_stage_ledger import session_window

PUSH_POP = 59
MARK = 34
FIRST_WINDOW_MESSAGE = 9
STEP = re.compile(r"sched\.(?:batch|launch) (decode|extend|mixed) bs=(\d+)")
PARTS = (
    "carried",
    "sat out",
    "extend",
    "result",
    "recv/input",
    "pick",
    "idle",
    "other",
    "no range",
)


def part_of(label: str, carried: bool) -> str:
    step = STEP.match(label)
    if step:
        if step.group(1) != "decode":
            return "extend"
        else:
            return "carried" if carried else "sat out"
    else:
        pass
    kind = label.split(" ", 1)[0]
    if kind == "sched.result":
        return "result"
    elif kind in ("sched.recv", "sched.input"):
        return "recv/input"
    elif kind == "sched.pick":
        return "pick"
    elif kind == "sched.idle":
        return "idle"
    else:
        return "other"


def top_level(items: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
    """Ranges of one thread not nested in another, from (start, end, label) sorted by start, end descending."""
    out = []
    for start, end, label in items:
        if out and start < out[-1][1]:
            continue
        else:
            out.append((start, end, label))
    return out


def overlap(a: int, b: int, start: int, end: int) -> int:
    return max(0, min(b, end) - max(a, start))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    args = parser.parse_args()
    db = sqlite3.connect(args.report)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = session_window(db, args.window)
    names: dict[int, str] = {}
    ranges: dict[int, list] = collections.defaultdict(list)
    rows_marks: dict[int, list] = collections.defaultdict(list)
    sent: dict[str, list[int]] = collections.defaultdict(list)
    arrived: dict[str, list[int]] = collections.defaultdict(list)
    for start, end, tid, text, text_id, kind in db.execute(
        "select start, end, globalTid, text, textId, eventType from NVTX_EVENTS where eventType in (?, ?) "
        "order by start, end desc",
        (PUSH_POP, MARK),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if kind == PUSH_POP:
            if end is not None and label:
                ranges[tid].append((start, end, label))
            else:
                pass
            continue
        else:
            pass
        if label.startswith("thread name="):
            names[tid] = label.split(" ")[1][5:]
        elif label.startswith("sched.rows "):
            match = re.match(r"sched\.rows n=\d+ rids=(\S*)", label)
            rows_marks[tid].append(
                (start, frozenset(match.group(1).split(",")) if match else frozenset())
            )
        else:
            match = re.match(r"q talker_ar\.out\.put rid=(\S+) t=stream", label)
            if match:
                sent[match.group(1)].append(start)
                continue
            else:
                pass
            match = re.match(r"coord\.recv rid=(\S+) from=code2wav n=(\d+)", label)
            if match and int(match.group(2)) > 0:
                arrived[match.group(1)].append(start)
            else:
                pass
    by_name = {name: tid for tid, name in names.items()}
    talker = by_name.get("scheduler-talker_ar")
    thinker = by_name.get("scheduler-thinker")
    if talker is None or thinker is None:
        print(f"no talker or thinker scheduler thread among {sorted(by_name)}")
        return
    else:
        pass
    if not rows_marks.get(talker):
        print(
            "no sched.rows marks on the talker thread; the probe predates the per-step rows mark"
        )
        return
    else:
        pass
    talker_ranges = top_level(ranges[talker])
    talker_starts = [r[0] for r in talker_ranges]
    marks = sorted(rows_marks[talker])
    mark_times = [m[0] for m in marks]
    steps = []
    for start, end, label in talker_ranges:
        step = STEP.match(label)
        if step:
            index = bisect.bisect_left(mark_times, start)
            rids = (
                marks[index][1]
                if index < len(marks) and marks[index][0] <= end
                else frozenset()
            )
            steps.append((start, end, step.group(1), int(step.group(2)), rids))
        else:
            pass
    thinker_prefill = [
        (start, end)
        for start, end, label in top_level(ranges[thinker])
        if (step := STEP.match(label)) and step.group(1) != "decode"
    ]
    prefill_starts = [p[0] for p in thinker_prefill]

    def prefill_overlap(start: int, end: int) -> int:
        index = max(bisect.bisect_right(prefill_starts, start) - 1, 0)
        total = 0
        while index < len(thinker_prefill) and thinker_prefill[index][0] < end:
            total += overlap(start, end, *thinker_prefill[index])
            index += 1
        return total

    decode = [
        (start, end, rows, rids, prefill_overlap(start, end))
        for start, end, mode, rows, rids in steps
        if mode == "decode" and t0 <= start <= t1
    ]
    alone: dict[int, list[int]] = collections.defaultdict(list)
    for start, end, rows, _, shared in decode:
        if shared == 0:
            alone[rows].append(end - start)
        else:
            pass
    baseline = {rows: statistics.median(walls) for rows, walls in alone.items()}
    stretch_of = {
        start: (end - start) - baseline[rows]
        for start, end, rows, _, shared in decode
        if rows in baseline
    }

    rows_out = []
    for request_id, arrivals in arrived.items():
        windows = sent.get(request_id, [])[FIRST_WINDOW_MESSAGE:]
        count = min(len(windows), len(arrivals))
        for k in range(1, count):
            if not t0 <= arrivals[k] <= t1:
                continue
            else:
                pass
            a, b = windows[k - 1], windows[k]
            parts = collections.Counter()
            counts = collections.Counter()
            stretch = 0
            index = max(bisect.bisect_right(talker_starts, a) - 1, 0)
            while index < len(talker_ranges) and talker_ranges[index][0] < b:
                start, end, label = talker_ranges[index]
                index += 1
                inside = overlap(a, b, start, end)
                if inside == 0:
                    continue
                else:
                    pass
                step = STEP.match(label)
                carried = False
                if step:
                    position = bisect.bisect_left(mark_times, start)
                    carried = (
                        position < len(marks)
                        and marks[position][0] <= end
                        and request_id in marks[position][1]
                    )
                else:
                    pass
                part = part_of(label, carried)
                parts[part] += inside
                if a <= start < b:
                    counts[part] += 1
                    if part == "carried":
                        stretch += stretch_of.get(start, 0)
                    else:
                        pass
                else:
                    pass
            parts["no range"] = (b - a) - sum(parts.values())
            rows_out.append(
                (arrivals[k] - arrivals[k - 1], b - a, parts, counts, stretch)
            )
    if not rows_out:
        print("no gaps matched; check the probe marks")
        return
    else:
        pass
    rows_out.sort(key=lambda row: row[0])
    n = len(rows_out)
    print(
        f"window {(t1 - t0) / 1e9:.1f} s, {n} gaps; talker decode steps {len(decode)}"
    )
    print("  talker gap split (ms per gap; steps starting in the gap in brackets)")
    print(
        f"  {'band':<12}{'gaps':>6}{'gap':>8}{'talker':>8}"
        + "".join(f"{p:>14}" for p in PARTS)
        + f"{'stretch':>9}"
    )
    bands = (
        ("<= p50", 0, n // 2),
        ("p50 to p90", n // 2, int(n * 0.9)),
        ("p90 to p99", int(n * 0.9), int(n * 0.99)),
        ("> p99", int(n * 0.99), n),
    )
    for name, lo, hi in bands:
        band = rows_out[lo:hi]
        if not band:
            continue
        else:
            pass
        line = (
            f"  {name:<12}{len(band):>6}{statistics.fmean(r[0] for r in band) / 1e6:>8.1f}"
            f"{statistics.fmean(r[1] for r in band) / 1e6:>8.1f}"
        )
        for part in PARTS:
            value = statistics.fmean(r[2][part] for r in band) / 1e6
            steps_in = statistics.fmean(r[3][part] for r in band)
            line += (
                f"{value:>8.1f} [{steps_in:>4.1f}]"
                if part in ("carried", "sat out", "extend")
                else f"{value:>14.1f}"
            )
        line += f"{statistics.fmean(r[4] for r in band) / 1e6:>9.1f}"
        print(line)
    print("  talker decode steps by the thinker prefill overlapping them (ms)")
    print(
        f"  {'thinker prefill':<22}{'steps':>7}{'rows':>7}{'wall':>8}{'p50':>8}{'p90':>8}{'over baseline':>15}"
    )
    classes = (
        ("none", lambda share: share == 0),
        ("under half the step", lambda share: 0 < share < 0.5),
        ("half the step or more", lambda share: share >= 0.5),
    )
    for name, test in classes:
        chosen = [d for d in decode if test(d[4] / max(d[1] - d[0], 1))]
        if not chosen:
            continue
        else:
            pass
        walls = sorted(d[1] - d[0] for d in chosen)
        extra = [stretch_of[d[0]] for d in chosen if d[0] in stretch_of]
        print(
            f"  {name:<22}{len(chosen):>7}{statistics.fmean(d[2] for d in chosen):>7.2f}"
            f"{statistics.fmean(walls) / 1e6:>8.2f}{walls[len(walls) // 2] / 1e6:>8.2f}"
            f"{walls[int(len(walls) * 0.9)] / 1e6:>8.2f}{(statistics.fmean(extra) if extra else 0) / 1e6:>15.2f}"
        )
    print(
        "  baseline wall by rows (median, no thinker prefill overlap): "
        + ", ".join(
            f"{rows}: {value / 1e6:.2f}" for rows, value in sorted(baseline.items())
        )
    )


if __name__ == "__main__":
    main()
