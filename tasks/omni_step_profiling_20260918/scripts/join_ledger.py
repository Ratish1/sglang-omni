"""How often a prefill joined a decode step's predictor pass and what each kind of step cost, from one nsys report of a
joined head under the pipeline_nvtx probe, inside the benchmark's window.

usage: python3 join_ledger.py serve.sqlite --bench-log bench.log
"""

import argparse
import bisect
import re
import sqlite3
import statistics

from pipeline_census import MARK, PUSH_POP, bench_window, pct


def summary(values: list[float]) -> str:
    if not values:
        return "n 0"
    return f"n {len(values)}, mean {statistics.fmean(values):.2f}, p50 {pct(values, 0.5):.2f}, p95 {pct(values, 0.95):.2f} ms"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--bench-log", required=True)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = bench_window(db, args.bench_log)
    decode_steps: list[tuple[int, int]] = []
    extend_steps: list[tuple[float, int]] = []
    hits: list[tuple[int, int]] = []
    refused: list[int] = []
    polls: list[tuple[int, int]] = []
    ranges: dict[str, list[float]] = {"mr.join": [], "mr.join_predict": [], "mr.join_finish": [], "mr.predictor": []}
    for start, end, text, text_id, kind in db.execute(
        "select start, end, text, textId, eventType from NVTX_EVENTS "
        "where eventType in (?, ?) and start >= ? and start <= ? order by start",
        (PUSH_POP, MARK, t0, t1),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if kind == MARK:
            if label.startswith("sched.join rows="):
                hits.append((start, int(label.split("=")[1])))
            elif label.startswith("sched.join_refused"):
                refused.append(start)
            continue
        if end is None:
            continue
        if label.startswith("sched.batch decode"):
            decode_steps.append((start, end))
        elif label.startswith("sched.batch extend"):
            rows = re.search(r"bs=(\d+)", label)
            extend_steps.append(((end - start) / 1e6, int(rows.group(1))))
        elif label == "sched.join_poll":
            polls.append((start, end))
        elif label in ranges:
            ranges[label].append((end - start) / 1e6)
    hit_starts = [start for start, _ in hits]
    joined_walls: list[float] = []
    alone_walls: list[float] = []
    for start, end in decode_steps:
        index = bisect.bisect_left(hit_starts, start)
        carries = index < len(hit_starts) and hit_starts[index] < end
        (joined_walls if carries else alone_walls).append((end - start) / 1e6)
    poll_hit: list[float] = []
    poll_empty: list[float] = []
    for start, end in polls:
        index = bisect.bisect_left(hit_starts, start)
        carries = index < len(hit_starts) and hit_starts[index] <= end
        (poll_hit if carries else poll_empty).append((end - start) / 1e6)
    extend_walls = [wall for wall, _ in extend_steps]
    alone_rows = sum(rows for _, rows in extend_steps)
    joined_rows = sum(rows for _, rows in hits)
    print(f"window {(t1 - t0) / 1e9:.2f} s")
    print(f"decode step alone: {summary(alone_walls)}")
    print(f"decode step carrying a prefill: {summary(joined_walls)}")
    print(f"prefill step alone: {summary(extend_walls)}; rows a step {alone_rows / max(len(extend_steps), 1):.2f}")
    print(f"joins {len(hits)}; rows a join {joined_rows / max(len(hits), 1):.2f}")
    print(f"prefill rows alone {alone_rows}, joined {joined_rows}; joined share {100 * joined_rows / max(alone_rows + joined_rows, 1):.1f} %")
    print(f"polls {len(polls)}: found a prefill {summary(poll_hit)}; found none {summary(poll_empty)}; waiting but refused {len(refused)}")
    for name, values in ranges.items():
        print(f"{name}: {summary(values)}")
    if joined_walls and alone_walls and extend_walls:
        apart = statistics.fmean(alone_walls) + statistics.fmean(extend_walls)
        together = statistics.fmean(joined_walls)
        print(f"one prefill and one decode: apart {apart:.2f} ms, joined {together:.2f} ms, saved {apart - together:.2f} ms a join")
        print(f"saved in the window: {(apart - together) * len(joined_walls):.0f} ms of {(t1 - t0) / 1e6:.0f}")
    else:
        pass


if __name__ == "__main__":
    main()
