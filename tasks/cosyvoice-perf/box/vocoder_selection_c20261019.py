"""S0 of DESIGN_VOCODER_STANDARD_20261019.md: what the Fun-CosyVoice3 vocoder chose at every step and what each
first hop and final waited behind, from one nsys report taken with cosy_profile_probe.

A. Selections (the voc.rank marks of one select call, grouped by its sel id): runnable streams by class (first hop:
   causal window with no audio yet; started hop; final; fallback), chosen ones, how often both kinds were runnable,
   how often the head's kind had more runnable rows than the step took (the row cap), how often a first hop was
   runnable and not taken, and the most first hops runnable at once.
B. Steps (voc.step ranges): count, rows and wall by plan; causal steps that ran a plain whole history hop
   (flow.hop, a row the prefix pool could not hold) beside or instead of the prefix hop.
C. Waits: for each first hop and each final, from the moment it was ready to the start of its own step: the rest of
   the step running when it became ready, the whole steps that ran before its own (by plan), and the loop's time
   between steps; then its own step's wall and the time to its stream put.
   Ready is when its input reached the vocoder's inbox (the runtime's put, not the vocoder's ingest, which only
   happens between steps): a first hop at its first chunk's put; a final at the latest of its stream_done put, its
   payload's put and the end of its stream's previous step.

usage: python3 vocoder_selection_c20261019.py REPORT.sqlite [--skip-head-s S] [--skip-tail-s S] [--cap N]
"""

from __future__ import annotations

import argparse
import bisect
import re
import sqlite3
import statistics
from collections import Counter, defaultdict

RANK = re.compile(r"voc\.rank rid=(\S+) plan=(\S+) slack=(\S+) chosen=(\d) sel=(\d+)")
STEPRID = re.compile(r"voc\.steprid rid=(\S+) plan=(\S+)")
STEP = re.compile(r"voc\.step (\S+) rows=(\d+)")
QUEUE = re.compile(r"q (\S+)\.(put|get) rid=(\S+) t=(\S+)")
HOP_PLAIN = re.compile(r"flow\.hop rows=(\d+)")
HOP_PREFIX = re.compile(r"flow\.hop\.prefix rows=(\d+)")
MS = 1e-6


def load(path: str) -> list[tuple[int, int | None, int, str]]:
    db = sqlite3.connect(path)
    columns = {row[1] for row in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    if "textId" in columns:
        query = (
            "SELECT e.start, e.end, e.globalTid, COALESCE(e.text, s.value) FROM NVTX_EVENTS e "
            "LEFT JOIN StringIds s ON e.textId = s.id"
        )
    else:
        query = "SELECT start, end, globalTid, text FROM NVTX_EVENTS"
    return sorted((row for row in db.execute(query) if row[3]), key=lambda row: row[0])


def summary(values: list[float]) -> str:
    if not values:
        return "n 0"
    ordered = sorted(values)
    return (
        f"n {len(values)} mean {statistics.mean(values):.1f} p50 {ordered[len(ordered) // 2]:.1f} "
        f"p90 {ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]:.1f} max {ordered[-1]:.1f}"
    )


def stream_class(plan: str, slack: str) -> str:
    if plan == "causal_window":
        return "first hop" if slack == "u" else "started hop"
    elif plan == "leftover":
        return "final"
    else:
        return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--skip-head-s", type=float, default=1.0)
    parser.add_argument("--skip-tail-s", type=float, default=1.0)
    parser.add_argument("--cap", type=int, default=16)
    args = parser.parse_args()

    ranks: dict[int, list[tuple[int, str, str, str, bool]]] = defaultdict(list)
    steprids: dict[str, list[tuple[int, str]]] = defaultdict(list)
    steps: list[tuple[int, int, str, int]] = []
    payload_put: dict[str, int] = {}
    done_put: dict[str, int] = {}
    chunk_put: dict[str, int] = {}
    outs: dict[str, list[int]] = defaultdict(list)
    hop_ranges: list[tuple[int, str, int]] = []
    for start, end, _, text in load(args.report):
        if match := RANK.match(text):
            ranks[int(match[5])].append((start, match[1], match[2], match[3], match[4] == "1"))
        elif match := STEPRID.match(text):
            steprids[match[1]].append((start, match[2]))
        elif (match := STEP.match(text)) and end is not None:
            steps.append((start, end, match[1], int(match[2])))
        elif match := QUEUE.match(text):
            label, op, rid, kind = match[1], match[2], match[3], match[4]
            if label == "vocoder.in" and op == "put" and kind == "new_request":
                payload_put.setdefault(rid, start)
            elif label == "vocoder.in" and op == "put" and kind == "stream_done":
                done_put.setdefault(rid, start)
            elif label == "vocoder.in" and op == "put" and kind == "stream_chunk":
                chunk_put.setdefault(rid, start)
            elif label == "vocoder.out" and op == "put" and kind == "stream":
                outs[rid].append(start)
        elif (match := HOP_PREFIX.match(text)) and end is not None:
            hop_ranges.append((start, "prefix", int(match[1])))
        elif (match := HOP_PLAIN.match(text)) and end is not None:
            hop_ranges.append((start, "plain", int(match[1])))
    if not steps:
        print("no voc.step ranges")
        return
    steps.sort()
    t0 = steps[0][0] + args.skip_head_s * 1e9
    t1 = steps[-1][1] - args.skip_tail_s * 1e9
    window = [step for step in steps if t0 <= step[0] and step[1] <= t1]
    step_starts = [step[0] for step in steps]

    print(f"report {args.report}, window {(t1 - t0) / 1e9:.1f} s, steps {len(window)}")
    print("\nA. selections")
    sel_count = both = capped = passed = 0
    first_runnable: list[int] = []
    first_passed: list[int] = []
    runnable_total: list[int] = []
    head_classes: Counter = Counter()
    for _, entries in sorted(ranks.items()):
        time = entries[0][0]
        if not t0 <= time <= t1:
            continue
        sel_count += 1
        classes = Counter(stream_class(plan, slack) for _, _, plan, slack, _ in entries)
        chosen = [(plan, slack) for _, _, plan, slack, taken in entries if taken]
        runnable_total.append(len(entries))
        first_runnable.append(classes["first hop"])
        has_hop = classes["first hop"] + classes["started hop"] > 0
        both += has_hop and classes["final"] > 0
        if chosen:
            head_plan = chosen[0][0]
            head_classes[Counter(stream_class(plan, slack) for plan, slack in chosen).most_common(1)[0][0]] += 1
            same_kind = sum(1 for _, _, plan, _, _ in entries if plan == head_plan)
            capped += len(chosen) >= args.cap and same_kind > len(chosen)
        not_taken = sum(
            1 for _, _, plan, slack, taken in entries if stream_class(plan, slack) == "first hop" and not taken
        )
        first_passed.append(not_taken)
        passed += not_taken > 0
    if sel_count:
        print(
            f"  {sel_count} selections; runnable streams per selection {summary(runnable_total)}\n"
            f"  both kinds runnable {both} ({100 * both / sel_count:.1f} %); the head's kind past the {args.cap} row"
            f" cap {capped} ({100 * capped / sel_count:.1f} %)\n"
            f"  a first hop runnable and not taken {passed} ({100 * passed / sel_count:.1f} %), first hops left out"
            f" per selection {summary(first_passed)}\n"
            f"  first hops runnable per selection {summary(first_runnable)}\n"
            f"  steps by the class most of their rows belong to: "
            + ", ".join(f"{name} {count}" for name, count in head_classes.most_common())
        )

    print("\nB. steps")
    by_plan: dict[str, list[tuple[int, int, str, int]]] = defaultdict(list)
    for step in window:
        by_plan[step[2]].append(step)
    for plan, plan_steps in sorted(by_plan.items()):
        print(
            f"  {plan}: {len(plan_steps)} steps, rows {summary([float(s[3]) for s in plan_steps])},"
            f" wall ms {summary([(s[1] - s[0]) * MS for s in plan_steps])}"
        )
    plain_steps = Counter()
    for start, kind, rows in hop_ranges:
        index = bisect.bisect_right(step_starts, start) - 1
        if index >= 0 and steps[index][0] <= start <= steps[index][1] and t0 <= start <= t1:
            plain_steps[kind] += 1
    print(f"  Flow hops inside causal steps: prefix {plain_steps['prefix']}, plain whole history {plain_steps['plain']}")

    def wait_parts(ready: int, own_start: int) -> tuple[float, float, Counter, Counter, float]:
        """Rest of the running step, whole steps before its own (ms by plan, count by plan), loop time."""
        index = bisect.bisect_right(step_starts, ready) - 1
        running_rest = 0.0
        cursor = ready
        if index >= 0 and steps[index][0] <= ready < steps[index][1] and steps[index][0] < own_start:
            running_rest = (steps[index][1] - ready) * MS
            cursor = steps[index][1]
            index += 1
        else:
            index += 1
        whole = Counter()
        count = Counter()
        while index < len(steps) and steps[index][0] < own_start:
            whole[steps[index][2]] += (steps[index][1] - steps[index][0]) * MS
            count[steps[index][2]] += 1
            index += 1
        total = (own_start - ready) * MS
        return total, running_rest, whole, count, total - running_rest - sum(whole.values())

    print("\nC. waits from ready to the start of the stream's own step (ms)")
    for kind in ("first hop", "final"):
        totals, rests, loops, owns, outs_after = [], [], [], [], []
        whole_sum, count_sum = Counter(), Counter()
        n = 0
        for rid, carried in steprids.items():
            carried = sorted(carried)
            if kind == "first hop":
                if rid not in chunk_put or carried[0][1] != "causal_window":
                    continue
                ready, own_time = chunk_put[rid], carried[0][0]
            else:
                finals = [entry for entry in carried if entry[1] == "leftover"]
                if not finals or rid not in done_put:
                    continue
                own_time = finals[0][0]
                earlier = [entry[0] for entry in carried if entry[0] < own_time]
                previous_end = 0
                if earlier:
                    index = bisect.bisect_right(step_starts, earlier[-1]) - 1
                    previous_end = steps[index][1]
                ready = max(done_put[rid], payload_put.get(rid, 0), previous_end)
            if not t0 <= ready <= t1:
                continue
            own_index = bisect.bisect_right(step_starts, own_time) - 1
            own = steps[own_index]
            total, rest, whole, count, loop = wait_parts(ready, own[0])
            n += 1
            totals.append(total)
            rests.append(rest)
            loops.append(loop)
            owns.append((own[1] - own[0]) * MS)
            whole_sum.update(whole)
            count_sum.update(count)
            later_outs = [t for t in outs.get(rid, []) if t >= own[1]]
            if later_outs:
                outs_after.append((later_outs[0] - own[1]) * MS)
        if not n:
            print(f"  {kind}: none")
            continue
        print(
            f"  {kind}: {n}\n"
            f"    ready to own step {summary(totals)}\n"
            f"    rest of the running step {summary(rests)}\n"
            f"    whole steps before its own, ms per {kind}: "
            + ", ".join(f"{plan} {t / n:.1f} ({count_sum[plan] / n:.2f} steps)" for plan, t in whole_sum.most_common())
            + f"\n    loop between steps {summary(loops)}\n"
            f"    own step wall {summary(owns)}; own step end to its stream put {summary(outs_after)}"
        )


if __name__ == "__main__":
    main()
