#!/usr/bin/env python3
"""Where a speech stream's chunk gaps go, per arm, from hop probe records inside one events cell's window.

usage: python3 chunk_gap_parts.py --cell c16_events <boot> [<boot> ...]

Boots run with baseline.py --probe and OMNI_HOP_PROBE_TALKER_BATCHES=1. Per boot:
  periods      talker decode period (one decode step start to the next) split into the step (start to its result),
               its result processing, prefill steps between, and the rest of the loop
  by rows      period and step mean at each decode batch size, so arms are compared at equal rows
  row list     steps whose request list differs from the previous step's, and their step mean against the others
  sat out      steps a request missed between its first and last decode step
  thinker      decode steps during which the thinker sent a token, and their step mean against steps without
  chunk gaps   server side, between a stream's code2wav outputs, split into the gap between the talker frames each
               output closed on and the change in code2wav's delay after those frames
"""

import argparse
import bisect
import collections
import glob
import json
import os
import statistics

ROW_BUCKETS = (
    (1, 8),
    (9, 10),
    (11, 12),
    (13, 13),
    (14, 14),
    (15, 15),
    (16, 16),
    (17, 32),
    (33, 64),
)


def load_records(
    boot: str, cell: str, kinds: tuple[str, ...] = ("tb", "tr")
) -> tuple[list[dict], list[list], float, float]:
    with open(os.path.join(boot, cell, "window.json"), encoding="utf-8") as source:
        window = json.load(source)
    start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
    events, queue = [], []
    for path in glob.glob(os.path.join(boot, "probe", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                if isinstance(item, dict):
                    if item.get("k") in kinds and start_ns <= item["ns"] <= end_ns:
                        events.append(item)
                    else:
                        pass
                elif start_ns <= item[0] <= end_ns:
                    queue.append(item)
                else:
                    pass
    events.sort(key=lambda item: item["ns"])
    queue.sort(key=lambda item: item[0])
    return events, queue, start_ns, end_ns


def mean_ms(values_ns: list[float]) -> float:
    return statistics.fmean(values_ns) / 1e6 if values_ns else float("nan")


def report(boot: str, cell: str) -> None:
    events, queue, start_ns, end_ns = load_records(boot, cell)
    starts = [item for item in events if item["k"] == "tb"]
    results = [item for item in events if item["k"] == "tr"]
    result_ends = [item["ns"] for item in results]
    seconds = (end_ns - start_ns) / 1e9
    thinker_tokens = [
        item[0]
        for item in queue
        if item[1] == "thinker.out" and item[2] == "put" and item[6] == "talker_ar"
    ]
    print(f"== {boot} ({cell}, {seconds:.1f} s)")

    steps = []
    for index, start in enumerate(starts[:-1]):
        next_start = starts[index + 1]["ns"]
        first = bisect.bisect_left(result_ends, start["ns"])
        own = next(
            (
                results[i]
                for i in range(first, len(results))
                if results[i]["mode"] == "DECODE"
            ),
            None,
        )
        if own is None or own["ns"] > next_start:
            continue
        else:
            pass
        between = [
            item
            for item in results[first:]
            if own["ns"] < item["ns"] <= next_start and item["mode"] != "DECODE"
        ]
        prefill_ns = 0.0
        previous_end = own["ns"]
        for item in between:
            prefill_ns += item["ns"] - previous_end
            previous_end = item["ns"]
        tokens = bisect.bisect_left(thinker_tokens, own["ns0"]) - bisect.bisect_left(
            thinker_tokens, start["ns"]
        )
        steps.append(
            {
                "rows": start["rows"],
                "rids": start["rids"],
                "period": next_start - start["ns"],
                "step": own["ns0"] - start["ns"],
                "result": own["ns"] - own["ns0"],
                "prefills": len(between),
                "prefill": prefill_ns,
                "rest": next_start - own["ns"] - prefill_ns,
                "thinker_tokens": tokens,
            }
        )
    if not steps:
        print("  no decode steps")
        return
    else:
        pass
    print(
        f"  periods: {len(steps)} decode steps ({len(steps) / seconds:.1f}/s), rows mean "
        f"{statistics.fmean(s['rows'] for s in steps):.2f}; period {mean_ms([s['period'] for s in steps]):.2f} ms = "
        f"step {mean_ms([s['step'] for s in steps]):.2f} + result {mean_ms([s['result'] for s in steps]):.2f} + "
        f"prefill {mean_ms([s['prefill'] for s in steps]):.2f} ({statistics.fmean(s['prefills'] for s in steps):.2f} "
        f"a period) + rest {mean_ms([s['rest'] for s in steps]):.2f}"
    )
    for low, high in ROW_BUCKETS:
        bucket = [s for s in steps if low <= s["rows"] <= high]
        if bucket:
            print(
                f"  by rows {low:2d}-{high:2d}: {len(bucket):5d} steps, period {mean_ms([s['period'] for s in bucket]):6.2f}"
                f" step {mean_ms([s['step'] for s in bucket]):6.2f} result {mean_ms([s['result'] for s in bucket]):5.2f}"
                f" prefill {mean_ms([s['prefill'] for s in bucket]):5.2f} ms"
            )
        else:
            pass

    row_steps = sum(s["rows"] for s in steps)
    print(
        f"  stream period (each step weighted by its rows): {sum(s['rows'] * s['period'] for s in steps) / row_steps / 1e6:.2f} ms,"
        f" step {sum(s['rows'] * s['step'] for s in steps) / row_steps / 1e6:.2f},"
        f" prefill {sum(s['rows'] * s['prefill'] for s in steps) / row_steps / 1e6:.2f}"
    )
    steps[0]["change"] = "first"
    for prior, step in zip(steps, steps[1:]):
        if step["rids"] == prior["rids"]:
            step["change"] = "same"
        elif sorted(step["rids"]) == sorted(prior["rids"]):
            step["change"] = "order"
        else:
            step["change"] = "members"
    changed = [s for s in steps[1:] if s["change"] != "same"]
    kept = [s for s in steps[1:] if s["change"] == "same"]
    print(
        f"  row list: {len(changed)} of {len(steps) - 1} steps changed it ({100 * len(changed) / max(len(steps) - 1, 1):.1f} %;"
        f" order only {sum(1 for s in changed if s['change'] == 'order')}),"
        f" step {mean_ms([s['step'] for s in changed]):.2f} against {mean_ms([s['step'] for s in kept]):.2f} ms unchanged"
    )
    for low, high in ROW_BUCKETS:
        cells = []
        for change in ("same", "changed"):
            for thinker in ("idle", "sending"):
                values = [
                    s["step"]
                    for s in steps[1:]
                    if low <= s["rows"] <= high
                    and (s["change"] == "same") == (change == "same")
                    and bool(s["thinker_tokens"]) == (thinker == "sending")
                ]
                cells.append(
                    f"{change}/{thinker} {mean_ms(values):6.2f} ({len(values)})"
                )
        if sum(1 for s in steps[1:] if low <= s["rows"] <= high) >= 50:
            print(
                f"    rows {low:2d}-{high:2d} step by row list and thinker: "
                + ", ".join(cells)
            )
        else:
            pass

    appearances = collections.defaultdict(list)
    for index, step in enumerate(steps):
        for rid in step["rids"]:
            appearances[rid].append(index)
    sat_out = {
        rid: indices[-1] - indices[0] + 1 - len(indices)
        for rid, indices in appearances.items()
    }
    rows_total = sum(len(indices) for indices in appearances.values())
    print(
        f"  sat out: {sum(sat_out.values())} request steps over {len(sat_out)} requests "
        f"({100 * sum(sat_out.values()) / max(rows_total, 1):.2f} % of request steps), requests with any "
        f"{sum(1 for count in sat_out.values() if count)}"
    )

    with_tokens = [s for s in steps if s["thinker_tokens"]]
    without = [s for s in steps if not s["thinker_tokens"]]
    print(
        f"  thinker: {len(with_tokens)} steps with a thinker token sent during them ({100 * len(with_tokens) / len(steps):.1f} %),"
        f" step {mean_ms([s['step'] for s in with_tokens]):.2f} against {mean_ms([s['step'] for s in without]):.2f} ms"
    )
    for low, high in ROW_BUCKETS:
        a = [s["step"] for s in with_tokens if low <= s["rows"] <= high]
        b = [s["step"] for s in without if low <= s["rows"] <= high]
        if len(a) >= 20 and len(b) >= 20:
            print(
                f"    rows {low:2d}-{high:2d}: step with thinker tokens {mean_ms(a):6.2f} ({len(a)}), without {mean_ms(b):6.2f} ({len(b)}) ms"
            )
        else:
            pass

    frames = collections.defaultdict(list)
    outputs = collections.defaultdict(list)
    for item in queue:
        if item[2] != "put" or item[5] != "stream":
            continue
        elif item[1] == "talker_ar.out" and item[6] == "code2wav":
            frames[item[4]].append(item[0])
        elif item[1] == "code2wav.out":
            outputs[item[4]].append(item[0])
        else:
            pass
    last_thinker_token = {}
    for item in queue:
        if item[1] == "thinker.out" and item[2] == "put" and item[6] == "talker_ar":
            last_thinker_token[item[4]] = item[0]
        else:
            pass
    gaps_by_phase = {"thinker running": [], "thinker done": []}
    gaps_by_index = collections.defaultdict(list)
    gaps, frame_gaps, delay_changes, delays = [], [], [], []
    for rid, output_times in outputs.items():
        thinker_end = last_thinker_token.get(rid)
        for index, (out_a, out_b) in enumerate(
            zip(output_times, output_times[1:]), start=1
        ):
            if thinker_end is not None:
                phase = "thinker running" if out_b <= thinker_end else "thinker done"
                gaps_by_phase[phase].append(out_b - out_a)
            else:
                pass
            gaps_by_index[min(index, 6)].append(out_b - out_a)
        frame_times = frames.get(rid, [])
        if len(output_times) < 2 or not frame_times:
            continue
        else:
            pass
        closed = []
        for output_ns in output_times:
            index = bisect.bisect_right(frame_times, output_ns) - 1
            closed.append(frame_times[index] if index >= 0 else None)
        for (out_a, frame_a), (out_b, frame_b) in zip(
            zip(output_times, closed), zip(output_times[1:], closed[1:])
        ):
            if frame_a is None or frame_b is None:
                continue
            else:
                pass
            gaps.append(out_b - out_a)
            frame_gaps.append(frame_b - frame_a)
            delay_changes.append((out_b - frame_b) - (out_a - frame_a))
        delays.extend(
            out - frame for out, frame in zip(output_times, closed) if frame is not None
        )
    if gaps:
        print(
            f"  chunk gaps: {len(gaps)} gaps, mean {mean_ms(gaps):.1f} ms = frames {mean_ms(frame_gaps):.1f} + code2wav delay "
            f"change {mean_ms(delay_changes):.1f}; code2wav delay after its frames mean {mean_ms(delays):.1f} ms"
        )
    else:
        print("  chunk gaps: none matched")
    phase_counts = {phase: len(values) for phase, values in gaps_by_phase.items()}
    print(
        "  chunk gaps by the request's thinker: "
        + ", ".join(
            f"{phase} {mean_ms(values):.1f} ms ({phase_counts[phase]})"
            for phase, values in gaps_by_phase.items()
        )
    )
    print(
        "  chunk gaps by index (6 = sixth and later): "
        + ", ".join(
            f"{index}: {mean_ms(gaps_by_index[index]):.1f} ({len(gaps_by_index[index])})"
            for index in sorted(gaps_by_index)
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True)
    parser.add_argument("boots", nargs="+")
    args = parser.parse_args()
    for boot in args.boots:
        report(boot, args.cell)


if __name__ == "__main__":
    main()
