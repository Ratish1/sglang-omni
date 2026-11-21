#!/usr/bin/env python3
"""The thinker's steps and each speech request's thinker token gaps inside one events cell's window, from hop probe
records of a boot run with OMNI_HOP_PROBE_THINKER_STEPS=1.

usage: python3 thinker_cadence.py --cell c64_events <boot> [<boot> ...]

Per boot:
  periods      the thinker's step periods (one step's end to the next step's end, a step ending when its stream output
               is put), by the class of the step that ends the period: decode launched on the lookahead path, decode on
               the synchronous path, mixed (an extend that carries decode rows) and extend alone; mixed and extend also by
               their extend tokens; decode also by rows, so arms are compared at equal rows
  speech gaps  per speech request (one the thinker streams to the talker), the time between consecutive token sends,
               by the class of the step whose output carried the later send, with each class's share of all gap time
  mixed rows   the rows a mixed step carried: its prefill rows, speech decode rows and other decode rows
  before       the time from the previous step's end to a synchronous mixed or extend step's start, and the part of it
               in result processing, input processing, batch selection and lookahead drains (a drain holds the result
               processing of the step it resolves, so those two can overlap)
"""

import argparse
import bisect
import collections
import glob
import json
import os
import statistics

EXTEND_BUCKETS = (
    (1, 512),
    (513, 2048),
    (2049, 4096),
    (4097, 6144),
    (6145, 8192),
    (8193, 1 << 30),
)
ROW_BUCKETS = ((1, 8), (9, 16), (17, 24), (25, 32), (33, 48), (49, 64))


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def describe(values: list[float]) -> str:
    if not values:
        return "none"
    else:
        return (
            f"n {len(values):6d} mean {statistics.fmean(values):7.2f} p50 {quantile(values, 0.5):7.2f} "
            f"p90 {quantile(values, 0.9):7.2f} p99 {quantile(values, 0.99):7.2f} ms"
        )


def load(
    boot: str, cell: str
) -> tuple[list[dict], list[list], list[dict], float, float]:
    with open(os.path.join(boot, cell, "window.json"), encoding="utf-8") as source:
        window = json.load(source)
    start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
    steps, sends, inputs = [], [], []
    for path in glob.glob(os.path.join(boot, "probe", "*.jsonl")):
        launches, resolves = [], []
        with open(path, encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                if isinstance(item, dict):
                    kind = item.get("k")
                    if kind == "ts":
                        steps.append({**item, "path": "sync"})
                    elif kind == "tl":
                        launches.append(item)
                    elif kind == "tz":
                        resolves.append(item)
                    elif (
                        kind in ("tq", "tg", "td", "tp")
                        and start_ns <= item["ns"] <= end_ns
                    ):
                        inputs.append(item)
                    else:
                        pass
                elif (
                    item[1] == "thinker.out"
                    and item[2] == "put"
                    and item[6] == "talker_ar"
                ):
                    sends.append(item)
                else:
                    pass
        launches.sort(key=lambda item: item["ns"])
        resolves.sort(key=lambda item: item["ns0"])
        if launches and abs(len(launches) - len(resolves)) > 2:
            print(
                f"  {path}: {len(launches)} launches against {len(resolves)} resolves, pairing by order anyway"
            )
        else:
            pass
        for launch, resolve in zip(launches, resolves):
            steps.append(
                {
                    **launch,
                    "ns0": resolve["ns0"],
                    "ns": resolve["ns"],
                    "launch_ns": launch["ns"],
                    "path": "lookahead",
                }
            )
    steps.sort(key=lambda item: item["ns"])
    sends.sort(key=lambda item: item[0])
    return steps, sends, inputs, start_ns, end_ns


def step_class(step: dict) -> str:
    if step["mode"] in ("DECODE", "IDLE"):
        return f"decode {step['path']}"
    elif step["mode"] == "MIXED":
        return "mixed"
    else:
        return "extend"


def extend_bucket(tokens: int) -> str:
    for low, high in EXTEND_BUCKETS:
        if low <= tokens <= high:
            return f"{low}-{high if high < 1 << 30 else 'more'}"
        else:
            pass
    return "0"


def row_bucket(rows: int) -> str:
    for low, high in ROW_BUCKETS:
        if low <= rows <= high:
            return f"{low:2d}-{high:2d}"
        else:
            pass
    return "more"


def report(boot: str, cell: str) -> None:
    steps, sends, inputs, start_ns, end_ns = load(boot, cell)
    speech = {item[4] for item in sends}
    in_window = [step for step in steps if start_ns <= step["ns"] <= end_ns]
    print(
        f"{boot}: {len(in_window)} thinker steps in a {(end_ns - start_ns) / 1e9:.1f} s window, {len(speech)} speech requests"
    )
    periods = collections.defaultdict(list)
    before_step = collections.defaultdict(list)
    in_step = collections.defaultdict(list)
    by_extend = collections.defaultdict(list)
    by_rows = collections.defaultdict(list)
    mixed_rows = collections.defaultdict(list)
    previous_end = None
    for step in steps:
        if previous_end is not None and start_ns <= step["ns"] <= end_ns:
            period_ms = (step["ns"] - previous_end) / 1e6
            kind = step_class(step)
            periods[kind].append(period_ms)
            before_step[kind].append(max(0.0, (step["ns0"] - previous_end) / 1e6))
            in_step[kind].append((step["ns"] - max(step["ns0"], previous_end)) / 1e6)
            if kind in ("mixed", "extend"):
                by_extend[(kind, extend_bucket(step["ext"]))].append(period_ms)
            else:
                by_rows[(kind, row_bucket(len(step["rids"])))].append(period_ms)
            if kind == "mixed":
                prefill_rows = {row for row, _ in step["pre"]}
                decode_rids = [
                    rid
                    for row, rid in enumerate(step["rids"])
                    if row not in prefill_rows
                ]
                mixed_rows["prefill"].append(len(prefill_rows))
                mixed_rows["speech decode"].append(
                    sum(rid in speech for rid in decode_rids)
                )
                mixed_rows["other decode"].append(
                    sum(rid not in speech for rid in decode_rids)
                )
            else:
                pass
        else:
            pass
        previous_end = step["ns"]
    window_ms = (end_ns - start_ns) / 1e6
    print("  periods by the class of the step that ends them (share of the window):")
    for kind in sorted(periods):
        print(
            f"    {kind:18s} {100 * sum(periods[kind]) / window_ms:5.1f} %  {describe(periods[kind])}"
        )
        print(
            f"    {'':18s} before the step {statistics.fmean(before_step[kind]):7.2f} ms, "
            f"the step {statistics.fmean(in_step[kind]):7.2f} ms (from its start, or the previous end on lookahead)"
        )
    input_ms = [
        (item["ns"] - item["ns0"]) / 1e6 for item in inputs if item["k"] == "tq"
    ]
    print(
        f"  input processing (request builds inline): {100 * sum(input_ms) / window_ms:5.1f} % of the window, "
        f"{describe(input_ms)}, {sum(item['n'] for item in inputs if item['k'] == 'tq')} inputs"
    )
    parts = sorted(inputs, key=lambda item: item["ns"])
    part_ends = [item["ns"] for item in parts]
    before_parts = collections.defaultdict(list)
    previous_end = None
    for step in steps:
        if (
            previous_end is not None
            and start_ns <= step["ns"] <= end_ns
            and step_class(step) in ("mixed", "extend")
            and step["path"] == "sync"
        ):
            low = bisect.bisect_left(part_ends, previous_end)
            high = bisect.bisect_right(part_ends, step["ns0"])
            for name, kind in (
                ("result processing", "tp"),
                ("inputs", "tq"),
                ("selection", "tg"),
                ("drain", "td"),
            ):
                before_parts[name].append(
                    sum(
                        (item["ns"] - max(item["ns0"], previous_end)) / 1e6
                        for item in parts[low:high]
                        if item["k"] == kind
                    )
                )
            before_parts["all"].append((step["ns0"] - previous_end) / 1e6)
        else:
            pass
        previous_end = step["ns"]
    if before_parts["all"]:
        print(
            "  before a mixed or extend step: "
            + ", ".join(
                f"{name} {statistics.fmean(values):.2f} ms"
                for name, values in before_parts.items()
            )
        )
    else:
        pass
    print("  mixed and extend periods by extend tokens:")
    for key in sorted(
        by_extend, key=lambda item: (item[0], int(item[1].split("-")[0]))
    ):
        print(f"    {key[0]:7s} {key[1]:>11s} {describe(by_extend[key])}")
    print("  decode periods by rows:")
    for key in sorted(by_rows):
        print(f"    {key[0]:18s} rows {key[1]} {describe(by_rows[key])}")
    if mixed_rows:
        print(
            "  mixed rows: "
            + ", ".join(
                f"{name} {statistics.fmean(values):.2f}"
                for name, values in mixed_rows.items()
            )
        )
    else:
        pass
    fits = collections.defaultdict(list)
    for step in in_window:
        kind = step_class(step)
        step_ms = (step["ns"] - step["ns0"]) / 1e6
        if kind in ("mixed", "extend"):
            prefill_rids = [step["rids"][row] for row, _ in step["pre"]]
            source = (
                "video" if any(rid not in speech for rid in prefill_rids) else "speech"
            )
            fits[f"{kind} with {source} prefill, ms on extend tokens"].append(
                (step["ext"], step_ms)
            )
        elif step["path"] == "sync":
            fits["decode sync, ms on rows"].append((len(step["rids"]), step_ms))
        else:
            pass
    print("  step duration (its start to its stream output) fitted on its size:")
    for name in sorted(fits):
        points = fits[name]
        if len(points) < 3 or len({x for x, _ in points}) < 2:
            print(f"    {name}: {len(points)} steps, too few sizes to fit")
            continue
        else:
            pass
        slope, intercept = statistics.linear_regression(
            [float(x) for x, _ in points], [y for _, y in points]
        )
        sizes = [x for x, _ in points]
        print(
            f"    {name}: {len(points)} steps, {intercept:.2f} ms + {1000 * slope:.3f} ms per 1000, "
            f"sizes {min(sizes)} to {max(sizes)} (median {quantile(sizes, 0.5)})"
        )
    ends = [step["ns"] for step in steps]
    gaps = collections.defaultdict(list)
    gaps_by_extend = collections.defaultdict(list)
    last_send = {}
    for item in sends:
        ns, rid = item[0], item[4]
        previous = last_send.get(rid)
        last_send[rid] = ns
        if previous is None or not start_ns <= ns <= end_ns or ns == previous:
            continue
        else:
            pass
        index = bisect.bisect_left(ends, ns)
        if index == len(steps):
            continue
        else:
            pass
        step = steps[index]
        kind = step_class(step)
        gap_ms = (ns - previous) / 1e6
        gaps[kind].append(gap_ms)
        if kind in ("mixed", "extend"):
            gaps_by_extend[(kind, extend_bucket(step["ext"]))].append(gap_ms)
        else:
            pass
    total = sum(sum(values) for values in gaps.values())
    every_gap = [gap for values in gaps.values() for gap in values]
    print(f"  speech token gaps: {describe(every_gap)}")
    for kind in sorted(gaps):
        print(
            f"    {kind:18s} {100 * sum(gaps[kind]) / total:5.1f} % of gap time  {describe(gaps[kind])}"
        )
    for key in sorted(
        gaps_by_extend, key=lambda item: (item[0], int(item[1].split("-")[0]))
    ):
        values = gaps_by_extend[key]
        print(
            f"    {key[0]:7s} {key[1]:>11s} {100 * sum(values) / total:5.1f} %  {describe(values)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("boots", nargs="+")
    args = parser.parse_args()
    for boot in args.boots:
        report(boot, args.cell)


if __name__ == "__main__":
    main()
