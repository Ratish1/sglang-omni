#!/usr/bin/env python3
"""Where a synchronous thinker step's time goes, by forward mode, rows and whether it carries a video request's row,
from hop probe "tx" records inside one events cell's window (boots run with OMNI_HOP_PROBE_THINKER_PARTS=1 and
OMNI_HOP_PROBE_THINKER_STEPS=1).

usage: python3 thinker_parts.py --cell c64_events <boot> [<boot> ...]

Per mode, row bucket and media (a step carries "video" when one of its rows is a request the thinker does not stream to
the talker, the looping video client's text requests), means in ms: build_forward_batch, prepare_and_forward (host
time to launch the forward), finalize (host token wait included), the rest of execute (the post_prefill or post_decode
hook and token publication), the whole execute, and the forward's device span (CUDA events around prepare_and_forward).
"""

import argparse
import bisect
import collections
import glob
import json
import os
import statistics

ROW_BUCKETS = ((1, 4), (5, 8), (9, 16), (17, 64))


def report(boot: str, cell: str) -> None:
    with open(os.path.join(boot, cell, "window.json"), encoding="utf-8") as source:
        window = json.load(source)
    start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
    parts, steps, speech = [], [], set()
    for path in glob.glob(os.path.join(boot, "probe", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                item = json.loads(line)
                if isinstance(item, list):
                    if (
                        item[1] == "thinker.out"
                        and item[2] == "put"
                        and item[6] == "talker_ar"
                    ):
                        speech.add(item[4])
                    else:
                        pass
                elif item.get("k") == "tx" and start_ns <= item["ns"] <= end_ns:
                    parts.append(item)
                elif item.get("k") == "ts":
                    steps.append(item)
                else:
                    pass
    steps.sort(key=lambda item: item["ns0"])
    starts = [item["ns0"] for item in steps]
    groups = collections.defaultdict(list)
    for item in parts:
        index = bisect.bisect_right(starts, item["ns0"]) - 1
        step = steps[index] if index >= 0 and steps[index]["ns"] >= item["ns"] else None
        media = (
            "unknown"
            if step is None
            else (
                "video" if any(rid not in speech for rid in step["rids"]) else "speech"
            )
        )
        bucket = next(
            (
                f"{low:2d}-{high:2d}"
                for low, high in ROW_BUCKETS
                if low <= item["rows"] <= high
            ),
            "more",
        )
        if "graph" in item:
            media = (
                f"{media}/{'graph' if item['graph'] else 'eager'}/"
                f"{'long' if item['long'] else 'short'}"
            )
        else:
            pass
        groups[(item["mode"], media, bucket)].append(item)
    print(f"{boot}:")
    print(
        "  mode     media             rows      n   build  launch finalize   rest  execute  device"
    )
    for (mode, media, bucket), items in sorted(groups.items()):
        total = [(item["ns"] - item["ns0"]) / 1e6 for item in items]
        build = [item.get("build_forward_batch", 0.0) for item in items]
        launch = [item.get("prepare_and_forward", 0.0) for item in items]
        finalize = [item.get("finalize", 0.0) for item in items]
        rest = [t - b - l - f for t, b, l, f in zip(total, build, launch, finalize)]
        device = [item["gpu"] for item in items if "gpu" in item]
        print(
            f"  {mode:8s} {media:17s} {bucket:5s} {len(items):6d} {statistics.fmean(build):7.2f} "
            f"{statistics.fmean(launch):7.2f} {statistics.fmean(finalize):8.2f} {statistics.fmean(rest):6.2f} "
            f"{statistics.fmean(total):8.2f} {statistics.fmean(device) if device else float('nan'):7.2f}"
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
