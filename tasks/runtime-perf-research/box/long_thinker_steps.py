#!/usr/bin/env python3
"""Thinker steps over a length per pass, from hop probe "ts" records (boots run with OMNI_HOP_PROBE_THINKER_STEPS=1).

usage: python3 long_thinker_steps.py RUN_DIR BOOT [BOOT ...] [--over-ms 30]

Per boot and pass (each cell directory with a window.json): steps in the window, steps over the length and their
summed time, and the long steps by mode.
"""

import argparse
import collections
import glob
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("boots", nargs="+")
    parser.add_argument("--over-ms", type=float, default=30.0)
    args = parser.parse_args()
    for boot in args.boots:
        steps = []
        for path in glob.glob(os.path.join(args.run_dir, boot, "probe", "*.jsonl")):
            with open(path, encoding="utf-8") as source:
                for line in source:
                    if '"k": "ts"' in line:
                        steps.append(json.loads(line))
                    else:
                        pass
        for window_path in sorted(
            glob.glob(os.path.join(args.run_dir, boot, "c*_*", "window.json"))
        ):
            with open(window_path, encoding="utf-8") as source:
                window = json.load(source)
            start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
            inside = [
                step
                for step in steps
                if start_ns <= step["ns0"] and step["ns"] <= end_ns
            ]
            long_steps = [
                step
                for step in inside
                if (step["ns"] - step["ns0"]) / 1e6 > args.over_ms
            ]
            long_ms = sum((step["ns"] - step["ns0"]) / 1e6 for step in long_steps)
            modes = collections.Counter(step["mode"] for step in long_steps)
            cell = os.path.basename(os.path.dirname(window_path))
            print(
                f"{boot} {cell}: steps {len(inside)}, over {args.over_ms:.0f} ms {len(long_steps)} "
                f"({long_ms / 1e3:.2f} s), by mode {dict(modes)}"
            )


if __name__ == "__main__":
    main()
