#!/usr/bin/env python3
"""Synchronizing CUDA calls inside one cell's window by thread and site, from hop probe "sync" records (boots run with
--probe and OMNI_HOP_PROBE_SYNC_SITES=1; each record holds one process's running counts).

usage: python3 sync_window.py <boot dir> [--cell c64_events] [--thread scheduler] [--top 25]

Per process file: the counts at the last record inside the window minus those at the last record before it, for sites
whose thread name holds --thread, as counts and counts a second.
"""

import argparse
import glob
import json
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("boot")
    parser.add_argument("--cell", default="c64_events")
    parser.add_argument("--thread", default="scheduler")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    with open(
        os.path.join(args.boot, args.cell, "window.json"), encoding="utf-8"
    ) as source:
        window = json.load(source)
    start_ns, end_ns = window["start"] * 1e9, window["end"] * 1e9
    seconds = window["end"] - window["start"]
    for path in sorted(glob.glob(os.path.join(args.boot, "probe", "*.jsonl"))):
        before, inside = {}, {}
        with open(path, encoding="utf-8") as source:
            for line in source:
                if '"k": "sync"' not in line:
                    continue
                else:
                    pass
                item = json.loads(line)
                if item["ns"] < start_ns:
                    before = item["counts"]
                elif item["ns"] <= end_ns:
                    inside = item["counts"]
                else:
                    pass
        delta = {
            site: count - before.get(site, 0)
            for site, count in inside.items()
            if args.thread in site.split(" | ")[0] and count > before.get(site, 0)
        }
        if not delta:
            continue
        else:
            pass
        print(
            f"{os.path.basename(path)} {args.cell}: {sum(delta.values())} syncs in {seconds:.1f} s"
        )
        for site, count in sorted(delta.items(), key=lambda pair: -pair[1])[: args.top]:
            print(f"  {count:7d} {count / seconds:7.1f}/s  {site}")


if __name__ == "__main__":
    main()
