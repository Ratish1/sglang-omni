"""Pick the split sampler's launch from sampler_split_sweep.py results.

Reads one or more sweep JSON files (devices, top_p settings) and prints, per width, the
split configurations ranked by their worst ratio to the best configuration of each
(file, batch) cell, with their worst ratio to the served single kernel; then the ranking
of one configuration shared by every width where the split is valid.

usage: python sampler_sweep_pick.py [--block-ks 64,128] FILE [FILE ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict

SERVED_SINGLE = ("single", 1, 8, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-ks")
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()
    cells = defaultdict(dict)
    for path in args.files:
        with open(path) as handle:
            results = json.load(handle)
        for cell in results["cells"]:
            if args.block_ks and str(cell["block_k"]) not in args.block_ks.split(","):
                continue
            else:
                pass
            key = (path, cell["block_k"], cell["rows"])
            cells[key][tuple(cell["config"])] = statistics.median(cell["us"])

    by_width = defaultdict(list)
    for key in cells:
        by_width[key[1]].append(key)

    shared = defaultdict(lambda: (0.0, 0.0))
    widths = sorted(by_width)
    for block_k in widths:
        keys = by_width[block_k]
        configs = set.intersection(*(set(cells[key]) for key in keys))
        split = [config for config in configs if config[0] == "split"]
        regret = {}
        against_served = {}
        for config in split:
            regret[config] = max(
                cells[key][config] / min(cells[key].values()) for key in keys
            )
            against_served[config] = max(
                cells[key][config] / cells[key][SERVED_SINGLE] for key in keys
            )
            worst_regret, worst_served = shared[config]
            shared[config] = (
                max(worst_regret, regret[config]),
                max(worst_served, against_served[config]),
            )
        ranked = sorted(split, key=lambda config: regret[config])
        print(f"block_k {block_k}: {len(keys)} cells, {len(split)} split configs")
        print("  regret  vs served  config")
        for config in ranked[:8]:
            print(f"  {regret[config]:6.3f}  {against_served[config]:9.3f}  {config}")

    valid_everywhere = [
        config
        for config in shared
        if all(config in cells[key] for block_k in widths for key in by_width[block_k])
    ]
    print(f"one configuration for widths {widths}:")
    print("  regret  vs served  config")
    for config in sorted(valid_everywhere, key=lambda config: shared[config][0])[:8]:
        print(f"  {shared[config][0]:6.3f}  {shared[config][1]:9.3f}  {config}")


if __name__ == "__main__":
    main()
