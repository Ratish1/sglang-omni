"""Every row of a stored pair log (serve_pair_c20261017.py's table) that moved the wrong way by more than a threshold,
per cell and metric with both card orders side by side: req/s and completed should rise, everything else fall.
Reads only the logs already copied, so no box run.

usage: python3 pair_regressions_c20261017.py <pair log> [threshold percent, default 1.0]"""

import re
import sys
from collections import defaultdict

HIGHER_IS_BETTER = {"req/s", "completed"}
ROW = re.compile(r"^\| (a|b) \| (\S+) \| ([^|]+?) \| ([^|]*) \| ([^|]*) \| ([+-][0-9.]+) % \|")

threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
changes: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
for line in open(sys.argv[1]):
    if match := ROW.match(line):
        order, cell, metric, _, _, change = match.groups()
        changes[(cell, metric)][order] = float(change)
for (cell, metric), by_order in sorted(changes.items()):
    worse = [
        order
        for order, change in by_order.items()
        if (change < -threshold if metric in HIGHER_IS_BETTER else change > threshold)
    ]
    if worse:
        both = " ; ".join(f"{by_order.get(order, float('nan')):+.1f} %" for order in ("a", "b"))
        print(f"{cell:4} {metric:18} {both:18} worse in order {', '.join(sorted(worse))}")
