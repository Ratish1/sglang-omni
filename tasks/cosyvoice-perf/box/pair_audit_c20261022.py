"""Audit of every same time pair of one run home, from each pair's report table and files:
- per pair, the metric rows worse in both card orders, better in both, and mixed (req/s higher is better, every other
  row lower), with the A/A pairs (both arms one tree name) apart, so the rule "worse in both orders is a regression"
  is read against pairs that change nothing; then, per metric over the A/A pairs, each pair's mean of its two orders;
- each pair's trees by commit, and whether the first arm's commit is an ancestor of the second's (an unmatched base
  when not);
- each cell's window on the box clock (UTC), from the speed results' write times, against the given windows of other
  tenants' work on the same NUMA node.
usage: python3 pair_audit_c20261022.py <run home> <repo> [HH:MM:SS-HH:MM:SS ...]"""

import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

COUNT_ROWS = {"completed", "failed", "at the token cap"}
ROW = re.compile(r"^\| ([ab]) \| (\S+) \| ([^|]+?) \| ([^|]*?) \| ([^|]*?) \| [^|]*\|$")
HEADER = re.compile(r"^\| order \| cell \| metric \| ([^|]+?) \| ([^|]+?) \| change \|$")


def seconds(clock):
    hours, minutes, secs = (int(part) for part in clock.split(":"))
    return 3600 * hours + 60 * minutes + secs


def clock(epoch):
    return time.strftime("%H:%M:%S", time.gmtime(epoch))


def read_table(log):
    names, rows, cells = None, {}, []
    for line in log.read_text(errors="replace").splitlines():
        header = HEADER.match(line)
        if header:
            names = header.groups()
            continue
        row = ROW.match(line)
        if not row or row.group(3) in COUNT_ROWS:
            continue
        order, cell, metric, first, second = row.groups()
        try:
            first, second = float(first), float(second)
        except ValueError:
            continue
        if cell not in cells:
            cells.append(cell)
        if first:
            rows[(order, cell, metric)] = 100 * (second - first) / first
    return names, rows, cells


def worse(metric, change):
    return change < 0 if metric == "req/s" else change > 0


def commit(pair_dir, order, gpu):
    path = pair_dir / order / f"identity_gpu{gpu}.txt"
    return path.read_text().split()[0] if path.exists() else None


def windows(pair_dir, cells):
    spans = {}
    for order in ("a", "b"):
        identity = pair_dir / order / "identity_gpu0.txt"
        if not identity.exists():
            continue
        start = identity.stat().st_mtime
        for cell in cells:
            ends = [
                path.stat().st_mtime
                for path in (pair_dir / order / f"gpu{gpu}" / cell / "speed_results.json" for gpu in (0, 1))
                if path.exists()
            ]
            if not ends:
                continue
            spans[(order, cell)] = (start, max(ends))
            start = max(ends)
    return spans


def main():
    home, repo = Path(sys.argv[1]), sys.argv[2]
    others = [tuple(seconds(edge) for edge in window.split("-")) for window in sys.argv[3:]]
    aa_changes = {}
    print("| pair | arms | first is ancestor of second | rows | worse in both | better in both | mixed | worse in both (rows) |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    overlap_lines = []
    for log in sorted(home.glob("p_*.log")):
        pair_dir = home / log.stem
        names, rows, cells = read_table(log)
        if names is None or not rows:
            continue
        keys = sorted({(cell, metric) for _, cell, metric in rows})
        both_worse, both_better, mixed = [], [], []
        for cell, metric in keys:
            if ("a", cell, metric) not in rows or ("b", cell, metric) not in rows:
                continue
            change_a, change_b = rows[("a", cell, metric)], rows[("b", cell, metric)]
            if worse(metric, change_a) and worse(metric, change_b):
                both_worse.append(f"{cell} {metric} {change_a:+.1f};{change_b:+.1f}")
            elif not worse(metric, change_a) and not worse(metric, change_b):
                both_better.append((cell, metric))
            else:
                mixed.append((cell, metric))
            if names[0] == names[1]:
                aa_changes.setdefault((cell, metric), []).append((log.stem, (change_a + change_b) / 2))
        first, second = commit(pair_dir, "a", 0), commit(pair_dir, "a", 1)
        ancestor = ""
        if first and second:
            ancestor = (
                "same" if first == second else
                "yes" if subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", first, second]).returncode == 0
                else "NO"
            )
        total = len(both_worse) + len(both_better) + len(mixed)
        print(
            f"| {log.stem} | {names[0]} {(first or '')[:9]} / {names[1]} {(second or '')[:9]} | {ancestor} | {total} | "
            f"{len(both_worse)} | {len(both_better)} | {len(mixed)} | {', '.join(both_worse)} |"
        )
        for (order, cell), (start, end) in sorted(windows(pair_dir, cells).items(), key=lambda item: item[1][0]):
            day_start, day_end = start % 86400, end % 86400
            hits = [f"{clock(lo)}-{clock(hi)}" for lo, hi in others if lo < day_end and hi > day_start]
            overlap_lines.append(f"| {log.stem} | {order} | {cell} | {clock(start)} | {clock(end)} | {', '.join(hits) or 'none'} |")
    print()
    print("A/A pairs, each pair's mean of its two orders (%), per metric:")
    print("| cell | metric | pairs | values | mean | sd |")
    print("| --- | --- | --- | --- | --- | --- |")
    for (cell, metric), values in sorted(aa_changes.items()):
        changes = [value for _, value in values]
        spread = statistics.stdev(changes) if len(changes) > 1 else 0.0
        print(
            f"| {cell} | {metric} | {len(changes)} | {' '.join(f'{value:+.1f}' for value in changes)} | "
            f"{statistics.mean(changes):+.2f} | {spread:.2f} |"
        )
    print()
    print("Cell windows (UTC; boot included in each order's first cell) against other tenants' windows:")
    print("| pair | order | cell | start | end | overlapping windows |")
    print("| --- | --- | --- | --- | --- | --- |")
    for line in overlap_lines:
        print(line)


if __name__ == "__main__":
    main()
