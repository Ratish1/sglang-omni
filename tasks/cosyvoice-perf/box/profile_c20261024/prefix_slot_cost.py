"""What a prefix graph replay costs per tier frame and per row slot, and what each tier design would cost on the steps
a profile replayed (10-24, L6's c1 lean). For each label=report.sqlite:rule (rule l4: tiers of row_tiers(max_rows)
rows times the longest hop, a slot per row; rule l6: the prefill ladder over frames, min(max_rows, frames //
min_hop_frames) slots), every flow.prefix_graph replay's rows, frames, its tier and slots under the rule, and its own
graph device time. Fits device ms = a + b x tier frames + c x slots over every replay of every report (least squares),
prints the fit and the mean error per report, then prices each design on each report's replayed (rows, frames):
l4, l6, l6 plus one row tiers (1 slot for frames up to the longest hop), and the two dimensional ladder (row tiers
by the frame ladder within each row tier's frames, min(rows tier, frames // min_hop_frames) slots).
usage: python3 prefix_slot_cost.py name=<c16.sqlite>:l4 name=<c16.sqlite>:l6 ... [--max-rows 16 --hop-frames 200
       --min-hop-frames 50]"""

import argparse
import bisect
import collections
import re
import sqlite3
import statistics

from sglang.srt.arg_groups.cuda_graph_hook import generate_prefill_cuda_graph_batch_sizes

from cosy_ledger import RANGE_EVENTS, has_table, pid_of

RANGE = re.compile(r"flow\.prefix_graph rows=(\d+) frames=(\d+)")
MS = 1e6


def row_tiers(max_rows):
    return sorted({rows for rows in (1, 2, 4, 8, 12, *range(16, max_rows + 1, 8)) if rows <= max_rows} | {max_rows})


def designs(max_rows, hop, min_hop):
    top = max_rows * hop
    ladder = [f for f in generate_prefill_cuda_graph_batch_sizes(top - 1) if f >= min_hop] + [top]
    l4 = [(rows * hop, rows) for rows in row_tiers(max_rows)]
    l6 = [(f, min(max_rows, f // min_hop)) for f in ladder]
    one_row = sorted(set(l6) | {(f, 1) for f in ladder if f <= hop} | {(hop, 1)})
    two_d = set()
    previous = 0
    for rows in row_tiers(max_rows):
        bound = (previous + 1) * min_hop
        for f in generate_prefill_cuda_graph_batch_sizes(rows * hop - 1):
            if f >= bound:
                two_d.add((f, min(rows, f // min_hop)))
            else:
                pass
        two_d.add((rows * hop, min(rows, rows * hop // min_hop)))
        previous = rows
    return {"l4": sorted(l4), "l6": sorted(l6), "l6+1row": one_row, "2d": sorted(two_d)}


def pick(tiers, rows, frames):
    """The fitting tier with the fewest slots, then the fewest frames."""
    fits = [(slots, f) for f, slots in tiers if f >= frames and slots >= rows]
    if not fits:
        return None
    else:
        slots, f = min(fits)
        return f, slots


def replays(path):
    db = sqlite3.connect(path)
    strings = dict(db.execute("select id, value from StringIds"))
    assert has_table(db, "CUPTI_ACTIVITY_KIND_GRAPH_TRACE")
    graph_work = list(db.execute("select start, end, correlationId, globalPid from CUPTI_ACTIVITY_KIND_GRAPH_TRACE"))
    t0 = min(w[0] for w in graph_work) + int(1e9)
    t1 = max(w[1] for w in graph_work) - int(1e9)
    launches = collections.defaultdict(list)
    for start, tid, corr in db.execute("select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"):
        launches[tid].append((start, corr))
    for tid in launches:
        launches[tid].sort()
    graph_time = {(pid_of(g), c): e - s for s, e, c, g in graph_work}
    out = []
    for start, end, tid, text, text_id in db.execute(
        f"select start, end, globalTid, text, textId from NVTX_EVENTS where eventType in {RANGE_EVENTS} and end is not null"
    ):
        match = RANGE.match((text if text is not None else strings.get(text_id, "")) or "")
        if not match or not t0 <= start <= t1:
            continue
        else:
            pass
        calls = launches[tid]
        a = bisect.bisect_left(calls, (start, -1))
        b = bisect.bisect_right(calls, (end, 1 << 62))
        device = sum(graph_time.get((pid_of(tid), corr), 0) for _, corr in calls[a:b]) / MS
        if device > 0:
            out.append((int(match[1]), int(match[2]), device))
        else:
            pass
    return out


def solve(rows):
    """Least squares for y = a + b x1 + c x2 by the normal equations."""
    n = len(rows)
    sums = collections.defaultdict(float)
    for x1, x2, y in rows:
        for key, value in (("1", 1), ("x1", x1), ("x2", x2), ("x1x1", x1 * x1), ("x2x2", x2 * x2), ("x1x2", x1 * x2),
                           ("y", y), ("x1y", x1 * y), ("x2y", x2 * y)):
            sums[key] += value
    matrix = [[n, sums["x1"], sums["x2"]], [sums["x1"], sums["x1x1"], sums["x1x2"]], [sums["x2"], sums["x1x2"], sums["x2x2"]]]
    vector = [sums["y"], sums["x1y"], sums["x2y"]]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(matrix[r][col]))
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        vector[col], vector[pivot] = vector[pivot], vector[col]
        for r in range(3):
            if r != col:
                factor = matrix[r][col] / matrix[col][col]
                matrix[r] = [x - factor * y for x, y in zip(matrix[r], matrix[col])]
                vector[r] -= factor * vector[col]
            else:
                pass
    return [vector[i] / matrix[i][i] for i in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--max-rows", type=int, default=16)
    parser.add_argument("--hop-frames", type=int, default=200)
    parser.add_argument("--min-hop-frames", type=int, default=50)
    args = parser.parse_args()
    tier_sets = designs(args.max_rows, args.hop_frames, args.min_hop_frames)
    print("designs: " + ", ".join(f"{name} {len(tiers)} graphs" for name, tiers in tier_sets.items()))
    data = {}
    fit_rows = []
    for item in args.reports:
        name, _, rest = item.partition("=")
        path, _, rule = rest.rpartition(":")
        found = []
        for rows, frames, device in replays(path):
            tier = pick(tier_sets[rule], rows, frames)
            if tier is None:
                continue
            else:
                found.append((rows, frames, tier[0], tier[1], device))
                fit_rows.append((tier[0], tier[1], device))
        data[name] = (rule, found)
    a, b, c = solve(fit_rows)
    print(f"fit over {len(fit_rows)} replays: device ms = {a:.3f} + {b * 1000:.3f} x tier frames / 1000 + {c:.3f} x slots")
    for name, (rule, found) in data.items():
        errors = [d - (a + b * f + c * s) for _, _, f, s, d in found]
        by_rows = collections.Counter(min(r, 9) for r, *_ in found)
        print(f"\n{name} ({rule}): {len(found)} replays, measured {sum(d for *_, d in found):.1f} ms, fit error mean "
              f"{statistics.mean(errors):+.3f} ms, rows {dict(sorted(by_rows.items()))}")
        for design, tiers in tier_sets.items():
            priced = [pick(tiers, rows, frames) for rows, frames, *_ in found]
            total = sum(a + b * f + c * s for f, s in priced)
            one_row = [a + b * f + c * s for (rows, *_), (f, s) in zip(found, priced) if rows == 1]
            print(f"  {design:8s} priced {total:8.1f} ms, one row replays mean {statistics.mean(one_row):6.2f} ms"
                  if one_row else f"  {design:8s} priced {total:8.1f} ms")


if __name__ == "__main__":
    main()
