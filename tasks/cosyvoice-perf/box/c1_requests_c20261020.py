"""Each seeded request of a pair's cell against itself in the other arm, both orders (order a: main gpu0, the branch
gpu1; order b swapped). Per metric: the branch minus main per request (median, how many slower), the requests slower
by over 1 ms in both orders, main against itself and the branch against itself across the orders (the per request
spread), the p95 and p99 of each arm, and each arm's three slowest requests with the other arm's time beside them.
usage: python3 c1_requests_c20261020.py <pair dir> <cell>"""

import json
import statistics
import sys


def load(pair_dir, order, gpu, cell):
    with open(f"{pair_dir}/{order}/{gpu}/{cell}/speed_results.json") as handle:
        records = json.load(handle)["per_request"]
    return {record["id"]: record for record in records if record["is_success"]}


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def absolute_p95(values):
    return percentile([abs(value) for value in values], 0.95)


METRICS = (
    ("first audio", lambda record: record["audio_ttfp_s"]),
    ("inter chunk mean", lambda record: statistics.mean(record["inter_chunk_s"]) if record["inter_chunk_s"] else None),
    ("inter chunk max", lambda record: max(record["inter_chunk_s"]) if record["inter_chunk_s"] else None),
    ("latency", lambda record: record["latency_s"]),
)

pair_dir, cell = sys.argv[1], sys.argv[2]
main_a, branch_a = load(pair_dir, "a", "gpu0", cell), load(pair_dir, "a", "gpu1", cell)
main_b, branch_b = load(pair_dir, "b", "gpu1", cell), load(pair_dir, "b", "gpu0", cell)
ids = [request_id for request_id in main_a if request_id in branch_a and request_id in main_b and request_id in branch_b]
same_tokens = [
    sum(main[i]["completion_tokens"] == branch[i]["completion_tokens"] for i in ids)
    for main, branch in ((main_a, branch_a), (main_b, branch_b))
]
print(f"{cell}: {len(ids)} requests in all four runs, same completion tokens {same_tokens[0]} ; {same_tokens[1]}")
for name, value in METRICS:
    rows = [(i, value(main_a[i]), value(branch_a[i]), value(main_b[i]), value(branch_b[i])) for i in ids]
    rows = [row for row in rows if row[1] is not None]
    delta_a = [(row[2] - row[1]) * 1e3 for row in rows]
    delta_b = [(row[4] - row[3]) * 1e3 for row in rows]
    slower_both = [(row[0], round(a, 1), round(b, 1)) for row, a, b in zip(rows, delta_a, delta_b) if a > 1 and b > 1]
    print(f"== {name}, {len(rows)} requests")
    print(f"  branch - main median {statistics.median(delta_a):+.2f} ; {statistics.median(delta_b):+.2f} ms,"
          f" mean {statistics.mean(delta_a):+.2f} ; {statistics.mean(delta_b):+.2f},"
          f" branch slower in {sum(d > 0 for d in delta_a)} ; {sum(d > 0 for d in delta_b)}")
    print(f"  slower by over 1 ms in both orders: {len(slower_both)} {slower_both[:10]}")
    print(f"  |branch - main| p95 {absolute_p95(delta_a):.1f} ; {absolute_p95(delta_b):.1f} ms;"
          f" |main b - main a| p95 {absolute_p95([(row[3] - row[1]) * 1e3 for row in rows]):.1f},"
          f" |branch b - branch a| p95 {absolute_p95([(row[4] - row[2]) * 1e3 for row in rows]):.1f}")
    for order, main_index, branch_index in (("a", 1, 2), ("b", 3, 4)):
        main_values = [row[main_index] * 1e3 for row in rows]
        branch_values = [row[branch_index] * 1e3 for row in rows]
        print(f"  order {order}: p95 {percentile(main_values, 0.95):.1f} -> {percentile(branch_values, 0.95):.1f},"
              f" p99 {percentile(main_values, 0.99):.1f} -> {percentile(branch_values, 0.99):.1f} ms;"
              f" main's slowest {[(row[0][-8:], round(row[main_index] * 1e3, 1), round(row[branch_index] * 1e3, 1)) for row in sorted(rows, key=lambda row: -row[main_index])[:3]]};"
              f" branch's slowest {[(row[0][-8:], round(row[main_index] * 1e3, 1), round(row[branch_index] * 1e3, 1)) for row in sorted(rows, key=lambda row: -row[branch_index])[:3]]}")
