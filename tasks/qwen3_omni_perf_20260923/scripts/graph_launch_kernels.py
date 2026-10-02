"""Per-kernel device time inside one process's CUDA graph launches, from a node-traced nsys report.

Node kernels carry the correlation id of their cudaGraphLaunch, so the kernels of one launch are
one group. The process is the one that ran a kernel whose name contains --marker; launches with
at least --min-kernels kernels are kept (the talker decode graph runs about a thousand). Prints,
per launch, the span (first start to last end) against the kernel sum, then every kernel name's
count and device microseconds per launch, largest first.
usage: python graph_launch_kernels.py REPORT.sqlite [--marker norm_qkv_rope_store] [--min-kernels 900]
"""

import argparse
import sqlite3
import statistics
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("sqlite")
parser.add_argument("--marker", default="norm_qkv_rope_store")
parser.add_argument("--min-kernels", type=int, default=900)
parser.add_argument("--top", type=int, default=40)
args = parser.parse_args()

db = sqlite3.connect(args.sqlite)
names = dict(db.execute("select id, value from StringIds"))
marker_ids = [i for i, value in names.items() if args.marker in value]
(process,) = db.execute(
    f"select globalPid from CUPTI_ACTIVITY_KIND_KERNEL where shortName in ({','.join('?' * len(marker_ids))}) limit 1",
    marker_ids,
).fetchone()

launches: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
for correlation, start, end, short_name in db.execute(
    "select correlationId, start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL "
    "where globalPid = ? and graphNodeId is not null",
    (process,),
):
    launches[correlation].append((start, end, short_name))
decode = {c: k for c, k in launches.items() if len(k) >= args.min_kernels}
print(
    f"process {process}: {len(launches)} graph launches, {len(decode)} with >= {args.min_kernels} kernels"
)

spans, sums, counts = [], [], []
per_name_durations: dict[str, list[float]] = defaultdict(list)
for kernels in decode.values():
    spans.append((max(e for _, e, _ in kernels) - min(s for s, _, _ in kernels)) / 1e3)
    sums.append(sum(e - s for s, e, _ in kernels) / 1e3)
    counts.append(len(kernels))
    for start, end, short_name in kernels:
        per_name_durations[names.get(short_name, "?")].append((end - start) / 1e3)
launch_count = len(decode)
print(
    f"per launch: kernels p50 {statistics.median(counts)}, span p50 {statistics.median(spans):.0f} us, "
    f"kernel sum p50 {statistics.median(sums):.0f} us, gap share {1 - statistics.median(sums) / statistics.median(spans):.1%}"
)
# note (ratish): a kernel in flight while another context is resident is stretched by the
# switched-out time, so the median duration stands for the kernel's own cost.
rows = []
for name, durations in per_name_durations.items():
    per_launch = len(durations) / launch_count
    rows.append(
        (
            statistics.median(durations) * per_launch,
            sum(durations) / launch_count,
            per_launch,
            statistics.median(durations),
            name,
        )
    )
total = sum(row[0] for row in rows)
print(f"median-based kernel sum per launch {total:.0f} us")
print(
    f"{'us/launch':>10} {'share':>6} {'mean us/l':>9} {'n/launch':>8} {'us p50':>7}  kernel"
)
for median_us, mean_us, per_launch, median_each, name in sorted(
    rows, key=lambda row: -row[0]
)[: args.top]:
    print(
        f"{median_us:10.1f} {median_us / total:6.1%} {mean_us:9.1f} {per_launch:8.1f} {median_each:7.2f}  {name[:100]}"
    )
