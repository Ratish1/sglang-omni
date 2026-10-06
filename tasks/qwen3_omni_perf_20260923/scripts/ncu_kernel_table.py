"""Per-kernel table of an ncu --csv log (one row per kernel name and grid): calls and the mean of each metric.

usage: python ncu_kernel_table.py NCU.csv [rows]
"""

import collections
import csv
import statistics
import sys

with open(sys.argv[1]) as handle:
    rows = [r for r in csv.reader(handle) if r and not r[0].startswith("==")]
head = rows[0]
idx = {k: i for i, k in enumerate(head)}
per = collections.defaultdict(dict)
meta = {}
for r in rows[1:]:
    kid = r[idx["ID"]]
    name = r[idx["Kernel Name"]]
    meta[kid] = (name[:48], r[idx["Grid Size"]], r[idx["Block Size"]])
    try:
        per[kid][r[idx["Metric Name"]]] = float(r[idx["Metric Value"]].replace(",", ""))
    except ValueError:
        pass
groups = collections.defaultdict(list)
for kid, m in per.items():
    groups[meta[kid]].append(m)
cols = [
    ("gpu__time_duration.sum", "us", 1e-3),
    ("sm__inst_issued.avg.pct_of_peak_sustained_active", "issue%act", 1),
    ("sm__inst_issued.avg.pct_of_peak_sustained_elapsed", "issue%el", 1),
    ("sm__cycles_active.avg.pct_of_peak_sustained_elapsed", "SMact%", 1),
    ("sm__warps_active.avg.pct_of_peak_sustained_active", "occ%", 1),
    ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "dram%", 1),
    (
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
        "tensor%",
        1,
    ),
]
print(
    f"{'kernel':48s} {'grid':>14s} {'n':>5s} " + " ".join(f"{c[1]:>9s}" for c in cols)
)
for key, ms in sorted(
    groups.items(),
    key=lambda kv: -statistics.fmean(m.get("gpu__time_duration.sum", 0) for m in kv[1])
    * len(kv[1]),
)[: int(sys.argv[2]) if len(sys.argv) > 2 else 30]:
    vals = []
    for metric, _, scale in cols:
        v = [m[metric] for m in ms if metric in m]
        vals.append(f"{statistics.fmean(v) * scale:9.1f}" if v else f"{'-':>9s}")
    print(f"{key[0]:48s} {key[1][:14]:>14s} {len(ms):5d} " + " ".join(vals))
