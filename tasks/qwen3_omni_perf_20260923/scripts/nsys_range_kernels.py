"""Kernels inside a process's runner/execute ranges of the given modes, grouped by kernel
name: count and device ms per range. Stage is the thinker unless --stage talker_ar.
usage: python nsys_range_kernels.py REPORT.sqlite --modes EXTEND,MIXED [--top 25]"""

import argparse
import bisect
import json
import sqlite3
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("sqlite")
ap.add_argument("--modes", default="EXTEND,MIXED")
ap.add_argument("--top", type=int, default=25)
ap.add_argument("--stage", default="thinker")
a = ap.parse_args()
db = sqlite3.connect(a.sqlite)
kmin, kmax = db.execute(
    "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
).fetchone()
span = kmax - kmin
lo, hi = kmin + int(span * 0.15), kmax - int(span * 0.05)
names = {i: v for i, v in db.execute("select id, value from StringIds")}
kern = {}
for corr, start, end, sn in db.execute(
    "select correlationId, start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ?",
    (lo, hi),
):
    kern.setdefault(corr, []).append(((end - start) / 1e6, names.get(sn, "?")))
runtime = defaultdict(list)
for start, gtid, corr in db.execute(
    "select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?",
    (lo, hi),
):
    runtime[gtid].append((start, corr))
for g in runtime:
    runtime[g].sort()
starts = {g: [r[0] for r in rs] for g, rs in runtime.items()}
talker_pids = set()
for (pid,) in db.execute(
    """select distinct e.globalTid >> 24 from NVTX_EVENTS e left join StringIds s on e.textId = s.id
    where coalesce(e.text, s.value) like '%"stage":"talker"%'"""
):
    talker_pids.add(pid)
modes = set(a.modes.split(","))
agg = defaultdict(lambda: [0, 0.0])
nranges = 0
total = 0.0
for start, end, gtid, text in db.execute(
    """select e.start, e.end, e.globalTid, coalesce(e.text, s.value) from NVTX_EVENTS e
    left join StringIds s on e.textId = s.id where e.end is not null and e.start >= ? and e.start < ? and coalesce(e.text, s.value) like '%"op":"execute"%'""",
    (lo, hi),
):
    meta = json.loads(text.split(":", 1)[1])
    pid = gtid >> 24
    stage = "talker_ar" if pid in talker_pids else "thinker"
    if stage != a.stage or meta.get("mode") not in modes:
        continue
    nranges += 1
    rs = runtime.get(gtid, [])
    i = bisect.bisect_left(starts.get(gtid, []), start)
    while i < len(rs) and rs[i][0] < end:
        for ms, name in kern.get(rs[i][1], []):
            agg[name][0] += 1
            agg[name][1] += ms
            total += ms
        i += 1
print(
    f"{a.stage} {a.modes}: {nranges} ranges, {total/nranges:.2f} device ms per range, {sum(v[0] for v in agg.values())/nranges:.0f} kernels per range"
)
for name, (n, ms) in sorted(agg.items(), key=lambda kv: -kv[1][1])[: a.top]:
    print(f"{ms/nranges:8.3f} ms/range {n/nranges:7.1f} /range  {name[:90]}")
