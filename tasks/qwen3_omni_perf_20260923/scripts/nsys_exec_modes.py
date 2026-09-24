"""Per-process runner/execute ranges of an Nsight sqlite export grouped by forward mode
and batch size: ranges, host ms, device ms of the kernels launched inside them (through
the runtime rows on the range's thread, graph node kernels through their launch), tokens.

usage: python nsys_exec_modes.py REPORT.sqlite [--window window.txt]
"""

from __future__ import annotations

import argparse
import bisect
import json
import sqlite3
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--window")
    args = ap.parse_args()
    db = sqlite3.connect(args.sqlite)

    kmin, kmax = db.execute(
        "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    span = kmax - kmin
    lo, hi = kmin + int(span * 0.15), kmax - int(span * 0.05)
    print(f"window: middle of the kernel span, {(hi - lo) / 1e9:.1f} s")

    kern = defaultdict(float)
    for corr, start, end in db.execute(
        "select correlationId, start, end from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ?",
        (lo, hi),
    ):
        kern[corr] += (end - start) / 1e6

    runtime = defaultdict(list)
    for start, gtid, corr in db.execute(
        "select start, globalTid, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and start < ?",
        (lo, hi),
    ):
        runtime[gtid].append((start, corr))
    for gtid in runtime:
        runtime[gtid].sort()
    rt_starts = {gtid: [r[0] for r in rs] for gtid, rs in runtime.items()}

    stage_of_pid = {}
    for pid, text in db.execute(
        """select e.globalTid >> 24, coalesce(e.text, s.value) from NVTX_EVENTS e
           left join StringIds s on e.textId = s.id
           where coalesce(e.text, s.value) like '%"stage":"talker"%' or coalesce(e.text, s.value) like '%"stage":"code2wav"%'"""
    ):
        stage_of_pid[pid] = "talker_ar"

    groups = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for start, end, gtid, text in db.execute(
        """select e.start, e.end, e.globalTid, coalesce(e.text, s.value) from NVTX_EVENTS e
           left join StringIds s on e.textId = s.id
           where e.end is not null and e.start >= ? and e.start < ?
           and coalesce(e.text, s.value) like '%"op":"execute"%'""",
        (lo, hi),
    ):
        meta = json.loads(text.split(":", 1)[1])
        pid = gtid >> 24
        stage = stage_of_pid.get(pid, "thinker")
        mode = meta.get("mode", "?")
        bs = int(meta.get("batch_size", 0))
        key = (stage, mode, bs)
        g = groups[key]
        g[0] += 1
        g[1] += (end - start) / 1e6
        rs = runtime.get(gtid, [])
        i = bisect.bisect_left(rt_starts.get(gtid, []), start)
        dev = 0.0
        while i < len(rs) and rs[i][0] < end:
            dev += kern.get(rs[i][1], 0.0)
            i += 1
        g[2] += dev
        g[3] += bs

    print(
        f"{'stage':<10} {'mode':<8} {'bs':>3} {'ranges':>7} {'host ms':>9} {'host/rng':>9} {'dev ms':>9} {'dev/rng':>8}"
    )
    totals = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for (stage, mode, bs), (n, host, dev, toks) in sorted(groups.items()):
        print(
            f"{stage:<10} {mode:<8} {bs:>3} {n:>7} {host:>9.0f} {host/n:>9.2f} {dev:>9.0f} {dev/n:>8.2f}"
        )
        t = totals[(stage, mode)]
        t[0] += n
        t[1] += host
        t[2] += dev
        t[3] += toks
    print()
    for (stage, mode), (n, host, dev, toks) in sorted(totals.items()):
        print(
            f"{stage:<10} {mode:<8} all {n:>7} host {host:>8.0f} ms dev {dev:>8.0f} ms  mean bs {toks/n:>5.2f}  dev/range {dev/n:>6.2f} ms  dev/token {dev/max(toks,1):>6.3f} ms"
        )


if __name__ == "__main__":
    main()
