"""Memcpy census of one process in an Nsight Systems sqlite export.

Every cudaMemcpyAsync / cudaMemcpy runtime row of the chosen process is joined by
correlation id to its CUPTI memcpy record, then grouped by calling thread, copy kind,
source and destination memory kind, byte bucket and the innermost NVTX range on that
thread that encloses the call. Host time is the runtime row's duration.

usage: python nsys_memcpy_census.py REPORT.sqlite --pid-match thinker [--window window.txt]
       [--top 40]

--pid-match picks the process whose NVTX texts contain the string (the pipeline marks
carry the stage name); --pid takes a globalPid directly.
"""

from __future__ import annotations

import argparse
import bisect
import sqlite3
from collections import defaultdict


def has_table(db, name):
    row = db.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


def enum_names(db, table):
    if not has_table(db, table):
        return {}
    return {row[0]: row[1] for row in db.execute(f"select id, name from {table}")}


def byte_bucket(n):
    if n <= 64:
        return "<=64B"
    if n <= 1024:
        return "<=1KiB"
    if n <= 65536:
        return "<=64KiB"
    if n <= 1 << 20:
        return "<=1MiB"
    return ">1MiB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--pid", type=int)
    ap.add_argument("--pid-match")
    ap.add_argument("--window")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    db = sqlite3.connect(args.sqlite)

    lo, hi = None, None
    if args.window:
        with open(args.window) as f:
            t0, t1 = [float(x) for x in f.read().split()[:2]]
        # window.txt is wall clock seconds; nsys start/end are ns since session start,
        # so cut by the kernel span instead when the offsets are unknown
        kmin, kmax = db.execute(
            "select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()
        span = kmax - kmin
        lo = kmin + int(span * 0.15)
        hi = kmax - int(span * 0.05)
        print(
            f"window: middle of the kernel span, {lo} .. {hi} ns ({(hi-lo)/1e9:.1f} s)"
        )

    pid = args.pid
    if pid is None:
        rows = db.execute(
            """select e.globalTid >> 24, count(*) from NVTX_EVENTS e
               left join StringIds s on e.textId = s.id where coalesce(e.text, s.value) like ?
               group by 1 order by 2 desc""",
            (f"%{args.pid_match}%",),
        ).fetchall()
        if not rows:
            raise SystemExit(f"no NVTX text contains {args.pid_match!r}")
        pid = rows[0][0]
        print(
            f"process {pid} selected by NVTX text {args.pid_match!r} ({rows[0][1]} marks)"
        )

    copy_kind = enum_names(db, "ENUM_CUDA_MEMCPY_OPER")
    mem_kind = enum_names(db, "ENUM_CUDA_MEM_KIND")
    thread_names = {}
    if has_table(db, "ThreadNames"):
        for gtid, name in db.execute(
            "select t.globalTid, s.value from ThreadNames t join StringIds s on t.nameId = s.id"
        ):
            thread_names[gtid] = name

    where = ""
    params = []
    if lo is not None:
        where = " and r.start >= ? and r.start < ?"
        params = [lo, hi]
    runtime = db.execute(
        f"""select r.start, r.end, r.globalTid, r.correlationId, s.value
            from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId = s.id
            where (r.globalTid >> 24) = ? and s.value like 'cudaMemcpy%'{where}""",
        [pid] + params,
    ).fetchall()
    memcpy = {}
    for corr, ck, sk, dk, nbytes, mstart, mend in db.execute(
        """select correlationId, copyKind, srcKind, dstKind, bytes, start, end
           from CUPTI_ACTIVITY_KIND_MEMCPY where (globalPid >> 24) = ?""",
        (pid,),
    ):
        memcpy[corr] = (ck, sk, dk, nbytes, mstart, mend)

    ranges = defaultdict(list)
    for start, end, gtid, text in db.execute(
        """select e.start, e.end, e.globalTid, coalesce(e.text, s.value) from NVTX_EVENTS e
           left join StringIds s on e.textId = s.id
           where (e.globalTid >> 24) = ? and e.end is not null""",
        (pid,),
    ):
        ranges[gtid].append((start, end, text or "?"))
    for gtid in ranges:
        ranges[gtid].sort()
    starts = {gtid: [r[0] for r in rs] for gtid, rs in ranges.items()}

    def enclosing(gtid, t):
        rs = ranges.get(gtid)
        if not rs:
            return "-"
        i = bisect.bisect_right(starts[gtid], t) - 1
        best = None
        while i >= 0 and i > bisect.bisect_right(starts[gtid], t) - 64:
            s, e, text = rs[i]
            if s <= t < e and (best is None or e - s < best[0]):
                best = (e - s, text)
            i -= 1
        return best[1] if best else "-"

    groups = defaultdict(lambda: [0, 0.0, 0.0])
    unmatched = 0
    for start, end, gtid, corr, name in runtime:
        m = memcpy.get(corr)
        if m is None:
            unmatched += 1
            key = (
                thread_names.get(gtid, str(gtid & 0xFFFFFF)),
                name,
                "?",
                "?",
                "?",
                enclosing(gtid, start),
            )
            g = groups[key]
            g[0] += 1
            g[1] += (end - start) / 1e6
            continue
        ck, sk, dk, nbytes, mstart, mend = m
        key = (
            thread_names.get(gtid, str(gtid & 0xFFFFFF)),
            name,
            copy_kind.get(ck, str(ck)),
            f"{mem_kind.get(sk, sk)}->{mem_kind.get(dk, dk)}",
            byte_bucket(nbytes),
            enclosing(gtid, start),
        )
        g = groups[key]
        g[0] += 1
        g[1] += (end - start) / 1e6
        g[2] += (mend - mstart) / 1e6

    total_calls = sum(g[0] for g in groups.values())
    total_host = sum(g[1] for g in groups.values())
    print(
        f"{total_calls} memcpy runtime calls, {total_host:.1f} ms host, {unmatched} without a memcpy record"
    )
    print(
        f"{'thread':>10} {'call':<16} {'kind':<6} {'src->dst':<22} {'bytes':<8} {'calls':>6} {'host ms':>9} {'mean us':>8} {'dev ms':>8}  range"
    )
    for key, (n, host_ms, dev_ms) in sorted(groups.items(), key=lambda kv: -kv[1][1])[
        : args.top
    ]:
        thread, call, ck, sd, bb, rng = key
        print(
            f"{thread:>10} {call:<16} {ck:<6} {sd:<22} {bb:<8} {n:>6} {host_ms:>9.1f} {host_ms/n*1e3:>8.0f} {dev_ms:>8.1f}  {rng}"
        )


if __name__ == "__main__":
    main()
