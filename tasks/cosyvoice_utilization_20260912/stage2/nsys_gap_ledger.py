"""Nsight Systems gap ledger for the CosyVoice3 c16 streaming capture.

Answers, over the window spanned by the first and last GPU kernel:
  Q1 GPU busy/idle union and the idle gap histogram plus GPU metric means.
  Q2 every kernel launching thread, and the runtime call split of the top 3.
  Q3 per thread GIL hold/wait from the NVTX "GIL Trace" domain.
  Q4 attribution of every idle gap over 1 ms to the state of the thread whose
     kernel ends the gap.
  Q5 device time by kernel class per role, and AR/vocoder kernel concurrency.

Usage: python nsys_gap_ledger.py <trace.sqlite> <out_dir>
"""

import json
import os
import sqlite3
import sys
from bisect import bisect_left

NS = 1e6  # ns -> ms

SYNC_CALLS = (
    "cudaStreamSynchronize",
    "cudaDeviceSynchronize",
    "cudaEventSynchronize",
    "cuStreamSynchronize",
    "cuCtxSynchronize",
)
LAUNCH_CALLS = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cudaLaunchCooperativeKernel",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
)
GRAPH_CALLS = ("cudaGraphLaunch",)
MEMCPY_CALLS = ("cudaMemcpyAsync", "cudaMemcpy", "cuMemcpyAsync", "cuMemcpyHtoDAsync")

# note(ratish): first match wins, so conv kernels implemented as cutlass gemms
# land in conv and not in matmul.
KERNEL_CLASSES = (
    ("flash_attn", ("flash", "fmha", "fa3")),
    ("conv_cudnn", ("cudnn", "conv", "nchw", "nhwc", "wgrad", "implicit", "fprop")),
    ("matmul", ("gemm", "cutlass", "cublas", "gemv", "sgemm", "xmma")),
    ("layer_norm", ("layer_norm", "layernorm", "rms_norm", "rmsnorm")),
    ("elementwise", ("elementwise", "vectorized", "unrolled")),
)


def classify(name):
    low = name.lower()
    for cls, pats in KERNEL_CLASSES:
        for p in pats:
            if p in low:
                return cls
    return "reduce_other"


def merge(rows):
    """Union of (start, end) pairs given in start order."""
    out = []
    cs = ce = None
    for s, e in rows:
        if e <= s:
            continue
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            if e > ce:
                ce = e
        else:
            out.append((cs, ce))
            cs, ce = s, e
    if cs is not None:
        out.append((cs, ce))
    return out


def overlap(a, b):
    """Total overlap between two sorted, disjoint interval lists."""
    i = j = 0
    tot = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def clip(rows, w0, w1):
    out = []
    for s, e in rows:
        s = max(s, w0)
        e = min(e, w1)
        if e > s:
            out.append((s, e))
    return out


def pct(vals, q):
    if not vals:
        return 0
    k = int(round(q * (len(vals) - 1)))
    return vals[k]


def main():
    db_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    con = sqlite3.connect("file:" + db_path + "?mode=ro", uri=True)
    con.execute("PRAGMA temp_store=FILE")
    con.execute("PRAGMA cache_size=-2000000")
    q = con.execute
    R = {}

    # ---- window -------------------------------------------------------
    w0, w1 = q("select min(start), max(end) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
    R["window"] = {"start_ns": w0, "end_ns": w1, "length_ms": (w1 - w0) / NS}

    # ---- kernel -> launching thread -----------------------------------
    q(
        """create temp table kt as
           select k.start s, k.end e, r.globalTid%16777216 tid,
                  k.demangledName dn, k.streamId st
           from CUPTI_ACTIVITY_KIND_KERNEL k
           join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = k.correlationId"""
    )
    q("create index kt_s on kt(s)")
    q("create index kt_tid on kt(tid)")
    n_kern = q("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    n_join = q("select count(*) from kt").fetchone()[0]
    R["kernel_join"] = {"kernels": n_kern, "joined_to_a_runtime_call": n_join}

    q(
        """create temp table mt as
           select m.start s, m.end e, r.globalTid%16777216 tid
           from CUPTI_ACTIVITY_KIND_MEMCPY m
           join CUPTI_ACTIVITY_KIND_RUNTIME r on r.correlationId = m.correlationId"""
    )
    q("create index mt_s on mt(s)")

    # ---- Q1 busy / idle ------------------------------------------------
    def busy_union(with_memset):
        sql = """select start, end from CUPTI_ACTIVITY_KIND_KERNEL
                 union all select start, end from CUPTI_ACTIVITY_KIND_MEMCPY"""
        if with_memset:
            sql += " union all select start, end from CUPTI_ACTIVITY_KIND_MEMSET"
        return merge(clip(q(sql + " order by 1").fetchall(), w0, w1))

    busy = busy_union(False)
    busy_ms = busy_union(True)
    wlen = w1 - w0

    def summarize(u):
        b = sum(e - s for s, e in u)
        return {
            "busy_ms": b / NS,
            "idle_ms": (wlen - b) / NS,
            "idle_share": (wlen - b) / wlen,
        }

    R["q1"] = {
        "window_ms": wlen / NS,
        "kernel_plus_memcpy": summarize(busy),
        "kernel_plus_memcpy_plus_memset": summarize(busy_ms),
    }

    gaps = [(busy[i][1], busy[i + 1][0]) for i in range(len(busy) - 1)]
    gaps = [g for g in gaps if g[1] > g[0]]
    edges = [(0, 0.1), (0.1, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, float("inf"))]
    hist = []
    for lo, hi in edges:
        sel = [g for g in gaps if lo <= (g[1] - g[0]) / NS < hi]
        hist.append(
            {
                "bucket_ms": f"{lo}-{hi}",
                "count": len(sel),
                "total_ms": sum(e - s for s, e in sel) / NS,
            }
        )
    R["q1"]["idle_gap_histogram"] = hist
    R["q1"]["idle_gap_count"] = len(gaps)

    mids = {10: "GR Active [%]", 14: "SMs Active [%]", 15: "SM Issue [%]"}
    met = {}
    for mid, nm in mids.items():
        row = q(
            "select avg(value), count(*), max(value) from GPU_METRICS "
            "where metricId=? and timestamp between ? and ?",
            (mid, w0, w1),
        ).fetchone()
        met[nm] = {"mean": row[0], "samples": row[1], "max": row[2]}
    R["q1"]["gpu_metrics_window_mean"] = met

    # ---- Q2 threads ----------------------------------------------------
    names = dict(
        q(
            "select t.globalTid%16777216, s.value from ThreadNames t "
            "join StringIds s on s.id=t.nameId"
        ).fetchall()
    )
    thr = []
    for tid, n, t in q(
        "select tid, count(*), sum(e-s) from kt group by tid order by 3 desc"
    ):
        thr.append(
            {"tid": tid, "name": names.get(tid, ""), "kernels": n, "kernel_ms": t / NS}
        )
    R["q2"] = {"threads": thr}

    voc = max(
        thr,
        key=lambda r: q(
            "select coalesce(sum(k.e-k.s),0) from kt k join StringIds d on d.id=k.dn "
            "where k.tid=? and (d.value like '%flash%' or d.value like '%cudnn%')",
            (r["tid"],),
        ).fetchone()[0],
    )["tid"]
    ar = q(
        """select r.globalTid%16777216, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME r
           join StringIds s on s.id=r.nameId where s.value like 'cudaGraphLaunch%'
           group by 1 order by 2 desc limit 1"""
    ).fetchone()[0]
    R["q2"]["roles"] = {"vocoder_tid": voc, "ar_tid": ar}

    def role(tid):
        return "vocoder" if tid == voc else ("AR" if tid == ar else "other")

    top3 = [r["tid"] for r in thr[:3]]
    rt = {}
    for tid in top3:
        rows = q(
            """select s.value, count(*), sum(r.end-r.start)
               from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId
               where r.globalTid%16777216=? and r.start<? and r.end>?
               group by 1 order by 3 desc""",
            (tid, w1, w0),
        ).fetchall()
        buckets = {
            k: {"calls": 0, "ms": 0.0}
            for k in ("launch", "graph_launch", "memcpy_async", "synchronize", "other")
        }
        for nm, n, t in rows:
            base = nm.split("_v")[0]
            if base in LAUNCH_CALLS:
                k = "launch"
            elif base in GRAPH_CALLS:
                k = "graph_launch"
            elif base in MEMCPY_CALLS:
                k = "memcpy_async"
            elif base in SYNC_CALLS:
                k = "synchronize"
            else:
                k = "other"
            buckets[k]["calls"] += n
            buckets[k]["ms"] += t / NS
        rt[str(tid)] = {
            "role": role(tid),
            "total_calls": sum(r[1] for r in rows),
            "total_ms": sum(r[2] for r in rows) / NS,
            "buckets": buckets,
            "top_names": [
                {"name": r[0], "calls": r[1], "ms": r[2] / NS} for r in rows[:12]
            ],
        }
    R["q2"]["runtime_calls_top3"] = rt

    # ---- Q3 GIL --------------------------------------------------------
    gil_dom = q(
        "select distinct domainId, text from NVTX_EVENTS where eventType=75"
    ).fetchall()
    sid = dict(
        q(
            "select value, id from StringIds where value in ('Holding GIL','Waiting for GIL')"
        )
    )
    hold_id, wait_id = sid["Holding GIL"], sid["Waiting for GIL"]
    R["q3"] = {
        "source_table": "NVTX_EVENTS",
        "nvtx_domains": [{"domainId": d, "name": t} for d, t in gil_dom],
        "range_texts": {"Holding GIL": hold_id, "Waiting for GIL": wait_id},
        "eventType": 59,
    }
    pid = q("select globalPid from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone()[0]

    def gil_rows(text_id, tid=None):
        sql = (
            "select start, end from NVTX_EVENTS where domainId=1 and eventType=59 "
            "and textId=? and (globalTid/16777216)*16777216=? and start<? and end>?"
        )
        args = [text_id, pid, w1, w0]
        if tid is not None:
            sql += " and globalTid%16777216=?"
            args.append(tid)
        return clip(sorted(q(sql + " order by 1", args).fetchall()), w0, w1)

    gil_tids = [
        r[0]
        for r in q(
            "select globalTid%16777216, count(*) from NVTX_EVENTS where domainId=1 "
            "and eventType=59 and (globalTid/16777216)*16777216=? and start<? and end>? "
            "group by 1 order by 2 desc",
            (pid, w1, w0),
        ).fetchall()
    ]
    per_thread = []
    for tid in gil_tids:
        h = gil_rows(hold_id, tid)
        w = gil_rows(wait_id, tid)
        wd = sorted(e - s for s, e in w)
        per_thread.append(
            {
                "tid": tid,
                "role": role(tid),
                "hold_ms": sum(e - s for s, e in h) / NS,
                "hold_count": len(h),
                "wait_ms": sum(e - s for s, e in w) / NS,
                "wait_count": len(w),
                "wait_p50_us": pct(wd, 0.5) / 1e3,
                "wait_p95_us": pct(wd, 0.95) / 1e3,
                "wait_max_ms": (wd[-1] / NS) if wd else 0.0,
            }
        )
    per_thread.sort(key=lambda r: -r["hold_ms"])
    R["q3"]["per_thread"] = per_thread[:25]
    R["q3"]["threads_with_gil_events"] = len(gil_tids)
    all_hold = merge(gil_rows(hold_id))
    R["q3"]["any_thread_holding_ms"] = sum(e - s for s, e in all_hold) / NS
    R["q3"]["gil_free_ms"] = (wlen - sum(e - s for s, e in all_hold)) / NS

    # ---- Q4 gap attribution -------------------------------------------
    big = [g for g in gaps if (g[1] - g[0]) >= 1e6]
    kstarts = [r[0] for r in q("select s from kt order by s").fetchall()]
    ktids = dict()
    for s, t in q("select s, tid from kt order by s, tid"):
        ktids.setdefault(s, t)
    mtids = dict()
    for s, t in q("select s, tid from mt order by s, tid"):
        mtids.setdefault(s, t)

    by_thread = {}
    unattributed = 0
    for gs, ge in big:
        t = ktids.get(ge, mtids.get(ge))
        if t is None:
            i = bisect_left(kstarts, ge)
            t = ktids.get(kstarts[i]) if i < len(kstarts) else None
        if t is None:
            unattributed += ge - gs
            continue
        by_thread.setdefault(t, []).append((gs, ge))

    states = (
        "a_wait_gil",
        "b_hold_gil",
        "c_cuda_sync",
        "d_osrt",
        "e_none",
        "x_any_cuda_runtime_call",
    )
    agg = {r: {s: 0.0 for s in states} for r in ("AR", "vocoder", "other")}
    agg_total = {r: 0.0 for r in ("AR", "vocoder", "other")}
    agg_count = {r: 0 for r in ("AR", "vocoder", "other")}
    osrt_by_role = {r: {} for r in ("AR", "vocoder", "other")}
    for tid, gl in by_thread.items():
        gl.sort()
        rl = role(tid)
        agg_total[rl] += sum(e - s for s, e in gl) / NS
        agg_count[rl] += len(gl)
        a = merge(gil_rows(wait_id, tid))
        b = merge(gil_rows(hold_id, tid))
        csync = merge(
            clip(
                sorted(
                    q(
                        "select r.start, r.end from CUPTI_ACTIVITY_KIND_RUNTIME r "
                        "join StringIds s on s.id=r.nameId where r.globalTid%16777216=? "
                        "and ("
                        + " or ".join("s.value like '%s%%'" % c for c in SYNC_CALLS)
                        + ")"
                        " and r.start<? and r.end>?",
                        (tid, w1, w0),
                    ).fetchall()
                ),
                w0,
                w1,
            )
        )
        osrt_rows = q(
            "select o.start, o.end, s.value from OSRT_API o join StringIds s on s.id=o.nameId "
            "where o.globalTid%16777216=? and o.start<? and o.end>? order by 1",
            (tid, w1, w0),
        ).fetchall()
        d = merge(clip([(r[0], r[1]) for r in osrt_rows], w0, w1))
        agg[rl]["a_wait_gil"] += overlap(gl, a) / NS
        agg[rl]["b_hold_gil"] += overlap(gl, b) / NS
        agg[rl]["c_cuda_sync"] += overlap(gl, csync) / NS
        agg[rl]["d_osrt"] += overlap(gl, d) / NS
        cov = merge(sorted(a + b + csync + d))
        agg[rl]["e_none"] += (sum(e - s for s, e in gl) - overlap(gl, cov)) / NS
        # note(ratish): cross check for e, the thread sitting in any traced CUDA
        # runtime call, launches included.
        anyrt = merge(
            clip(
                sorted(
                    q(
                        "select start, end from CUPTI_ACTIVITY_KIND_RUNTIME "
                        "where globalTid%16777216=? and start<? and end>?",
                        (tid, w1, w0),
                    ).fetchall()
                ),
                w0,
                w1,
            )
        )
        agg[rl]["x_any_cuda_runtime_call"] += overlap(gl, anyrt) / NS
        for nm in set(r[2] for r in osrt_rows):
            iv = merge(
                clip(sorted((r[0], r[1]) for r in osrt_rows if r[2] == nm), w0, w1)
            )
            ov = overlap(gl, iv) / NS
            if ov > 0:
                osrt_by_role[rl][nm] = osrt_by_role[rl].get(nm, 0.0) + ov

    R["q4"] = {
        "gap_threshold_ms": 1.0,
        "gaps_over_1ms": len(big),
        "gap_time_over_1ms_ms": sum(e - s for s, e in big) / NS,
        "unattributed_ms": unattributed / NS,
        "note": "a-d are independent overlaps and can double count; e is gap time "
        "covered by none of a-d.",
        "per_role": {
            r: {"gaps": agg_count[r], "gap_ms": agg_total[r], "states": agg[r]}
            for r in agg
        },
        "osrt_top_by_role": {
            r: sorted(v.items(), key=lambda x: -x[1])[:8]
            for r, v in osrt_by_role.items()
        },
    }

    # who held the GIL during those gaps, per ending role
    holders = {r: {} for r in ("AR", "vocoder", "other")}
    role_gaps = {r: [] for r in ("AR", "vocoder", "other")}
    for tid, gl in by_thread.items():
        role_gaps[role(tid)].extend(gl)
    for r in role_gaps:
        role_gaps[r] = merge(sorted(role_gaps[r]))
    cur = {}
    for s, e, ht in q(
        "select start, end, globalTid%16777216 from NVTX_EVENTS where domainId=1 "
        "and eventType=59 and textId=? and (globalTid/16777216)*16777216=? "
        "and start<? and end>? order by 1",
        (hold_id, pid, w1, w0),
    ):
        cur.setdefault(ht, []).append((max(s, w0), min(e, w1)))
    for ht, iv in cur.items():
        iv = merge(sorted(iv))
        for r in holders:
            ov = overlap(role_gaps[r], iv) / NS
            if ov > 0.0:
                holders[r][ht] = ov
    R["q4"]["gil_holder_during_gaps"] = {
        r: [
            {"holder_tid": t, "holder_role": role(t), "ms": v}
            for t, v in sorted(v0.items(), key=lambda x: -x[1])[:10]
        ]
        for r, v0 in holders.items()
    }

    # ---- Q5 device time split -----------------------------------------
    q5 = {}
    for tid, label in ((voc, "vocoder"), (ar, "AR")):
        cls = {}
        for dn, n, t in q(
            "select d.value, count(*), sum(k.e-k.s) from kt k "
            "join StringIds d on d.id=k.dn where k.tid=? group by 1",
            (tid,),
        ):
            c = classify(dn)
            e = cls.setdefault(c, {"count": 0, "ms": 0.0, "top": []})
            e["count"] += n
            e["ms"] += t / NS
            e["top"].append((dn[:90], t / NS))
        for c in cls:
            cls[c]["top"] = sorted(cls[c]["top"], key=lambda x: -x[1])[:4]
        q5[label] = {
            "tid": tid,
            "total_ms": sum(v["ms"] for v in cls.values()),
            "classes": dict(sorted(cls.items(), key=lambda x: -x[1]["ms"])),
        }

    av = merge(
        clip(sorted(q("select s, e from kt where tid=?", (ar,)).fetchall()), w0, w1)
    )
    vv = merge(
        clip(sorted(q("select s, e from kt where tid=?", (voc,)).fetchall()), w0, w1)
    )
    both = overlap(av, vv)
    a_tot = sum(e - s for s, e in av)
    v_tot = sum(e - s for s, e in vv)
    q5["concurrency"] = {
        "ar_kernel_ms": a_tot / NS,
        "vocoder_kernel_ms": v_tot / NS,
        "both_ms": both / NS,
        "ar_only_ms": (a_tot - both) / NS,
        "vocoder_only_ms": (v_tot - both) / NS,
        "neither_ms": (wlen - (a_tot + v_tot - both)) / NS,
        "note": "neither counts time with no AR and no vocoder kernel; other "
        "threads' kernels, memcpy and memset may run then.",
    }
    R["q5"] = q5

    with open(os.path.join(out_dir, "gap_ledger.json"), "w") as f:
        json.dump(R, f, indent=1, default=str)
    write_md(R, os.path.join(out_dir, "gap_ledger.md"))
    print("wrote", out_dir)


def write_md(R, path):
    L = []
    w = L.append
    w("# nsys gap ledger, CosyVoice3 c16 streaming\n")
    q1 = R["q1"]
    w("## Q1 GPU busy and idle\n")
    w("| union | window ms | busy ms | idle ms | idle share |")
    w("|---|---|---|---|---|")
    for k in ("kernel_plus_memcpy", "kernel_plus_memcpy_plus_memset"):
        v = q1[k]
        w(
            "| %s | %.1f | %.1f | %.1f | %.3f |"
            % (k, q1["window_ms"], v["busy_ms"], v["idle_ms"], v["idle_share"])
        )
    w("\n| gap bucket ms | count | total ms |")
    w("|---|---|---|")
    for h in q1["idle_gap_histogram"]:
        w("| %s | %d | %.1f |" % (h["bucket_ms"], h["count"], h["total_ms"]))
    w("\n| gpu metric | window mean | samples |")
    w("|---|---|---|")
    for k, v in q1["gpu_metrics_window_mean"].items():
        w("| %s | %.2f | %d |" % (k, v["mean"], v["samples"]))

    w("\n## Q2 kernel launching threads\n")
    w("| tid | name | role | kernels | kernel ms |")
    w("|---|---|---|---|---|")
    roles = R["q2"]["roles"]
    for t in R["q2"]["threads"][:25]:
        rl = (
            "vocoder"
            if t["tid"] == roles["vocoder_tid"]
            else ("AR" if t["tid"] == roles["ar_tid"] else "other")
        )
        w(
            "| %d | %s | %s | %d | %.1f |"
            % (t["tid"], t["name"], rl, t["kernels"], t["kernel_ms"])
        )
    w("\ntotal threads launching kernels: %d\n" % len(R["q2"]["threads"]))
    w("| tid | role | bucket | calls | ms |")
    w("|---|---|---|---|---|")
    for tid, v in R["q2"]["runtime_calls_top3"].items():
        for b, bv in v["buckets"].items():
            w(
                "| %s | %s | %s | %d | %.1f |"
                % (tid, v["role"], b, bv["calls"], bv["ms"])
            )

    w("\n## Q3 GIL (NVTX domain 'GIL Trace')\n")
    w(
        "| tid | role | hold ms | holds | wait ms | waits | wait p50 us | wait p95 us | wait max ms |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for t in R["q3"]["per_thread"]:
        w(
            "| %d | %s | %.1f | %d | %.1f | %d | %.1f | %.1f | %.1f |"
            % (
                t["tid"],
                t["role"],
                t["hold_ms"],
                t["hold_count"],
                t["wait_ms"],
                t["wait_count"],
                t["wait_p50_us"],
                t["wait_p95_us"],
                t["wait_max_ms"],
            )
        )
    w(
        "\nany thread holding: %.1f ms, no thread holding: %.1f ms\n"
        % (R["q3"]["any_thread_holding_ms"], R["q3"]["gil_free_ms"])
    )

    w("## Q4 idle gaps over 1 ms\n")
    w(
        "gaps: %d, gap time: %.1f ms, unattributed: %.1f ms\n"
        % (
            R["q4"]["gaps_over_1ms"],
            R["q4"]["gap_time_over_1ms_ms"],
            R["q4"]["unattributed_ms"],
        )
    )
    w(
        "| ending role | gaps | gap ms | a wait gil | b hold gil | c cuda sync | "
        "d osrt | e none | x any cuda runtime call |"
    )
    w("|---|---|---|---|---|---|---|---|---|")
    for r, v in R["q4"]["per_role"].items():
        s = v["states"]
        w(
            "| %s | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f |"
            % (
                r,
                v["gaps"],
                v["gap_ms"],
                s["a_wait_gil"],
                s["b_hold_gil"],
                s["c_cuda_sync"],
                s["d_osrt"],
                s["e_none"],
                s["x_any_cuda_runtime_call"],
            )
        )
    w("\n| ending role | osrt function | overlapped ms |")
    w("|---|---|---|")
    for r, lst in R["q4"]["osrt_top_by_role"].items():
        for nm, v in lst:
            w("| %s | %s | %.1f |" % (r, nm, v))
    w("\n| ending role | gil holder tid | holder role | ms |")
    w("|---|---|---|---|")
    for r, lst in R["q4"]["gil_holder_during_gaps"].items():
        for h in lst:
            w(
                "| %s | %d | %s | %.1f |"
                % (r, h["holder_tid"], h["holder_role"], h["ms"])
            )

    w("\n## Q5 device time split\n")
    for label in ("vocoder", "AR"):
        v = R["q5"][label]
        w("\n%s thread %d, total %.1f ms\n" % (label, v["tid"], v["total_ms"]))
        w("| class | count | ms |")
        w("|---|---|---|")
        for c, cv in v["classes"].items():
            w("| %s | %d | %.1f |" % (c, cv["count"], cv["ms"]))
    c = R["q5"]["concurrency"]
    w("\n| concurrency | ms |")
    w("|---|---|")
    for k in (
        "ar_kernel_ms",
        "vocoder_kernel_ms",
        "both_ms",
        "ar_only_ms",
        "vocoder_only_ms",
        "neither_ms",
    ):
        w("| %s | %.1f |" % (k, c[k]))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
