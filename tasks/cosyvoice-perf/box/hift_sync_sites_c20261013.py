# Each blocking CUDA call of the vocoder thread inside hift.group ranges, by its order in the group and the runtime
# call just before it on the same thread (a copy with its direction, memory kinds and size), with count and time.
# usage: python3 hift_sync_sites_c20261013.py <trace.sqlite> [--skip-head-s S] [--skip-tail-s S]
import argparse
import bisect
import collections
import sqlite3

BLOCKING = ("cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize", "cudaMemcpy")
COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}
MEM_KIND = {0: "pageable", 1: "pinned", 2: "device"}

parser = argparse.ArgumentParser()
parser.add_argument("report")
parser.add_argument("--skip-head-s", type=float, default=1.0)
parser.add_argument("--skip-tail-s", type=float, default=1.0)
args = parser.parse_args()
db = sqlite3.connect(args.report)
strings = dict(db.execute("select id, value from StringIds"))
copies = {
    corr: (COPY_KIND.get(kind, kind), MEM_KIND.get(src, src), MEM_KIND.get(dst, dst), size)
    for corr, kind, src, dst, size in db.execute(
        "select correlationId, copyKind, srcKind, dstKind, bytes from CUPTI_ACTIVITY_KIND_MEMCPY"
    )
}
groups = collections.defaultdict(list)
for start, end, tid, text, text_id in db.execute(
    "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType in (59, 60) and end is not null"
):
    label = text if text is not None else strings.get(text_id, "")
    if label.startswith("hift.group"):
        groups[tid].append((start, end))
t0, t1 = db.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
t0 += int(args.skip_head_s * 1e9)
t1 -= int(args.skip_tail_s * 1e9)
for tid, ranges in groups.items():
    ranges.sort()
    calls = db.execute(
        "select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ? order by start",
        (tid,),
    ).fetchall()
    starts = [c[0] for c in calls]
    sites = collections.Counter()
    times = collections.Counter()
    group_count = 0
    for g_start, g_end in ranges:
        if g_start < t0 or g_end > t1:
            continue
        group_count += 1
        order = 0
        lo = bisect.bisect_left(starts, g_start)
        hi = bisect.bisect_left(starts, g_end)
        for k in range(lo, hi):
            start, end, corr, name_id = calls[k]
            name = strings.get(name_id, "").split("_v")[0]
            copy = copies.get(corr)
            if name in BLOCKING or (copy and copy[0] == "DtoH" and copy[2] == "pageable"):
                order += 1
                before = "none"
                if k > lo:
                    p_corr, p_name = calls[k - 1][2], strings.get(calls[k - 1][3], "").split("_v")[0]
                    p_copy = copies.get(p_corr)
                    before = f"{p_name} {' '.join(map(str, p_copy))}" if p_copy else p_name
                this = f"{name} {' '.join(map(str, copy))}" if copy else name
                sites[(order, this, before)] += 1
                times[(order, this, before)] += end - start
    print(f"thread {tid & 0xFFFFFF}: {group_count} hift.group ranges in the window")
    for key, count in sorted(sites.items()):
        order, this, before = key
        print(
            f"  #{order} {this} after {before}: {count} ({count / group_count:.2f} per group),"
            f" {times[key] / 1e6:.1f} ms total, {times[key] / count / 1e3:.0f} us mean"
        )
