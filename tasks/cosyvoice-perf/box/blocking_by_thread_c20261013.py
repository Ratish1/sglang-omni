# Blocking CUDA calls per thread over the window (synchronizes, synchronous copies, pageable device to host copies,
# pageable host to device copies), with the thread's name, count, total and mean time, and the top calls by time.
# usage: python3 blocking_by_thread_c20261013.py <trace.sqlite> [--skip-head-s S] [--skip-tail-s S]
import argparse
import collections
import sqlite3

BLOCKING = ("cudaStreamSynchronize", "cudaEventSynchronize", "cudaDeviceSynchronize", "cudaMemcpy")
COPY_KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}

parser = argparse.ArgumentParser()
parser.add_argument("report")
parser.add_argument("--skip-head-s", type=float, default=1.0)
parser.add_argument("--skip-tail-s", type=float, default=1.0)
args = parser.parse_args()
db = sqlite3.connect(args.report)
strings = dict(db.execute("select id, value from StringIds"))
names = {tid: strings.get(name_id, "?") for name_id, tid in db.execute("select nameId, globalTid from ThreadNames")}
pageable = {
    corr: f"{COPY_KIND.get(kind, kind)} pageable"
    for corr, kind in db.execute(
        "select correlationId, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY"
        " where (copyKind = 2 and dstKind = 0) or (copyKind = 1 and srcKind = 0)"
    )
}
t0, t1 = db.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
t0 += int(args.skip_head_s * 1e9)
t1 -= int(args.skip_tail_s * 1e9)
count = collections.Counter()
time = collections.Counter()
calls = collections.defaultdict(collections.Counter)
for start, end, tid, corr, name_id in db.execute(
    "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and end <= ?",
    (t0, t1),
):
    name = strings.get(name_id, "").split("_v")[0]
    if corr in pageable:
        name = f"{name} {pageable[corr]}"
    elif name not in BLOCKING:
        continue
    else:
        pass
    count[tid] += 1
    time[tid] += end - start
    calls[tid][name] += end - start
print(f"window {(t1 - t0) / 1e9:.2f} s")
for tid, total in time.most_common():
    top = ", ".join(f"{name} {t / 1e6:.1f} ms" for name, t in calls[tid].most_common(3))
    print(
        f"  {names.get(tid, '?')[:28]:28s} tid {tid & 0xFFFFFF:6d}: {count[tid]:7d} calls, {total / 1e6:9.1f} ms,"
        f" mean {total / count[tid] / 1e3:7.0f} us; {top}"
    )
