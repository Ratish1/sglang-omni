# CUDA runtime and driver API calls per thread over the window: count, total and mean time by call, for the threads
# with the most API time, so a call that got slower (a launch waiting on the driver behind another thread) shows by
# its mean.
# usage: python3 api_time_by_thread_c20261013.py <trace.sqlite> [--skip-head-s S] [--skip-tail-s S] [--threads N]
import argparse
import collections
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("report")
parser.add_argument("--skip-head-s", type=float, default=1.0)
parser.add_argument("--skip-tail-s", type=float, default=1.0)
parser.add_argument("--threads", type=int, default=2)
parser.add_argument("--top", type=int, default=10)
args = parser.parse_args()
db = sqlite3.connect(args.report)
strings = dict(db.execute("select id, value from StringIds"))
t0, t1 = db.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
t0 += int(args.skip_head_s * 1e9)
t1 -= int(args.skip_tail_s * 1e9)
count = collections.defaultdict(collections.Counter)
time = collections.defaultdict(collections.Counter)
for start, end, tid, name_id in db.execute(
    "select start, end, globalTid, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start >= ? and end <= ?", (t0, t1)
):
    name = strings.get(name_id, "").split("_v")[0]
    count[tid][name] += 1
    time[tid][name] += end - start
totals = collections.Counter({tid: sum(calls.values()) for tid, calls in time.items()})
print(f"window {(t1 - t0) / 1e9:.2f} s")
for tid, total in totals.most_common(args.threads):
    print(f"  tid {tid & 0xFFFFFF}: {sum(count[tid].values())} calls, {total / 1e6:.1f} ms")
    for name, t in time[tid].most_common(args.top):
        print(f"    {name:32s} {count[tid][name]:8d}  {t / 1e6:9.1f} ms  mean {t / count[tid][name] / 1e3:8.2f} us")
