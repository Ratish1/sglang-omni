# Every module load (cuModuleLoadData, cuModuleLoadDataEx, cuLibraryLoadData) and kernel attribute set
# (cuKernelSetAttribute, cuFuncSetAttribute) in the window, per thread: the innermost probe range at the call, the
# call's time, and the first kernel launched on that thread after it (by correlation id), grouped by range and kernel.
# usage: python3 module_loads_c20261013.py <trace.sqlite> [--skip-head-s S] [--skip-tail-s S]
import argparse
import bisect
import collections
import sqlite3

LOADS = ("cuModuleLoadData", "cuModuleLoadDataEx", "cuLibraryLoadData", "cuKernelSetAttribute", "cuFuncSetAttribute")
LAUNCHES = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch")

parser = argparse.ArgumentParser()
parser.add_argument("report")
parser.add_argument("--skip-head-s", type=float, default=1.0)
parser.add_argument("--skip-tail-s", type=float, default=1.0)
args = parser.parse_args()
db = sqlite3.connect(args.report)
strings = dict(db.execute("select id, value from StringIds"))
kernel_name = {
    corr: strings.get(name_id, "?")
    for corr, name_id in db.execute("select correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL")
}
t0, t1 = db.execute("select min(start), max(end) from CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()
t0 += int(args.skip_head_s * 1e9)
t1 -= int(args.skip_tail_s * 1e9)
ranges = collections.defaultdict(list)
for start, end, tid, text, text_id in db.execute(
    "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType in (59, 60) and end is not null"
):
    label = text if text is not None else strings.get(text_id, "")
    ranges[tid].append((start, end, " ".join(w for w in label.split(" ") if "=" not in w)[:40]))


def innermost(tid, t):
    best = None
    for start, end, label in ranges[tid]:
        if start <= t <= end and (best is None or start >= best[0]):
            best = (start, label)
    return best[1] if best else "outside"


calls = collections.defaultdict(list)
for start, end, tid, corr, name_id in db.execute(
    "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start"
):
    calls[tid].append((start, end, corr, strings.get(name_id, "").split("_v")[0]))
groups = collections.Counter()
times = collections.Counter()
for tid, rows in calls.items():
    for index, (start, end, corr, name) in enumerate(rows):
        if name not in LOADS or start < t0 or end > t1:
            continue
        following = "none"
        for _, _, next_corr, next_name in rows[index + 1 : index + 40]:
            if next_name in LAUNCHES:
                following = kernel_name.get(next_corr, next_name)[:48]
                break
        key = (tid & 0xFFFFFF, name, innermost(tid, start), following)
        groups[key] += 1
        times[key] += end - start
print(f"window {(t1 - t0) / 1e9:.2f} s")
for key, t in times.most_common(30):
    tid, name, label, following = key
    print(f"  tid {tid} {name} in {label} before {following}: {groups[key]}, {t / 1e6:.1f} ms, mean {t / groups[key] / 1e3:.0f} us")
