import gzip
import json
import statistics
import sys

path = sys.argv[1]
mode = sys.argv[2]
with gzip.open(path) as trace_file:
    ev = json.load(trace_file)["traceEvents"]
kern = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
launch_cats = ("cuda_runtime", "cuda_driver")
cpu = [e for e in ev if e.get("cat") in launch_cats + ("cpu_op", "python_function")]
ops = sorted([e for e in cpu if e.get("cat") == "cpu_op"], key=lambda e: e["ts"])


def enclosing_op(ts):
    best = None
    for o in ops:
        if o["ts"] <= ts <= o["ts"] + o.get("dur", 0) and (
            best is None or o["dur"] < best["dur"]
        ):
            best = o
    return best


if mode == "graphstep":
    ks = sorted(kern, key=lambda k: k["ts"])
    # split into steps at gaps > 300 us
    steps, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if b["ts"] - (a["ts"] + a["dur"]) > 300:
            steps.append(cur)
            cur = []
        cur.append(b)
    steps.append(cur)
    big = [s for s in steps if len(s) > 500]
    print("steps with >500 kernels:", len(big), "of", len(steps))
    s = big[len(big) // 2]
    span = s[-1]["ts"] + s[-1]["dur"] - s[0]["ts"]
    tot = sum(k["dur"] for k in s)
    gaps = [b["ts"] - (a["ts"] + a["dur"]) for a, b in zip(s, s[1:])]
    print(
        f"one step: kernels {len(s)} span {span:.0f}us sum {tot:.0f}us gaps sum {sum(gaps):.0f}us gap p50 {statistics.median(gaps):.2f} p90 {sorted(gaps)[int(0.9*len(gaps))]:.2f} min {min(gaps):.2f}"
    )
    from collections import Counter, defaultdict

    cnt, dur = Counter(), defaultdict(float)
    for k in s:
        n = k["name"][:60]
        cnt[n] += 1
        dur[n] += k["dur"]
    for n, c in cnt.most_common(22):
        print(f"{c:5d} {dur[n]:8.1f}us {dur[n]/c:6.2f}us/each {n}")
    sys.exit()
pf = sorted(
    [e for e in cpu if e.get("cat") == "python_function" and e["name"].endswith(mode)],
    key=lambda e: e["ts"],
)
span = pf[len(pf) // 2]
t0, t1 = span["ts"], span["ts"] + span["dur"]
launches = [
    e
    for e in cpu
    if e.get("cat") in launch_cats
    and t0 <= e["ts"] <= t1
    and "correlation" in e.get("args", {})
]
corrs = {e["args"]["correlation"]: e for e in launches}
ks = sorted(
    [k for k in kern if k.get("args", {}).get("correlation") in corrs],
    key=lambda k: corrs[k["args"]["correlation"]]["ts"],
)
print(
    "kernels in span:",
    len(ks),
    "host us",
    span["dur"],
    "launch cats",
    sorted({corrs[k["args"]["correlation"]]["cat"] for k in ks}),
)
for i, k in enumerate(ks):
    l = corrs[k["args"]["correlation"]]
    o = enclosing_op(l["ts"])
    dims = str(o.get("args", {}).get("Input Dims", ""))[:60] if o else ""
    print(
        f"{i:3d} {k['dur']:6.1f}us {k['name'][:58]:58s} {(o['name'][:26] if o else ''):26s} {dims}"
    )
