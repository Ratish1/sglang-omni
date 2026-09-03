"""Account for the host wall of each capture region from the CUDA runtime
and driver calls of the scheduler thread, which is all the trace holds
for that thread.

Usage: python trace_runtime.py windows.jsonl [detail_ordinal]
"""

import json
import sys
from collections import Counter

SCHED_TID = 92268224
# (label, begin capture ts, end capture ts, warmup1 ms, warmup2 ms) from the
# capture lines of the same run
CAPTURES = [
    ("bucket 1", 5454210113534.857, 5454210158686.305, 639.1, 43.7),
    ("bucket 16", 5454211358334.262, 5454211408003.265, 428.9, 46.2),
    ("bucket 12", 5454212192978.878, 5454212228903.358, 459.9, 45.4),
    ("bucket 8", 5454212784832.459, 5454212822191.473, 453.9, 45.7),
    ("bucket 4", 5454213339565.310, 5454213375128.737, 435.0, 45.1),
    ("bucket 2", 5454215239731.939, 5454215276011.169, 476.5, 44.9),
]


def load(path):
    events = []
    with open(path) as handle:
        for line in handle:
            ev = json.loads(line)
            if ev.get("ph") != "X":
                continue
            if ev.get("tid") != SCHED_TID and ev.get("cat") not in (
                "kernel",
                "gpu_memcpy",
                "gpu_memset",
            ):
                continue
            events.append(ev)
    events.sort(key=lambda e: e["ts"])
    return events


def region(events, a, b):
    return [e for e in events if a <= e["ts"] < b and e["tid"] == SCHED_TID]


def summarize(label, evs, a, b, detail):
    wall = (b - a) / 1e3
    by_cat = Counter()
    by_name = Counter()
    n_name = Counter()
    for e in evs:
        d = e.get("dur", 0) / 1e3
        by_cat[e.get("cat")] += d
        by_name[(e.get("cat"), e["name"])] += d
        n_name[(e.get("cat"), e["name"])] += 1
    api_total = sum(by_cat.values())
    print(
        f"  {label}: wall {wall:.1f} ms, api calls {len(evs)} summing {api_total:.1f} ms "
        f"(runtime {by_cat.get('cuda_runtime', 0):.1f}, driver {by_cat.get('cuda_driver', 0):.1f}), "
        f"host outside api calls {wall - api_total:.1f} ms"
    )
    if detail:
        print(
            "    by call:",
            [(c[1], n_name[c], round(v, 1)) for c, v in by_name.most_common(10)],
        )
        longest = sorted(evs, key=lambda e: -e.get("dur", 0))[:6]
        print(
            "    longest:",
            [
                (
                    e["name"],
                    round(e.get("dur", 0) / 1e3, 2),
                    round((e["ts"] - a) / 1e3, 1),
                )
                for e in longest
            ],
        )
        gaps = []
        for x, y in zip(evs, evs[1:]):
            gap = y["ts"] - (x["ts"] + x.get("dur", 0))
            gaps.append(
                (gap / 1e3, x["name"], y["name"], round((x["ts"] - a) / 1e3, 1))
            )
        gaps.sort(reverse=True)
        print(
            "    largest gaps (ms, after, before, at ms):",
            [(round(g, 2), p, q, t) for g, p, q, t in gaps[:6]],
        )
        # distribution of launch call durations
        launches = [e.get("dur", 0) for e in evs if e["name"] == "cudaLaunchKernel"]
        if launches:
            launches.sort()
            n = len(launches)
            print(
                f"    cudaLaunchKernel n={n} p50={launches[n // 2]:.0f}us p90={launches[int(n * 0.9)]:.0f}us max={launches[-1]:.0f}us"
            )
        # gaps between consecutive launches, i.e. host time per launch
        lts = [e for e in evs if e["name"] == "cudaLaunchKernel"]
        inter = [(y["ts"] - x["ts"]) for x, y in zip(lts, lts[1:])]
        if inter:
            inter.sort()
            n = len(inter)
            print(
                f"    launch to launch interval p50={inter[n // 2]:.0f}us p90={inter[int(n * 0.9)]:.0f}us max={inter[-1]:.0f}us"
            )


def main():
    events = load(sys.argv[1])
    detail_ordinal = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    for ordinal, (label, begin, end, w1_ms, w2_ms) in enumerate(CAPTURES, 1):
        print(f"== ordinal {ordinal} {label}")
        pre = [
            e
            for e in events
            if e["tid"] == SCHED_TID and begin - 1.5e6 <= e["ts"] < begin
        ]
        records = [e for e in pre if e["name"].startswith("cudaEventRecord")]
        syncs = [e for e in pre if e["name"] == "cudaDeviceSynchronize"][-2:]
        if len(records) < 2 or not syncs:
            print("  anchors missing", len(records), len(syncs))
            continue
        # warmup_done is the last event record before the first synchronize
        first_sync = syncs[0]
        done = [r for r in records if r["ts"] < first_sync["ts"]][-1]
        print(
            f"  anchors: warmup_done record at {(done['ts']-begin)/1e3:.1f} ms, device syncs at {[round((x['ts']-begin)/1e3,1) for x in syncs]} ms before capture begin"
        )
        w1_start = done["ts"] - (w1_ms + w2_ms) * 1e3
        w2_start = done["ts"] - w2_ms * 1e3
        detail = ordinal == detail_ordinal
        for rlabel, a, b in (
            ("warmup 1", w1_start, w2_start),
            ("warmup 2", w2_start, done["ts"]),
            ("drain flush enter", done["ts"], begin),
            ("capture pass", begin, end),
        ):
            summarize(
                rlabel, region(events, a, b), a, b, detail or rlabel == "warmup 1"
            )
        # driver call names in warmup 1 against warmup 2
        for rlabel, a, b in (
            ("warmup 1", w1_start, w2_start),
            ("warmup 2", w2_start, done["ts"]),
        ):
            drv = Counter()
            for e in region(events, a, b):
                if e.get("cat") == "cuda_driver":
                    drv[e["name"]] += 1
            print(f"  driver calls in {rlabel}: {drv.most_common(8)}")


if __name__ == "__main__":
    main()
