"""Where a talker step's wall time goes along its kernel chain, from check 2's node-traced windows (11-01).

Under PDL a kernel starts before its predecessor ends and waits inside itself, so per-kernel durations overlap and
their sum overstates the step. This reader walks each replay's kernels in start order and charges each kernel only
its critical-path increment: its end minus the latest end of every kernel before it (zero when it ends earlier). The
increments of a replay sum to its span. Replays are grouped by kernel count (the predictor-only graph, about 425
kernels; the talker decode graph, about 940), and per group the increments are summed by kernel name per replay;
for the decode graph the kernels are also split into the talker body (until the first predictor kernel) and the rest.

usage: python3 predictor_chain.py REPORT.sqlite [--device 1]
"""

from __future__ import annotations

import argparse
import sqlite3
import statistics
from collections import defaultdict

PREDICTOR_KERNELS = ("norm_qkv_rope_store", "gemv_add", "norm_gate_up_silu", "decode_attention", "codebook_step")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--device", type=int, default=1)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    names = dict(db.execute("select id, value from StringIds"))
    replays: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for start, end, correlation, short in db.execute(
        "select start, end, correlationId, shortName from CUPTI_ACTIVITY_KIND_KERNEL "
        "where deviceId = ? and graphNodeId is not null",
        (args.device,),
    ):
        replays[correlation].append((start, end, names.get(short, "?")))
    groups = {"predictor graph (400 to 450 kernels)": (400, 450), "talker decode graph (900 to 1000 kernels)": (900, 1000)}
    for label, (low, high) in groups.items():
        chosen = [k for k in replays.values() if low <= len(k) <= high]
        if not chosen:
            print(f"{label}: none")
            continue
        else:
            pass
        per_name: dict[str, list[float]] = defaultdict(list)
        spans, body, rest = [], [], []
        for kernels in chosen:
            kernels.sort()
            latest = kernels[0][0]
            increments: dict[str, float] = defaultdict(float)
            in_body = True
            body_us = rest_us = 0.0
            for start, end, name in kernels:
                increment = max(end - latest, 0) / 1e3
                latest = max(latest, end)
                increments[name] += increment
                if in_body and any(marker in name for marker in PREDICTOR_KERNELS):
                    in_body = False
                else:
                    pass
                if in_body:
                    body_us += increment
                else:
                    rest_us += increment
            spans.append((latest - kernels[0][0]) / 1e3)
            body.append(body_us)
            rest.append(rest_us)
            for name, value in increments.items():
                per_name[name].append(value)
        count = len(chosen)
        print(
            f"\n{label}: {count} replays, span p50 {statistics.median(spans):.0f} us; "
            f"before the first predictor kernel p50 {statistics.median(body):.0f} us, from it on p50 {statistics.median(rest):.0f} us"
        )
        rows = sorted(((sum(v) / count, name) for name, v in per_name.items()), key=lambda row: -row[0])
        total = sum(row[0] for row in rows)
        print(f"{'us/replay':>10} {'share':>6}  kernel (critical-path increment, mean a replay)")
        for us, name in rows[:22]:
            print(f"{us:10.1f} {us / total:6.1%}  {name[:100]}")


if __name__ == "__main__":
    main()
