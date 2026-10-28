"""The thinker's eager steps (prefill and mixed chunks) in a card 0 nsys trace: span, own device time, other stages'
device time inside, kernel mix. usage: python3 - TRACE.sqlite SERVE.LOG < thinker_steps.py"""

import re
import sqlite3
import statistics
import sys
from collections import defaultdict

STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=\d+ .*pid=(\d+)")
GAP_NS = 3_000_000


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def main() -> None:
    trace, serve_log = sys.argv[1], sys.argv[2]
    stages = {}
    with open(serve_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            if match := STARTUP.search(line):
                stages[int(match[2])] = match[1]
    thinker = next(pid for pid, stage in stages.items() if stage == "thinker")
    db = sqlite3.connect(trace)
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    kernels = []
    others = []
    graphs = []
    for start, end, global_pid, name, graph in db.execute(
        "SELECT start, end, globalPid, shortName, graphId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId = 0"
    ):
        pid = pid_of(global_pid)
        if pid == thinker and not graph:
            kernels.append((start, end, strings.get(name, str(name))))
        elif pid != thinker:
            others.append((start, end, pid))
        else:
            pass
    for start, end, global_pid in db.execute(
        "SELECT start, end, globalPid FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE deviceId = 0"
    ):
        if pid_of(global_pid) == thinker:
            graphs.append((start, end))
        else:
            others.append((start, end, pid_of(global_pid)))
    kernels.sort()
    steps = []
    for kernel in kernels:
        if steps and kernel[0] - steps[-1][-1][1] < GAP_NS:
            steps[-1].append(kernel)
        else:
            steps.append([kernel])
    big = [step for step in steps if step[-1][1] - step[0][0] > 30_000_000]
    print(f"thinker pid {thinker}: {len(kernels)} eager kernels, {len(graphs)} graph replays, "
          f"{len(steps)} eager groups, {len(big)} longer than 30 ms")
    print("| start s | span ms | own busy ms | others inside ms (by stage) | kernels | heaviest |")
    print("|---|---|---|---|---|---|")
    origin = kernels[0][0] if kernels else 0
    spans = []
    for step in big:
        begin, finish = step[0][0], step[-1][1]
        own = sum(end - start for start, end, _ in step)
        inside = defaultdict(int)
        for start, end, pid in others:
            if end > begin and start < finish:
                inside[stages.get(pid, str(pid))] += min(end, finish) - max(start, begin)
            else:
                pass
        by_name = defaultdict(int)
        for start, end, name in step:
            by_name[name] += end - start
        heaviest = "; ".join(f"{name[:28]} {value / 1e6:.0f}" for name, value in
                             sorted(by_name.items(), key=lambda item: -item[1])[:4])
        inside_text = ", ".join(f"{name} {value / 1e6:.0f}" for name, value in sorted(inside.items()))
        spans.append((finish - begin) / 1e6)
        print(f"| {(begin - origin) / 1e9:.2f} | {(finish - begin) / 1e6:.0f} | {own / 1e6:.0f} | {inside_text} | "
              f"{len(step)} | {heaviest} |")
    for step in big:
        begin, finish = step[0][0], step[-1][1]
        own = sum(end - start for start, end, _ in step)
        if own > 0.5 * (finish - begin):
            continue
        else:
            pass
        replays_inside = sum(1 for start, end in graphs if begin < start < finish)
        gaps = sorted(((step[i + 1][0] - step[i][1], step[i][2], step[i + 1][2]) for i in range(len(step) - 1)),
                      reverse=True)
        print(f"\nhost-bound group at {(begin - origin) / 1e9:.2f} s: {replays_inside} decode replays inside, "
              f"gaps above 0.5 ms {sum(1 for gap in gaps if gap[0] > 500_000)}, their sum "
              f"{sum(gap[0] for gap in gaps if gap[0] > 500_000) / 1e6:.0f} ms")
        for gap, before, after in gaps[:6]:
            print(f"  {gap / 1e6:.2f} ms after {before[:40]} before {after[:40]}")
    if spans:
        print(f"spans: median {statistics.median(spans):.0f} ms, mean {statistics.mean(spans):.0f}")
    else:
        pass
    durations = sorted((end - start) / 1e3 for start, end in graphs)
    if durations:
        print(f"decode graph replays: median {statistics.median(durations):.0f} us, "
              f"p90 {durations[int(0.9 * len(durations))]:.0f} us")
    else:
        pass


if __name__ == "__main__":
    main()
