"""Host load each second while a pair serves: the busy share of each slot's pinned cores and of the other NUMA node's
cores (from /proc/stat, host wide in the container), the load average, and every card's utilization and memory, so a
cell that shared its cores or the host with another tenant's work is named from data. Runs until killed; with
--summary, reads the samples back per cell window of a pair directory instead.
usage: python3 host_load_c20261022.py <out jsonl>
       python3 host_load_c20261022.py --summary <out jsonl> <pair dir> <cells>"""

import json
import subprocess
import sys
import time
from pathlib import Path

SETS = {
    "slot0": [*range(0, 16), *range(64, 80)],
    "slot1": [*range(16, 32), *range(80, 96)],
    "node1": [*range(32, 64), *range(96, 128)],
}


def core_times():
    times = {}
    for line in open("/proc/stat"):
        name, *fields = line.split()
        if name.startswith("cpu") and name != "cpu":
            values = [int(field) for field in fields]
            times[int(name[3:])] = (values[3] + values[4], sum(values))
    return times


def cards():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return {row.split(",")[0].strip(): [int(value) for value in row.split(",")[1:]] for row in out.strip().splitlines()}


def sample(out):
    previous = core_times()
    with open(out, "a") as handle:
        while True:
            time.sleep(1)
            current = core_times()
            busy = {}
            for name, cores in SETS.items():
                idle = sum(current[core][0] - previous[core][0] for core in cores if core in current)
                total = sum(current[core][1] - previous[core][1] for core in cores if core in current)
                busy[name] = round(1 - idle / total, 3) if total else None
            previous = current
            handle.write(json.dumps({
                "t": time.time(), "busy": busy, "loadavg": open("/proc/loadavg").read().split()[0], "cards": cards(),
            }) + "\n")
            handle.flush()


def summary(out, pair_dir, cells):
    samples = [json.loads(line) for line in open(out)]
    print("| order | cell | samples | slot0 busy mean | slot1 busy mean | node1 busy mean | card 2 util max | card 3 util max |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for order in ("a", "b"):
        start = (Path(pair_dir) / order / "identity_gpu0.txt").stat().st_mtime
        for cell in cells.split(","):
            end = max(
                (Path(pair_dir) / order / f"gpu{gpu}" / cell / "speed_results.json").stat().st_mtime for gpu in (0, 1)
            )
            window = [entry for entry in samples if start <= entry["t"] <= end]
            start = end
            if not window:
                continue

            def mean(name):
                values = [entry["busy"][name] for entry in window if entry["busy"][name] is not None]
                return f"{sum(values) / len(values):.3f}" if values else ""

            def card_max(index):
                return max(entry["cards"].get(index, [0, 0])[0] for entry in window)

            print(
                f"| {order} | {cell} | {len(window)} | {mean('slot0')} | {mean('slot1')} | {mean('node1')} | "
                f"{card_max('2')} | {card_max('3')} |"
            )


if __name__ == "__main__":
    if sys.argv[1] == "--summary":
        summary(*sys.argv[2:5])
    else:
        sample(sys.argv[1])
