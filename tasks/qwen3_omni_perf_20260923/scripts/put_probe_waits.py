"""Wall time against thread CPU time of every probed pack and SHM block creation, per process.

Reads the shm_put_size_probe records of each boot and groups them by process, function and
whether the packed tensors included accelerator tensors (a device to host copy inside the pack).
Per group it prints the count, wall p50, p90, p99 and max, thread CPU p50 and p90, and the p90 and
p99 of wall minus CPU, the time the call's thread spent blocked (a device synchronization or a
wait for the GIL), so a slow send names whether its time was work or waiting.

usage: python put_probe_waits.py RECORDS_DIR [RECORDS_DIR ...]
"""

from __future__ import annotations

import collections
import glob
import json
import sys


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    for records_dir in sys.argv[1:]:
        groups: dict[tuple[str, str, bool], list[tuple[float, float]]] = (
            collections.defaultdict(list)
        )
        for path in glob.glob(f"{records_dir}/put_*.jsonl"):
            with open(path) as records_file:
                for line in records_file:
                    record = json.loads(line)
                    key = (
                        record["process"],
                        record["function"],
                        record.get("cuda_inputs", 0) > 0,
                    )
                    groups[key].append((record["wall_us"], record["cpu_us"]))
        print(f"== {records_dir}")
        for (process, function, has_cuda), calls in sorted(groups.items()):
            walls = [wall for wall, _ in calls]
            cpus = [cpu for _, cpu in calls]
            blocked = [wall - cpu for wall, cpu in calls]
            print(
                f"  {process:28s} {function:24s} cuda {str(has_cuda):5s} n {len(calls):5d}"
                f" wall p50 {percentile(walls, 0.5):8.1f} p90 {percentile(walls, 0.9):8.1f}"
                f" p99 {percentile(walls, 0.99):8.1f} max {max(walls):9.1f} us"
                f" cpu p50 {percentile(cpus, 0.5):7.1f} p90 {percentile(cpus, 0.9):7.1f}"
                f" blocked p90 {percentile(blocked, 0.9):8.1f} p99 {percentile(blocked, 0.99):8.1f} us"
            )


if __name__ == "__main__":
    main()
