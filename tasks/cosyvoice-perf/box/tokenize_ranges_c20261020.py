"""The speech tokenizer's calls in a profile (10-20): every pre.tokenize range (one extract_speech_token call, the run
lock's wait included) with its wall time, the device time of the kernels its thread launched inside it, and how many
calls overlap another. Main's c16 profile measured 25.5 ms of wall for 4.85 ms of kernels; the same call takes 8.5 ms
on an idle card (tokenizer_modes_c20261020.py).
usage: python3 tokenize_ranges_c20261020.py <name>=<profile.sqlite> ..."""

import bisect
import collections
import sqlite3
import statistics
import sys


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def report(name: str, path: str) -> None:
    cursor = sqlite3.connect(path).cursor()
    ranges = cursor.execute(
        """select e.start, e.end, e.globalTid from NVTX_EVENTS e left join StringIds s on e.textId = s.id
        where COALESCE(e.text, s.value) = 'pre.tokenize' and e.end is not null order by e.start"""
    ).fetchall()
    by_thread = collections.defaultdict(list)
    for start, end, thread in ranges:
        by_thread[thread].append((start, end))
    kernel_ns = collections.Counter()
    for kernel_start, kernel_end, thread, launch in cursor.execute(
        """select k.start, k.end, r.globalTid, r.start from CUPTI_ACTIVITY_KIND_KERNEL k
        join CUPTI_ACTIVITY_KIND_RUNTIME r on k.correlationId = r.correlationId"""
    ):
        spans = by_thread.get(thread)
        if not spans:
            continue
        index = bisect.bisect_right([start for start, _ in spans], launch) - 1
        if index >= 0 and spans[index][0] <= launch <= spans[index][1]:
            kernel_ns[(thread, index)] += kernel_end - kernel_start
        else:
            pass
    wall = [(end - start) / 1e6 for start, end, _ in ranges]
    kernels = [nanoseconds / 1e6 for nanoseconds in kernel_ns.values()]
    overlapping = sum(
        any(other_start < end and other_end > start for j, (other_start, other_end, _) in enumerate(ranges) if j != i)
        for i, (start, end, _) in enumerate(ranges)
    )
    print(
        f"{name}: calls {len(ranges)}, threads {len(by_thread)}, wall ms mean {statistics.mean(wall):.2f} "
        f"p50 {quantile(wall, 0.5):.2f} p95 {quantile(wall, 0.95):.2f}, kernel ms per call mean "
        f"{statistics.mean(kernels):.2f} p95 {quantile(kernels, 0.95):.2f}, calls overlapping another {overlapping}"
    )


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        label, sqlite_path = argument.split("=", 1)
        report(label, sqlite_path)
