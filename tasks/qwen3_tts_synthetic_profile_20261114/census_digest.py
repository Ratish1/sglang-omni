"""Step and vocoder batch census of one served timing cell (nsys SQLite export, probe labels on):
talker steps by mode, vocoder initial and follow-up batches by stream count, and kernel time per
CUDA stream, all inside the window from the first to the last labeled talker step.

Usage: python3 census_digest.py <cell.sqlite>
"""

import argparse
import re
import sqlite3
from collections import defaultdict

STEP = re.compile(r"step\[(\w+) bs=(\d+)")
VOCODER = re.compile(r"vocoder\.(initial|followup)\[streams=(\d+)\]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    args = parser.parse_args()
    connection = sqlite3.connect(args.sqlite)
    ranges = connection.execute(
        """
        select e.start, e.end, coalesce(s.value, e.text, '')
        from NVTX_EVENTS e left join StringIds s on s.id = e.textId
        where e.end is not null and e.end > e.start
        """
    ).fetchall()
    steps = [(start, end, text) for start, end, text in ranges if STEP.search(text)]
    window_start = min(start for start, _, _ in steps)
    window_end = max(end for _, end, _ in steps)
    window_ms = (window_end - window_start) / 1e6
    print(f"window {window_ms:.1f} ms, talker steps {len(steps)}, {window_ms / len(steps):.2f} ms a step")
    by_mode: dict[str, list[float]] = defaultdict(list)
    for start, end, text in steps:
        mode, _ = STEP.search(text).groups()
        by_mode[mode].append((end - start) / 1e6)
    for mode, durations in sorted(by_mode.items()):
        print(f"  {mode}: {len(durations)} steps, mean {sum(durations) / len(durations):.2f} ms")
    batches: dict[tuple[str, int], list[float]] = defaultdict(list)
    for start, end, text in ranges:
        match = VOCODER.search(text)
        if match and window_start <= start <= window_end:
            batches[(match.group(1), int(match.group(2)))].append((end - start) / 1e6)
        else:
            pass
    for kind in ("initial", "followup"):
        rows = sorted((streams, durations) for (name, streams), durations in batches.items() if name == kind)
        count = sum(len(durations) for _, durations in rows)
        streams_total = sum(streams * len(durations) for streams, durations in rows)
        busy = sum(sum(durations) for _, durations in rows)
        print(f"vocoder {kind}: {count} batches, {streams_total} streams, {busy:.1f} ms in batches")
        for streams, durations in rows:
            print(f"  streams={streams}: {len(durations)} batches, mean {sum(durations) / len(durations):.3f} ms")
    kernels = connection.execute(
        "select streamId, start, end from CUPTI_ACTIVITY_KIND_KERNEL where end > ? and start < ?",
        (window_start, window_end),
    ).fetchall()
    per_stream: dict[int, float] = defaultdict(float)
    for stream, start, end in kernels:
        per_stream[stream] += (min(end, window_end) - max(start, window_start)) / 1e6
    print("kernel time per stream in the window:")
    for stream, busy in sorted(per_stream.items(), key=lambda item: -item[1]):
        print(f"  stream {stream}: {busy:.1f} ms ({busy / window_ms:.1%})")


if __name__ == "__main__":
    main()
