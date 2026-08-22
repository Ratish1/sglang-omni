"""Per-thread GIL share and top frames from a py-spy raw recording.

    py-spy record --pid <pid> --gil --threads --format raw --rate 200 \
        --duration <s> -o capture.gil.raw
    python profile/gil/aggregate_pyspy_raw.py capture.gil.raw --rate 200 --duration <s>

With --gil, py-spy records a sample only when some thread holds the GIL,
so samples / (rate x duration) is the fraction of wall time the GIL was
held by anyone, and a thread's samples / all samples is its share of
that. The raw format is one collapsed stack per line, frames separated
by ';', a space and the sample count at the end; with --threads the first
frame names the thread.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

THREAD_RE = re.compile(
    r'thread[^"\']*["\']([^"\']+)["\']|thread \(([^)]+)\)|^(thread.*?)$'
)


def thread_key(frame: str) -> str:
    m = THREAD_RE.search(frame)
    if m:
        return next(g for g in m.groups() if g)
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument(
        "--rate",
        type=float,
        required=True,
        help="samples per second used for the recording",
    )
    ap.add_argument(
        "--duration", type=float, required=True, help="seconds the recording ran"
    )
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    per_thread = Counter()
    frames_by_thread: dict[str, Counter] = defaultdict(Counter)
    total = 0
    with open(args.raw) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            stack, _, count = line.rpartition(" ")
            try:
                n = int(count)
            except ValueError:
                continue
            frames = stack.split(";")
            thread = (
                thread_key(frames[0]) if frames[0].startswith("thread") else "unnamed"
            )
            per_thread[thread] += n
            total += n
            # leaf-most Python frame that is not the thread pseudo-frame
            leaf = frames[-1] if len(frames) > 1 else frames[0]
            frames_by_thread[thread][leaf] += n

    expected = args.rate * args.duration
    print(
        f"samples {total}; expected ticks {expected:.0f}; GIL held by some thread {100 * total / expected:.1f}% of wall\n"
    )
    print(
        "| thread | samples | share of GIL-held time | share of wall |\n|---|---:|---:|---:|"
    )
    for thread, n in per_thread.most_common():
        print(
            f"| {thread} | {n} | {100 * n / total:.1f}% | {100 * n / expected:.1f}% |"
        )
    print()
    for thread, n in per_thread.most_common():
        print(f"## {thread}: top leaf frames\n")
        for frame, c in frames_by_thread[thread].most_common(args.top):
            print(f"- {100 * c / n:.1f}%  {frame}")
        print()


if __name__ == "__main__":
    main()
