"""Interpreter lock ownership per thread from a py-spy raw (collapsed stack) record.

Usage: python gil_share.py <gil_raw.txt> [--rate 250] [--duration 60]

Feed it the output of
    py-spy record --pid <stage pid> --gil --threads --nonblocking --rate 250 \
        --duration 60 --format raw -o gil_raw.txt
With --gil, py-spy keeps only samples of the thread holding the interpreter lock, so
the sample count per thread is that thread's share of lock ownership. Each line of the
raw file is a stack, frames separated by semicolons, and a trailing sample count; with
--threads the first frame names the thread. With --rate and --duration the script also
prints the fraction of wall time the lock was held by anyone.
"""

import argparse
import re
from collections import Counter

THREAD = re.compile(r'thread \(?(\d+)\)?[^"]*"([^"]*)"')


def thread_key(first_frame: str) -> str:
    match = THREAD.search(first_frame)
    if match:
        return f"{match.group(2)} ({match.group(1)})"
    return first_frame.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--rate", type=float, default=None, help="samples per second")
    parser.add_argument("--duration", type=float, default=None, help="record seconds")
    args = parser.parse_args()

    per_thread: Counter[str] = Counter()
    with open(args.raw) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            stack, _, count = line.rpartition(" ")
            if not count.isdigit():
                continue
            frames = stack.split(";")
            per_thread[thread_key(frames[0])] += int(count)

    total = sum(per_thread.values())
    if total == 0:
        print("no samples")
        return 1
    print(f"{'thread':48s} {'samples':>9s} {'share %':>8s}")
    for name, count in per_thread.most_common():
        print(f"{name:48s} {count:9d} {100.0 * count / total:8.1f}")
    print(f"{'total':48s} {total:9d}")
    if args.rate and args.duration:
        held = total / (args.rate * args.duration)
        print(f"lock held by any thread: {100.0 * held:.1f} percent of the record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
