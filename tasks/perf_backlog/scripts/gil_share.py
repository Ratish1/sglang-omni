"""Interpreter lock ownership per thread from a py-spy raw (collapsed stack) record.

Usage: python gil_share.py <gil_raw.txt>

Feed it the output of
    py-spy record --pid <stage pid> --gil --threads --nonblocking --rate 250 \
        --duration 60 --format raw -o gil_raw.txt
With --gil, py-spy keeps only the trace of the thread holding the interpreter lock, so
the trace count per thread is that thread's share of lock ownership. Each line of the
raw file is a stack, frames separated by semicolons, and a trailing count; with
--threads the first frame names the thread.

The record does not give the fraction of time the lock was held: py-spy's "Samples: N"
on stderr counts traces written, not sampling intervals, and intervals with no holder
or with a torn read (Errors) leave nothing in the file. Read the fraction from a
second record without --gil and with --idle, where every interval writes one trace
per thread.
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
