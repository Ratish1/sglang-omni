"""Per thread sample shares from a py-spy raw (collapsed stack) profile.

Usage: python pyspy_thread_top.py <profile.txt> [--top 25]
"""

import argparse
from collections import Counter, defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()

    totals: Counter[str] = Counter()
    inclusive: dict[str, Counter[str]] = defaultdict(Counter)
    leaf: dict[str, Counter[str]] = defaultdict(Counter)
    with open(args.profile) as handle:
        for line in handle:
            stack, _, count = line.rstrip("\n").rpartition(" ")
            if not stack:
                continue
            samples = int(count)
            frames = stack.split(";")
            labels = [f for f in frames if f.startswith(("process", "thread"))]
            body = [f for f in frames if not f.startswith(("process", "thread"))]
            thread = " ".join(labels) or "thread ?"
            totals[thread] += samples
            for frame in set(body):
                inclusive[thread][frame] += samples
            if body:
                leaf[thread][body[-1]] += samples

    grand = sum(totals.values())
    for thread, samples in totals.most_common():
        print(
            f"\n== {thread}: {samples} samples, {100.0 * samples / grand:.1f}% of all"
        )
        print("  inclusive")
        for frame, count in inclusive[thread].most_common(args.top):
            print(f"    {100.0 * count / samples:5.1f}%  {frame}")
        print("  self")
        for frame, count in leaf[thread].most_common(args.top):
            print(f"    {100.0 * count / samples:5.1f}%  {frame}")


if __name__ == "__main__":
    main()
