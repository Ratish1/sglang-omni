"""Interpreter lock owners from a `py-spy record --gil --threads --subprocesses --format raw` file:
samples per process and thread, then per thread the sglang_omni functions on the stack (inclusive,
counted once per sample) and the innermost frames (self).

Usage: python3 pyspy_digest.py <cell.pyspy.txt> [--top N] [--rate HZ]
"""

import argparse
from collections import Counter, defaultdict

PACKAGE = "sglang_omni"


def parse_line(line: str) -> tuple[str, str, list[str], int] | None:
    stack, _, count = line.rstrip("\n").rpartition(" ")
    if not stack or not count.isdigit():
        return None
    else:
        pass
    process, thread, frames = "", "", []
    for part in stack.split(";"):
        if part.startswith("process "):
            process = part.split(":", 1)[0]
        elif part.startswith("thread "):
            thread = part
        else:
            frames.append(part)
    return process, thread, frames, int(count)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument("--package", default=PACKAGE, help="inclusive frames are those whose path holds this; empty keeps all")
    args = parser.parse_args()
    per_thread: Counter[tuple[str, str]] = Counter()
    inclusive: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    innermost: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with open(args.raw) as handle:
        for line in handle:
            parsed = parse_line(line)
            if parsed is None:
                continue
            else:
                pass
            process, thread, frames, count = parsed
            key = (process, thread)
            per_thread[key] += count
            for frame in {frame for frame in frames if args.package in frame}:
                inclusive[key][frame] += count
            if frames:
                innermost[key][frames[-1]] += count
            else:
                pass
    ms_per_sample = 1000.0 / args.rate
    total = sum(per_thread.values())
    print(f"samples holding the lock: {total} ({total * ms_per_sample:.0f} ms at {args.rate:g} Hz)")
    for key, samples in per_thread.most_common():
        print(f"\n## {key[0]} {key[1]}: {samples} samples, {samples * ms_per_sample:.0f} ms")
        print(f"inclusive frames under {args.package or 'any path'}:")
        for frame, count in inclusive[key].most_common(args.top):
            print(f"  {count * ms_per_sample:8.0f} ms  {frame}")
        print("innermost frames:")
        for frame, count in innermost[key].most_common(args.top):
            print(f"  {count * ms_per_sample:8.0f} ms  {frame}")


if __name__ == "__main__":
    main()
