"""py-spy raw (collapsed) stacks of a server boot reduced per process: wall seconds sampled, then the functions with the
most inclusive seconds (each function counted once per stack, line numbers dropped) and the most leaf seconds.

usage: python3 pyspy_reduce.py RAW_FILE --rate 100 [--top 45] [--processes 4]
"""

import argparse
import re
from collections import Counter, defaultdict

FRAME = re.compile(r"^(.*?) \(([^()]*?)(?::\d+)?\)$")


def label(frame: str) -> str:
    match = FRAME.match(frame.strip())
    if match is None:
        return frame.strip()[:120]
    else:
        pass
    path = match.group(2)
    for marker in ("site-packages/", "/workspace/", "lib/python3.12/"):
        if marker in path:
            path = path.split(marker, 1)[1]
            break
        else:
            pass
    return f"{match.group(1)} ({path})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--top", type=int, default=45)
    parser.add_argument("--processes", type=int, default=4)
    args = parser.parse_args()
    totals: Counter[str] = Counter()
    inclusive: dict[str, Counter[str]] = defaultdict(Counter)
    leaf: dict[str, Counter[str]] = defaultdict(Counter)
    for line in open(args.raw, errors="replace"):
        stack, _, count_text = line.rstrip("\n").rpartition(" ")
        if not stack or not count_text.isdigit():
            continue
        else:
            pass
        count = int(count_text)
        frames = stack.split(";")
        process_indices = [index for index, frame in enumerate(frames) if frame.startswith("process ")]
        if process_indices:
            process = frames[process_indices[-1]][:110]
            body = frames[process_indices[-1] + 1 :]
        else:
            process = "process ?"
            body = frames
        body = [frame for frame in body if not frame.startswith("thread ")]
        totals[process] += count
        for name in {label(frame) for frame in body}:
            inclusive[process][name] += count
        if body:
            leaf[process][label(body[-1])] += count
        else:
            pass
    for process, samples in totals.most_common(args.processes):
        print(f"== {process}: {samples / args.rate:.1f} s sampled")
        print("  inclusive:")
        for name, count in inclusive[process].most_common(args.top):
            print(f"  {count / args.rate:8.1f} s  {name[:150]}")
        print("  leaf:")
        for name, count in leaf[process].most_common(15):
            print(f"  {count / args.rate:8.1f} s  {name[:150]}")


if __name__ == "__main__":
    main()
