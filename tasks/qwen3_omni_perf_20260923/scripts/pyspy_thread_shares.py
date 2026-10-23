"""Where one thread's wall time goes, from a py-spy raw record taken with --threads --idle.

Each raw line is one stack, "thread (tid): name;frame (file:line);... count". The thread is
picked by its name prefix; every function's inclusive share counts a stack once however many
times the function appears in it, and the self share is the innermost frame's line. Shares are
of the thread's samples, idle ones included, so they are shares of wall time.

usage: python3 pyspy_thread_shares.py pyspy_talker.txt --thread scheduler-talker_ar [--top 40]
"""

from __future__ import annotations

import argparse
import collections
import re

FRAME = re.compile(r"^(.*) \((.*?)(?::(\d+))?\)$")


def function_of(frame: str) -> str:
    match = FRAME.match(frame)
    if match is None:
        return frame
    else:
        return f"{match.group(1)} ({match.group(2)})"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record")
    parser.add_argument("--thread", required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    inclusive: collections.Counter[str] = collections.Counter()
    self_lines: collections.Counter[str] = collections.Counter()
    total = 0
    with open(args.record) as handle:
        for line in handle:
            stack_text, _, count_text = line.rstrip("\n").rpartition(" ")
            frames = stack_text.split(";")
            name = frames[0].split(": ", 1)[-1]
            if not name.startswith(args.thread):
                continue
            else:
                pass
            count = int(count_text)
            total += count
            for function in {function_of(frame) for frame in frames[1:]}:
                inclusive[function] += count
            self_lines[frames[-1]] += count
    if total == 0:
        print(f"no samples on a thread named {args.thread}*")
        return
    else:
        pass
    print(f"{total} samples on {args.thread}*")
    print("  inclusive share")
    for function, count in inclusive.most_common(args.top):
        print(f"  {100 * count / total:6.1f} %  {function}")
    print("  self share (innermost line)")
    for frame, count in self_lines.most_common(args.top):
        print(f"  {100 * count / total:6.1f} %  {frame}")


if __name__ == "__main__":
    main()
