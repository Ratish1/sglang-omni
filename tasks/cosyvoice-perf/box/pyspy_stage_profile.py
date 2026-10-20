"""Where a stage thread spends its GIL time, from py-spy raw output recorded with --gil --threads
(one trace per interval, the lock holder). Trace counts are ownership shares, never a held
fraction of the recording.

Threads are picked by a function on their stacks (--thread-has, e.g. run_step for the
Fun-CosyVoice3 vocoder thread, event_loop_normal for the AR engine). Per picked thread: its
share of all holder traces, inclusive shares of the named phases, the top leaf frames
(file:line), and the top direct children of the expanded phases.

usage: python pyspy_stage_profile.py RAW.txt --thread-has FUNC [--phases a,b] [--expand a,b] [--top N]
"""

from __future__ import annotations

import argparse
import collections
import re

LINE = re.compile(r"^thread \((\d+)\): ([^;]*);?(.*) (\d+)$")


def function_of(frame: str) -> str:
    return frame.split(" (", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("--thread-has", required=True)
    parser.add_argument("--phases", default="")
    parser.add_argument("--expand", default="")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    phases = [p for p in args.phases.split(",") if p]
    expand = [p for p in args.expand.split(",") if p]
    stacks: dict[str, list[tuple[list[str], int]]] = collections.defaultdict(list)
    names: dict[str, str] = {}
    for line in open(args.raw):
        match = LINE.match(line.rstrip("\n"))
        if match is None:
            continue
        tid, name, frames, count = match.groups()
        names[tid] = name
        stacks[tid].append((frames.split(";") if frames else [], int(count)))
    all_traces = sum(count for entries in stacks.values() for _, count in entries)
    print(
        "GIL holder traces by thread: "
        + ", ".join(
            f"{names[tid]} {100 * sum(c for _, c in entries) / all_traces:.1f} %"
            for tid, entries in sorted(stacks.items(), key=lambda item: -sum(c for _, c in item[1]))[:8]
        )
    )
    for tid, entries in stacks.items():
        if not any(function_of(f) == args.thread_has for frames, _ in entries for f in frames):
            continue
        total = sum(count for _, count in entries)
        print(f"\nthread {names[tid]} ({tid}): {total} holder traces, {100 * total / all_traces:.1f} % of all")
        inclusive: collections.Counter = collections.Counter()
        leaves: collections.Counter = collections.Counter()
        children = {p: collections.Counter() for p in expand}
        for frames, count in entries:
            functions = [function_of(f) for f in frames]
            for phase in set(functions) & set(phases):
                inclusive[phase] += count
            if frames:
                leaves[frames[-1]] += count
            for phase in expand:
                if phase in functions:
                    index = len(functions) - 1 - functions[::-1].index(phase)
                    child = frames[index + 1] if index + 1 < len(frames) else "(self)"
                    children[phase][child] += count
        if phases:
            print("  phases, inclusive % of the thread's holder traces:")
            print("    " + ", ".join(f"{p} {100 * inclusive[p] / total:.1f}" for p in phases if inclusive[p]))
        print("  top leaf frames (% of the thread's holder traces):")
        for frame, count in leaves.most_common(args.top):
            print(f"    {100 * count / total:5.1f}  {frame}")
        for phase in expand:
            if not children[phase]:
                continue
            phase_total = sum(children[phase].values())
            print(f"  under {phase} ({100 * phase_total / total:.1f} %), direct children:")
            for frame, count in children[phase].most_common(12):
                print(f"    {100 * count / total:5.1f}  {frame}")


if __name__ == "__main__":
    main()
