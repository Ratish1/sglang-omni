"""Interpreter lock ownership by thread role from the kit's pyspygil pass (py-spy record --gil --threads --format raw).

usage: python3 gil_threads.py RUN_DIR [--cell c16_pyspygil]

Each server process's raw file has one line per distinct stack, "thread (0x..);frame;...;frame count", sampled at the
pass's rate for its duration, only while the thread holds the lock. A thread's role comes from the files its stacks
pass through (thinker, talker, code2wav, the stage event loop, other). Prints each process's and role's share of the
window: the fraction of ticks that role held the lock.
"""

import argparse
import collections
import json
import pathlib
import re

THREAD = re.compile(r"^thread \((0x[0-9a-fA-F]+)\)")
ROLE_MARKS = (
    ("code2wav", ("code2wav",)),
    ("talker", ("talker_scheduler.py", "talker_model_runner.py", "components/talker.py")),
    ("thinker", ("thinker_model_runner.py", "sglang_thinker.py", "thinker_scheduler.py")),
    ("event loop", ("asyncio/base_events.py", "asyncio/events.py")),
)


def role_of(files: set[str]) -> str:
    for role, marks in ROLE_MARKS:
        if any(mark in name for name in files for mark in marks):
            return role
        else:
            pass
    return "other"


def read_process(path: pathlib.Path) -> dict[str, int]:
    samples_by_thread: collections.Counter[str] = collections.Counter()
    files_by_thread: dict[str, set[str]] = collections.defaultdict(set)
    for line in path.read_text(errors="replace").splitlines():
        stack, _, count = line.rpartition(" ")
        if not stack or not count.isdigit():
            continue
        else:
            pass
        frames = stack.split(";")
        match = THREAD.match(frames[0])
        thread = match.group(1) if match else frames[0]
        samples_by_thread[thread] += int(count)
        for frame in frames[1:]:
            opening = frame.rfind("(")
            if opening >= 0:
                files_by_thread[thread].add(frame[opening + 1 :].split(":")[0])
            else:
                pass
    by_role: collections.Counter[str] = collections.Counter()
    for thread, samples in samples_by_thread.items():
        by_role[role_of(files_by_thread[thread])] += samples
    return dict(by_role)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--cell", default="c16_pyspygil")
    args = parser.parse_args()
    cell = pathlib.Path(args.run) / args.cell
    window = json.loads((cell / "pyspy.json").read_text())
    ticks = window["rate"] * window["duration"]
    print(f"{args.run} {args.cell}: {ticks} ticks ({window['rate']} Hz for {window['duration']} s)")
    for path in sorted(cell.glob("pyspy_*.txt")):
        by_role = read_process(path)
        total = sum(by_role.values())
        if total == 0:
            continue
        else:
            pass
        roles = ", ".join(f"{role} {100 * count / ticks:.1f} %" for role, count in sorted(by_role.items(), key=lambda item: -item[1]))
        print(f"  {path.stem}: lock held {100 * total / ticks:.1f} % ({roles})")


if __name__ == "__main__":
    main()
