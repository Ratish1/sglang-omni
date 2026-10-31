# Block 46 reduction: each boot's log split into launcher (first line to the stage process's CUDA IPC warm-up),
# preprocessing (to the vocoder's build), vocoder (to the engine's build), engine (to pipeline ready) and serve (to the
# server's ready line), plus the health time from queue.log; then each arm's median per phase over its rounds.
# usage: python3 boot_split.py <block46 dir>
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

MARKS = (
    ("launcher", "Warmed CUDA IPC"),
    ("preprocessing", "Building scheduler for vocoder"),
    ("vocoder", "Building scheduler for tts_engine"),
    ("engine", "Process pipeline ready with stages"),
    ("serve", "the server is ready"),
)


def stamp(line):
    return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f")


def split(log):
    lines = [line for line in log.read_text(errors="replace").splitlines() if re.match(r"\d{4}-\d\d-\d\d \d\d:", line)]
    previous = stamp(lines[0])
    phases = {}
    for name, mark in MARKS:
        hit = next(line for line in lines if mark in line)
        phases[name] = (stamp(hit) - previous).total_seconds()
        previous = stamp(hit)
    return phases


def main():
    directory = Path(sys.argv[1])
    health = {match.group(1): float(match.group(2)) for match in re.finditer(r"(warm_\S+) healthy after ([0-9.]+) s", (directory / "queue.log").read_text())}
    arms = {}
    for label, seconds in sorted(health.items()):
        arm = label.rsplit("_", 1)[0]
        phases = split(directory / "logs" / f"serve_{label}.log")
        phases["health"] = seconds
        arms.setdefault(arm, []).append(phases)
        print(label, " ".join(f"{name} {value:.1f}" for name, value in phases.items()))
    print()
    names = [name for name, _ in MARKS] + ["health"]
    print("median | " + " | ".join(names))
    for arm, runs in arms.items():
        print(f"{arm} ({len(runs)}) | " + " | ".join(f"{statistics.median(run[name] for run in runs):.1f}" for name in names))


if __name__ == "__main__":
    main()
