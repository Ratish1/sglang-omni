# Startup E reduction (11-01): one boot's GC census split by the boot log's phases (boot_split.py's marks: launcher,
# preprocessing, vocoder, engine, serve), for the stage process and the launcher: generation 2 collections (count,
# seconds, the longest), generation 1 (count, seconds) and generation 0 seconds, plus the process totals.
# usage: python3 gc_split.py <serve log> <gc census dir>
import re
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
    return datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()


def phase_bounds(log):
    lines = [line for line in log.read_text(errors="replace").splitlines() if re.match(r"\d{4}-\d\d-\d\d \d\d:", line)]
    bounds = [("start", stamp(lines[0]))]
    for name, mark in MARKS:
        bounds.append((name, stamp(next(line for line in lines if mark in line))))
    return bounds


def read(path):
    header = ""
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            header = line[2:]
        else:
            fields = line.split("\t")
            moment, generation, seconds, collected, young_count, young_seconds = fields[:6]
            trigger = fields[6] if len(fields) > 6 else ""
            rows.append((float(moment), int(generation), float(seconds), int(collected), int(young_count),
                         float(young_seconds), trigger))
    return header, rows


def summarize(name, rows, bounds):
    print(f"== {name}")
    print("phase | gen2 count | gen2 s | longest gen2 s | gen1 count | gen1 s | gen0 s")
    previous_young = 0.0
    for (_, began), (phase, ended) in zip(bounds, bounds[1:]):
        inside = [row for row in rows if began <= row[0] < ended]
        old = [row for row in inside if row[1] == 2]
        middle = [row for row in inside if row[1] == 1]
        young = max((row[5] for row in inside), default=previous_young)
        print(
            f"{phase} | {len(old)} | {sum(row[2] for row in old):.2f} | {max((row[2] for row in old), default=0):.2f} | "
            f"{len(middle)} | {sum(row[2] for row in middle):.2f} | {young - previous_young:.2f}"
        )
        previous_young = young
    old = [row for row in rows if row[1] == 2]
    print(f"total | {len(old)} | {sum(row[2] for row in old):.2f} | | {sum(row[1] == 1 for row in rows)} | "
          f"{sum(row[2] for row in rows if row[1] == 1):.2f} | {max((row[5] for row in rows), default=0):.2f}")
    for row in old:
        phase = next((name for (_, began), (name, ended) in zip(bounds, bounds[1:]) if began <= row[0] < ended), "after")
        print(f"  gen2 {phase} {row[2]:.2f} s: {row[6][:400]}")


def main():
    log, directory = Path(sys.argv[1]), Path(sys.argv[2])
    bounds = phase_bounds(log)
    processes = {}
    for path in directory.glob("gc_*.tsv"):
        header, rows = read(path)
        processes[path.name] = (header, rows)
    # sitecustomize runs before -m and -c are resolved, so a spawned stage process shows only --multiprocessing-fork
    # and the launcher only its own arguments
    stage = max((item for item in processes.items() if "--multiprocessing-fork" in item[1][0]),
                key=lambda item: len(item[1][1]), default=None)
    launcher = next((item for item in processes.items() if "--model-path" in item[1][0]), None)
    for label, item in (("stage process", stage), ("launcher", launcher)):
        if item is None:
            print(f"== {label}: not found")
        else:
            summarize(f"{label} {item[0]}", item[1][1], bounds)
    others = sum(sum(row[2] for row in rows if row[1] == 2) for name, (header, rows) in processes.items()
                 if item_name_not_in(name, stage, launcher))
    print(f"other processes ({len(processes) - 2}): gen2 seconds {others:.2f}")


def item_name_not_in(name, *items):
    return all(item is None or item[0] != name for item in items)


if __name__ == "__main__":
    main()
