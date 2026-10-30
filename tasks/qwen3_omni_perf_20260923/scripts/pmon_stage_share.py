"""Per stage SM and memory-bandwidth share of one boot, from nvidia-smi pmon samples.

The boot's request event files are named events_<stage>_<pid>.jsonl, which maps every stage
process id to its stages (a process serving two stages lists both). Over the samples taken while
the boot's benchmark arms ran (gen start to the last gen rc in progress.txt), it prints each
process's mean SM % and memory %, and its share of the summed SM %, so two boots compare on how
the card's compute was divided between stage processes.

usage: python pmon_stage_share.py BOOT_DIR PMON_LOG [BOOT_DIR PMON_LOG ...]
"""

from __future__ import annotations

import collections
import glob
import os
import re
import sys


def stage_by_pid(boot_dir: str) -> dict[str, str]:
    stages: dict[str, list[str]] = collections.defaultdict(list)
    for path in glob.glob(f"{boot_dir}/events/events_*.jsonl"):
        match = re.match(r"events_(.+)_(\d+)\.jsonl$", os.path.basename(path))
        if match is not None:
            stages[match.group(2)].append(match.group(1))
        else:
            pass
    return {pid: "+".join(sorted(names)) for pid, names in stages.items()}


def benchmark_window(boot_dir: str) -> tuple[str, str]:
    # a gen line can follow the profiler's reply on the same line, so the pattern is searched anywhere
    with open(f"{boot_dir}/progress.txt") as progress_file:
        progress = progress_file.read()
    starts = re.findall(r"gen \S+ start (\d\d:\d\d:\d\d)", progress)
    ends = re.findall(r"gen \S+ rc \d+ (\d\d:\d\d:\d\d)", progress)
    return min(starts), max(ends)


def main() -> None:
    arguments = sys.argv[1:]
    for boot_dir, pmon_log in zip(arguments[0::2], arguments[1::2]):
        names = stage_by_pid(boot_dir)
        start, end = benchmark_window(boot_dir)
        sm_samples: dict[str, list[float]] = collections.defaultdict(list)
        memory_samples: dict[str, list[float]] = collections.defaultdict(list)
        with open(pmon_log) as pmon_file:
            for line in pmon_file:
                words = line.split()
                # date-stamped pmon rows: date time gpu pid type sm mem enc dec ... command
                if len(words) < 7 or words[0].startswith("#") or words[3] == "-":
                    continue
                else:
                    pass
                if not (start <= words[1] <= end):
                    continue
                else:
                    pass
                pid = words[3]
                sm = float(words[5]) if words[5] != "-" else 0.0
                memory = float(words[6]) if words[6] != "-" else 0.0
                sm_samples[pid].append(sm)
                memory_samples[pid].append(memory)
        total_sm = sum(sum(values) for values in sm_samples.values()) or 1.0
        print(f"== {boot_dir} window {start} to {end}")
        for pid, values in sorted(sm_samples.items(), key=lambda item: -sum(item[1])):
            print(
                f"  {names.get(pid, 'pid ' + pid):28s} samples {len(values):5d}"
                f" sm mean {sum(values) / len(values):6.2f} % mem mean"
                f" {sum(memory_samples[pid]) / len(values):6.2f} % share of sm {100 * sum(values) / total_sm:6.2f} %"
            )


if __name__ == "__main__":
    main()
