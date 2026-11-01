"""The colocated card's memory, stage by stage, from one run_bench_boot.sh boot (check 5, 11-01).

Reads the boot's serve.log (each AR stage's sglang_ar_startup and sglang_ar_started lines with its pid and memory,
the colocated KV profile lines, SGLang's KV pool lines) and apps.csv (every process's card memory every 2 s).
Prints, per process on the card: its role (thinker and talker_ar by the pid their startup line logs, the rest
listed as other GPU stage processes), its memory after load, its peak while serving and its configured budget;
then the card's total at its peak sample against the card's capacity.

usage: python3 memory_budget.py BOOT_DIR
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    boot = Path(sys.argv[1])
    log = (boot / "serve.log").read_text(errors="replace")
    roles: dict[int, str] = {}
    for match in re.finditer(r"sglang_ar_startup stage=(\S+) .*?pid=(\d+)", log):
        roles[int(match.group(2))] = match.group(1)
    print("AR stage memory lines:")
    for line in log.splitlines():
        if (
            "sglang_ar_started" in line
            or "SGLang AR memory profile" in line
            or "KV Cache is allocated" in line
            or "Code2Wav incremental decode" in line
        ):
            print("  " + line.split("] ", 1)[-1][:400])
        else:
            pass
    samples: dict[str, dict[int, int]] = defaultdict(dict)
    for line in (boot / "apps.csv").read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        else:
            pass
        clock = parts[0].split()[0]
        try:
            pid = int(parts[1])
            mib = int(parts[2].split()[0])
        except ValueError:
            continue
        samples[clock][pid] = mib
    if not samples:
        print("apps.csv has no samples")
        return
    else:
        pass
    peak: dict[int, int] = defaultdict(int)
    first: dict[int, int] = {}
    for clock in sorted(samples):
        for pid, mib in samples[clock].items():
            peak[pid] = max(peak[pid], mib)
            first.setdefault(pid, mib)
    card_peak_clock = max(samples, key=lambda clock: sum(samples[clock].values()))
    print("\nprocesses on the card (MiB): first sample, peak")
    for pid in sorted(peak, key=lambda p: -peak[p]):
        print(f"  pid {pid:7d} {roles.get(pid, 'other GPU stage'):16s} first {first[pid]:7d} peak {peak[pid]:7d}")
    card_total = sum(samples[card_peak_clock].values())
    capacity = None
    gpus = boot / "gpus_before.txt"
    if gpus.exists():
        match = re.search(r"(\d+)MiB\s*/\s*(\d+)MiB", gpus.read_text())
        capacity = int(match.group(2)) if match else None
    else:
        pass
    print(
        f"\ncard peak at {card_peak_clock}: {card_total} MiB over {len(samples[card_peak_clock])} processes"
        + (f", capacity {capacity} MiB, free at peak {capacity - card_total} MiB" if capacity else "")
    )
    print(f"sum of per-process peaks: {sum(peak.values())} MiB")


if __name__ == "__main__":
    main()
