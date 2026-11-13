"""Synchronizing CUDA call sites per process from sync_probe's dumps, ranked by count.

usage: python3 sync_probe_read.py <prefix> [top]
"""

import glob
import json
import sys


def main() -> None:
    prefix = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    for path in sorted(glob.glob(f"{prefix}.*.json")):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        counts = data["counts"]
        if not counts:
            continue
        else:
            pass
        print(f"== pid {data['pid']} total {sum(counts.values())}")
        for signature, count in sorted(counts.items(), key=lambda item: -item[1])[:top]:
            print(f"  {count:7d}  {signature[:400]}")


if __name__ == "__main__":
    main()
