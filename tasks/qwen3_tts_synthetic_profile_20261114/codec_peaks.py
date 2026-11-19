"""Per arm and rate of a bar_overload_pair run: the arena's slots, its peak held rows, and how many acquires it
refused and streams it sent to the left-context decoder during the rate (counter rise from first to last sample).

usage: python3 codec_peaks.py OUT
"""

import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    print("seed arm rate slots peak_active exhausted_rise fallback_rise samples")
    for procs in sorted(root.glob("seed*/*/rps*.procs.jsonl")):
        samples = [json.loads(line).get("codec") or {} for line in procs.read_text().splitlines() if line.strip()]
        samples = [sample for sample in samples if sample.get("active_slots") is not None]
        if not samples:
            print(f"{procs.parent.parent.name} {procs.parent.name} {procs.name.split('.')[0]} no codec samples")
            continue
        else:
            pass
        rise = {
            name: (samples[-1].get(name) or 0) - (samples[0].get(name) or 0)
            for name in ("exhausted", "left_context_fallbacks")
        }
        print(
            f"{procs.parent.parent.name} {procs.parent.name} {procs.name.split('.')[0]} {samples[0].get('slots')} "
            f"{max(sample['active_slots'] for sample in samples)} {rise['exhausted']} "
            f"{rise['left_context_fallbacks']} {len(samples)}"
        )


if __name__ == "__main__":
    main()
