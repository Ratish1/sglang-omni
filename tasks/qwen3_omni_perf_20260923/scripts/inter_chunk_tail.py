"""Per boot of a SeedTTS round: every inter chunk gap pooled (count, mean, p99, p99.9), gaps over 0.5 s and 1.0 s, and the summed playback underrun, so a p99 change shows whether stalls grew in number or length.

usage: python inter_chunk_tail.py ROUND [ROUND ...]
"""

import glob
import json
import sys


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


for round_name in sys.argv[1:]:
    for boot in sorted(glob.glob(f"/data/ab/{round_name}/card*_*")):
        with open(f"{boot}/seedtts_en/speed_results.json") as results_file:
            records = json.load(results_file)["per_request"]
        gaps = [
            gap for record in records for gap in (record.get("inter_chunk_s") or [])
        ]
        underrun = sum(
            record.get("max_playback_underrun_s") or 0.0 for record in records
        )
        print(
            f"{round_name:8s} {boot.split('/')[-1]:14s} gaps {len(gaps):6d} mean {sum(gaps) / len(gaps):.4f}"
            f" p99 {percentile(gaps, 0.99):.4f} p99.9 {percentile(gaps, 0.999):.4f}"
            f" >0.5s {sum(gap > 0.5 for gap in gaps):5d} >1.0s {sum(gap > 1.0 for gap in gaps):4d}"
            f" underrun sum {underrun:7.2f} s"
        )
