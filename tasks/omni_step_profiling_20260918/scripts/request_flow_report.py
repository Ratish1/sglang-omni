"""Summarise the request_flow probe: the preprocessing split and the vocoder census.

usage: request_flow_report.py <prefix>   (reads <prefix>.*.jsonl)
"""

import glob
import json
import statistics
import sys
from collections import Counter, defaultdict

records = defaultdict(list)
for path in sorted(glob.glob(f"{sys.argv[1]}.*.jsonl")):
    with open(path) as lines:
        for line in lines:
            record = json.loads(line)
            records[record["kind"]].append(record)


def row(label, values):
    if not values:
        return
    values = sorted(values)
    p95 = values[min(len(values) - 1, int(0.95 * len(values)))]
    print(
        f"  {label:<34}{len(values):>7}{statistics.fmean(values) * 1000:>10.2f}"
        f"{values[len(values) // 2] * 1000:>10.2f}{p95 * 1000:>10.2f}"
        f"{sum(values):>10.2f}"
    )


header = (
    f"  {'part':<34}{'n':>7}{'mean ms':>10}{'p50 ms':>10}{'p95 ms':>10}{'sum s':>10}"
)
print("preprocessing, per request (wall = elapsed, cpu = this thread on a core)")
print(header)
row("prepare wall", [r["wall_s"] for r in records["prepare"]])
row("prepare cpu", [r["cpu_s"] for r in records["prepare"]])
row("reference encode wall", [r["wall_s"] for r in records["reference"]])
row("reference encode cpu", [r["cpu_s"] for r in records["reference"]])
for part in ("normalize", "resample", "speaker"):
    for clock in ("wall", "cpu"):
        row(
            f"  {part} {clock}",
            [r.get(f"{part}_{clock}_s", 0.0) for r in records["reference"]],
        )
row(
    "  rest of reference wall (code wait)",
    [
        r["wall_s"]
        - sum(r.get(f"{p}_wall_s", 0.0) for p in ("normalize", "resample", "speaker"))
        for r in records["reference"]
    ],
)
row("ref code encode wall (batcher)", [r["encode_wall_s"] for r in records["ref_code"]])
row("ref code encode cpu (batcher)", [r["encode_cpu_s"] for r in records["ref_code"]])
row("ref code stream sync", [r["wall_s"] for r in records["ref_code_sync"]])
batches = Counter(r["batch"] for r in records["ref_code_sync"])
print(f"  ref code batch sizes: {dict(sorted(batches.items()))}")

print("\nvocoder cohorts (one decode call of the scheduler)")
cohorts = records["cohort"]
for initial in (True, False):
    group = [c for c in cohorts if c["initial_stream"] is initial]
    if not group:
        continue
    name = "initial stream (bootstrap)" if initial else "follow-up streams"
    paths = Counter(c["path"] for c in group)
    print(f"  {name}: {len(group)} cohorts, paths {dict(paths)}")
    keys = Counter((c["width"], c["rows"]) for c in group)
    print("    (width, rows): count, top 14:", keys.most_common(14))
    splits = Counter(len(c["split"]) for c in group if c["split"])
    if splits:
        print(f"    windows per split cohort: {dict(sorted(splits.items()))}")
    row(f"  {name[:20]} host wall", [c["wall_s"] for c in group])

print("\nvocoder graph replays")
replays = records["replay"]
hits = [r for r in replays if r["hit"]]
misses = Counter((r["mode"], r["width"], r["rows"]) for r in replays if not r["hit"])
live = sum(r["rows"] for r in hits)
padded = sum((r["bucket"] or r["rows"]) - r["rows"] for r in hits)
print(f"  replays {len(hits)}, misses {sum(misses.values())}")
if hits:
    print(
        f"  live rows {live}, padded rows {padded} "
        f"({100.0 * padded / (live + padded):.1f}% of replayed rows are padding)"
    )
    frames = sum(r["rows"] * r["width"] for r in hits)
    padded_frames = sum(
        ((r["bucket"] or r["rows"]) - r["rows"]) * r["width"] for r in hits
    )
    print(
        f"  live frames {frames}, padded frames {padded_frames} "
        f"({100.0 * padded_frames / (frames + padded_frames):.1f}%)"
    )
    print(f"  rows per replay: mean {live / len(hits):.2f}")
    for mode in sorted({r["mode"] for r in hits}):
        group = [r for r in hits if r["mode"] == mode]
        keys = Counter((r["width"], r["rows"], r["bucket"]) for r in group)
        print(f"  {mode}: {len(group)} replays; (width, rows, bucket) top 12:")
        print("    ", keys.most_common(12))
if misses:
    print("  misses (mode, width, rows):", misses.most_common(8))

if cohorts:
    span = max(c["t"] for c in cohorts) - min(c["t"] for c in cohorts)
    print(
        f"\n  cohorts per second {len(cohorts) / span:.1f}, replays per second {len(hits) / span:.1f}"
    )
