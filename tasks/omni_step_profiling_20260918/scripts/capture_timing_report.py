"""Sum the capture_timing probe's rows per runner: where a vocoder capture pass spends time.

usage: capture_timing_report.py <prefix>   (reads <prefix>.*.jsonl)
"""

import glob
import json
import sys

rows, runners = [], []
for path in sorted(glob.glob(f"{sys.argv[1]}.*.jsonl")):
    with open(path) as lines:
        for line in lines:
            record = json.loads(line)
            (runners if "runner_total_s" in record else rows).append(record)

rows.sort(key=lambda r: r.get("at", 0.0))
passes: list[list[dict]] = []
for r in rows:
    if "key_total_s" not in r:
        continue
    if (
        not passes
        or passes[-1][-1]["mode"] != r["mode"]
        or (
            (r["batch"], r["frames"])
            > (passes[-1][-1]["batch"], passes[-1][-1]["frames"])
        )
    ):
        passes.append([])
    passes[-1].append(r)

columns = (
    "keys",
    "precompile",
    "warmup 1",
    "warmups 2+",
    "capture",
    "gc.collect",
    "empty_cache",
    "sum",
)
print(f"{'runner pass':<14}" + "".join(f"{c:>12}" for c in columns))
total = [0.0] * 7
for index, keys in enumerate(passes):
    parts = [
        sum(sum(k.get("precompile_s", [])) for k in keys),
        sum(k.get("warmup_s", [0.0])[0] for k in keys),
        sum(sum(k.get("warmup_s", [0.0])[1:]) for k in keys),
        sum(k["key_total_s"] - k.get("warmup_total_s", 0.0) for k in keys),
        sum(k.get("gc_collect_s", 0.0) for k in keys),
        sum(k.get("empty_cache_s", 0.0) for k in keys),
    ]
    parts.append(sum(parts))
    total = [a + b for a, b in zip(total, parts)]
    print(
        f"{index} {keys[0]['mode']:<12}{len(keys):>12}"
        + "".join(f"{p:>12.2f}" for p in parts)
    )
print(
    f"{'all':<14}{sum(len(k) for k in passes):>12}"
    + "".join(f"{p:>12.2f}" for p in total)
)

print("\nrunner totals (wall):")
for record in runners:
    print(
        f"  {record['mode']:<8}{record['runner_total_s']:>8.2f} s  "
        f"enabled={record['enabled']}  thread={record.get('thread')}"
    )

print("\nslowest keys:")
for k in sorted(rows, key=lambda r: -r.get("key_total_s", 0.0))[:8]:
    warmups = ", ".join(f"{w * 1000:.0f}" for w in k.get("warmup_s", []))
    print(
        f"  {k['mode']:<7} frames={k['frames']:<3} batch={k['batch']:<2} "
        f"compiled={k['compiled']!s:<5} key={k['key_total_s']:.2f} s  "
        f"precompile={sum(k.get('precompile_s', [])):.2f} s  warmups ms=[{warmups}]  "
        f"gc={k.get('gc_collect_s', 0.0) * 1000:.0f} ms  "
        f"empty_cache={k.get('empty_cache_s', 0.0) * 1000:.0f} ms"
    )
