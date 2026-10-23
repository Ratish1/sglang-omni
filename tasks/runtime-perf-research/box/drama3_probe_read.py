"""Read the drama3 probe: the short results of a full round, and the scheduler aborts around each.

usage: python3 drama3_probe_read.py <probe dir> [window ms]
"""

import glob
import json
import sys

probe_dir = sys.argv[1]
window_ns = int(float(sys.argv[2]) * 1e6) if len(sys.argv) > 2 else 400_000_000
records = []
for path in glob.glob(f"{probe_dir}/*.jsonl"):
    with open(path) as source:
        records.extend(json.loads(line) for line in source)
records.sort(key=lambda record: record["t"])
kinds = {}
for record in records:
    kinds[record["k"]] = kinds.get(record["k"], 0) + 1
print(kinds)
short = [
    record
    for record in records
    if record["k"] == "result"
    and str(record["inputs"]).startswith("line")
    and record["sampled"] != 20
]
print("short results:", len(short))
for record in short:
    print(
        json.dumps(
            {
                key: record.get(key)
                for key in (
                    "rid",
                    "inputs",
                    "sampled",
                    "last_main",
                    "output_ids",
                    "max_new_tokens",
                    "finished_reason",
                    "finish_reason",
                    "thread",
                )
            }
        )
    )
streams = [
    record
    for record in records
    if record["k"] == "result" and record["inputs"] == "hello"
]
print(
    "stream results:",
    len(streams),
    "reasons:",
    sorted({json.dumps(record["finished_reason"]) for record in streams}),
)
for record in short:
    print("aborts around", record["inputs"], record["rid"][:12], ":")
    for abort in records:
        if abort["k"] == "abort" and abs(abort["t"] - record["t"]) < window_ns:
            print(
                "  dt_ms",
                round((abort["t"] - record["t"]) / 1e6, 1),
                "aborted",
                abort["rid"][:12],
                "running",
                len(abort["running"]),
                [
                    (rid[:8], output_count, reason)
                    for rid, output_count, reason in abort["running"]
                ],
            )
        else:
            pass
