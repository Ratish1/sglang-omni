import collections
import glob
import json
import sys

buckets = [
    (1, "1 B (no tensors)"),
    (64, "2 to 64 B"),
    (4096, "65 B to 4 KiB"),
    (16384, "4 to 16 KiB"),
    (65536, "16 to 64 KiB"),
    (1 << 20, "64 KiB to 1 MiB"),
    (1 << 62, "over 1 MiB"),
]
for run in sys.argv[1:]:
    counts = collections.defaultdict(collections.Counter)
    for path in glob.glob(f"/data/sizes/{run}/card*_main08/records/*.jsonl"):
        with open(path) as records_file:
            for line in records_file:
                record = json.loads(line)
                if record["function"] != "pack_tensors":
                    continue
                label = next(
                    name for limit, name in buckets if record["bytes"] <= limit
                )
                counts[record["process"]][label] += 1
    print(f"== {run} (main, payload puts per process)")
    for process in sorted(counts):
        total = sum(counts[process].values())
        print(
            f"  {process:24s} {total:5d}: "
            + ", ".join(
                f"{name} {counts[process][name]}"
                for _, name in buckets
                if counts[process][name]
            )
        )
