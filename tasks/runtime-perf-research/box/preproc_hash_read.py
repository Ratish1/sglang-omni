"""Compare the preprocessing tensors of every request across runs, from preproc_hash_probe's JSON lines.

usage: python3 preproc_hash_read.py label=<glob> [label=<glob> ...]

A request is named by the hash of its input ids, so one question has one key on every server. Each key must carry one
set of tensor hashes over every record of every run; any key with two prints the tensors that differ and where.
"""

import collections
import glob
import json
import sys


def main() -> None:
    by_key = collections.defaultdict(list)
    for spec in sys.argv[1:]:
        label, pattern = spec.split("=", 1)
        records = 0
        threads = collections.Counter()
        for path in sorted(glob.glob(pattern)):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    by_key[record["key"]].append((label, record))
                    threads[record["thread"].rsplit("_", 1)[0]] += 1
                    records += 1
        print(f"{label}: {records} records, threads {dict(threads)}")
    differing = 0
    for key, entries in sorted(by_key.items()):
        prints = collections.defaultdict(list)
        for label, record in entries:
            signature = json.dumps(record["tensors"], sort_keys=True)
            prints[signature].append(label)
        if len(prints) == 1:
            continue
        else:
            pass
        differing += 1
        variants = [json.loads(signature) for signature in prints]
        paths = sorted(
            {
                path
                for variant in variants
                for path in variant
                if any(v.get(path) != variants[0].get(path) for v in variants)
            }
        )
        print(f"key {key}: {len(prints)} variants over {len(entries)} records")
        for signature, labels in prints.items():
            print(f"  {collections.Counter(labels)}")
        for path in paths[:8]:
            print(f"  differs: {path}")
    print(
        f"\n{len(by_key)} keys, {differing} with more than one set of tensor hashes, "
        f"{sum(len(entries) for entries in by_key.values())} records"
    )


if __name__ == "__main__":
    main()
