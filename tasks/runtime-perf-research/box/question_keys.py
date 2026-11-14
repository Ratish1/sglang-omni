"""Input id keys of Video-MME questions on one boot, from its c1 pass order and preproc_hash_probe's records.

usage: python3 question_keys.py <boot dir> 005-3,017-1,006-1

At c1 the client sends the questions one at a time in dataset order, so the boot's hash records hold a run whose prompt
token counts equal the pass's per question counts in order; question i takes that run's i-th key. Prints the keys
comma separated; fails unless exactly one run matches.
"""

import glob
import json
import sys
from pathlib import Path


def main() -> None:
    boot, wanted = Path(sys.argv[1]), sys.argv[2].split(",")
    samples = json.loads(
        (boot / "c1_clean/bench/rep0/videomme_results.json").read_text(encoding="utf-8")
    )["per_sample"]
    counts = [sample["prompt_tokens"] for sample in samples]
    records = []
    for path in glob.glob(str(boot / "hash/h.*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            records += [json.loads(line) for line in handle]
    records.sort(key=lambda record: record["time"])
    starts = [
        start
        for start in range(len(records) - len(counts) + 1)
        if [r["prompt_tokens"] for r in records[start : start + len(counts)]] == counts
    ]
    assert starts, "no run of records matches the pass's token counts"
    keys = {
        sample["question_id"]: records[starts[0] + index]["key"]
        for index, sample in enumerate(samples)
    }
    for start in starts[1:]:
        assert all(
            records[start + index]["key"] == keys[sample["question_id"]]
            for index, sample in enumerate(samples)
        ), "two matching runs disagree"
    print(",".join(keys[question] for question in wanted))


if __name__ == "__main__":
    main()
