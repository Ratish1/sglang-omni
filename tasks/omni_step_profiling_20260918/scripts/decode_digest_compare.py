"""Compare two servers' vocoder decode digests (decode_digest/sitecustomize.py) decode by decode.

usage: python3 decode_digest_compare.py DIGEST_DIR_A DIGEST_DIR_B

Each side's vocoder process is the one with the most records. Reports: how many decodes each side made per kind and
mode; walking both in call order, the first decode whose input codes differ and the first whose input codes match but
whose output differs; and, joining on the input codes (the k-th occurrence of an input on each side), how many matched
inputs gave different outputs, by kind, mode and width.
"""

import collections
import json
import sys
from pathlib import Path


def load(directory: str) -> list[dict]:
    files = sorted(Path(directory).glob("*.jsonl"), key=lambda path: path.stat().st_size, reverse=True)
    return [json.loads(line) for line in files[0].read_text(encoding="utf-8").splitlines()]


def label(entry: dict) -> str:
    width = entry["codes_shape"][-1]
    rows = entry["codes_shape"][0]
    mode = entry.get("mode") or ("compiled" if entry.get("compiled") else "eager")
    return f"{entry['kind']} {mode} width {width} rows {rows}"


def main() -> None:
    first, second = load(sys.argv[1]), load(sys.argv[2])
    for name, entries in (("A", first), ("B", second)):
        counts = collections.Counter(label(entry) for entry in entries)
        print(f"{name}: {len(entries)} decodes; " + ", ".join(f"{key}: {value}" for key, value in sorted(counts.items())))
    first_input_diff = first_output_diff = None
    for index, (a, b) in enumerate(zip(first, second)):
        same_input = (a["kind"], a.get("mode"), a["codes_shape"], a["codes"]) == (
            b["kind"], b.get("mode"), b["codes_shape"], b["codes"])
        if not same_input and first_input_diff is None:
            first_input_diff = (index, a, b)
        elif same_input and a["out"] != b["out"] and first_output_diff is None:
            first_output_diff = (index, a, b)
        else:
            pass
    for title, found in (("first input difference", first_input_diff), ("first output difference on equal input", first_output_diff)):
        if found is None:
            print(f"{title}: none in {min(len(first), len(second))} aligned decodes")
        else:
            index, a, b = found
            print(f"{title} at aligned decode {index}: A {label(a)} thread {a['thread']} | B {label(b)} thread {b['thread']}")
    keyed: dict[tuple, list[str]] = collections.defaultdict(list)
    for entry in second:
        keyed[(entry["kind"], entry.get("mode"), tuple(entry["codes_shape"]), entry["codes"])].append(entry["out"])
    seen: collections.Counter = collections.Counter()
    matched = differing = unmatched = 0
    differing_by: collections.Counter = collections.Counter()
    for entry in first:
        key = (entry["kind"], entry.get("mode"), tuple(entry["codes_shape"]), entry["codes"])
        occurrence = seen[key]
        seen[key] += 1
        outputs = keyed.get(key, [])
        if occurrence >= len(outputs):
            unmatched += 1
        elif outputs[occurrence] == entry["out"]:
            matched += 1
        else:
            differing += 1
            differing_by[label(entry)] += 1
    print(f"joined on input: {matched} equal outputs, {differing} different outputs, {unmatched} inputs only on A")
    for key, value in differing_by.most_common(12):
        print(f"  different: {key}: {value}")


if __name__ == "__main__":
    main()
