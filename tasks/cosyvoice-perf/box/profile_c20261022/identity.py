"""Byte identity of a pair's two arms per order and cell: the requests both completed, those with the same completion
token count, and those whose audio is byte identical.
usage: python3 identity.py <pair dir>"""

import hashlib
import json
import sys
from pathlib import Path


def records(cell):
    with open(cell / "speed_results.json") as handle:
        return {row["id"]: row for row in json.load(handle)["per_request"] if row["is_success"]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pair = Path(sys.argv[1])
    for order in ("a", "b"):
        for cell in sorted(path.name for path in (pair / order / "gpu0").iterdir() if path.is_dir()):
            first, second = records(pair / order / "gpu0" / cell), records(pair / order / "gpu1" / cell)
            ids = sorted(first.keys() & second.keys())
            same = [clip for clip in ids if first[clip]["completion_tokens"] == second[clip]["completion_tokens"]]
            identical = sum(digest(first[clip]["wav_path"]) == digest(second[clip]["wav_path"]) for clip in same)
            print(f"{pair.name} {order} {cell}: requests {len(ids)}, same token count {len(same)}, byte identical {identical}")


if __name__ == "__main__":
    main()
