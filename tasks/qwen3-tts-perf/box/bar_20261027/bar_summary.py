"""One block per rate of a queue_baseline.sh arm: every field of the client's summary.json whose flattened name mentions
audible TTFA, first byte, leading silence, underruns, stalls, failures, completions or the achieved rate.

usage: python3 bar_summary.py ARM_DIR
"""

import argparse
import json
import re
from pathlib import Path

FIELDS = re.compile(r"audible|ttfa|ttfb|first_byte|leading|silence|underrun|stall|fail|success|complet|achieved|rps", re.I)


def flatten(prefix: str, value: object, out: dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), inner, out)
    elif isinstance(value, list):
        out[prefix] = f"list of {len(value)}"
    else:
        out[prefix] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", type=Path)
    args = parser.parse_args()
    runs = sorted(args.arm.glob("seed*_rps*/summary.json"), key=lambda path: float(path.parent.name.split("rps")[1]))
    for path in runs:
        fields: dict[str, object] = {}
        flatten("", json.loads(path.read_text(encoding="utf-8")), fields)
        print(f"== {path.parent.name}")
        for name, value in fields.items():
            if FIELDS.search(name):
                print(f"{name}: {value}")
            else:
                pass


if __name__ == "__main__":
    main()
