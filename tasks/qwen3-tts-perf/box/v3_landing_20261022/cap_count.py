"""Frame-cap hits per prompt in pair.py runs over a repeated-prompt meta.lst: per arm, the requests that ran to the
2,048-frame cap (163.84 s of audio), by prompt.

usage: python3 cap_count.py LABEL [LABEL ...]   (labels under PAIR_ROOT)
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/v3_landing_20261022/pairs"))
CAP_SECONDS = 163.8


def main() -> None:
    totals: dict[str, Counter[str]] = {}
    for label in sys.argv[1:]:
        for arm in ("a", "b"):
            tree = (ROOT / label / arm / "tree.txt").read_text().split()[0]
            results = json.load(open(ROOT / label / arm / "bench" / "speed_results.json"))
            sent: Counter[str] = Counter()
            capped: Counter[str] = Counter()
            for request in results["per_request"]:
                prompt = request["id"].rsplit("_r", 1)[0]
                sent[prompt] += 1
                if float(request.get("audio_duration_s") or 0) >= CAP_SECONDS:
                    capped[prompt] += 1
                else:
                    pass
            print(f"{label} {arm} {tree}: {sum(capped.values())} capped of {sum(sent.values())}")
            for prompt in sent:
                print(f"  {prompt}: {capped[prompt]} of {sent[prompt]}")
            totals.setdefault(tree, Counter()).update({f"{prompt} capped": capped[prompt] for prompt in sent})
            totals[tree].update({f"{prompt} sent": sent[prompt] for prompt in sent})
    for tree, counts in totals.items():
        print(f"total {tree}:")
        for prompt in sorted({key.rsplit(' ', 1)[0] for key in counts}):
            print(f"  {prompt}: {counts[prompt + ' capped']} of {counts[prompt + ' sent']}")


if __name__ == "__main__":
    main()
