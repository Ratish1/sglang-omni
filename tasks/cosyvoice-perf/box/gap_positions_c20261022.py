"""Where a pair's inter chunk gap difference sits in a request: every request's gaps by position (the first gap, the
middle ones, the last, which ends at the final chunk), pooled per arm over both orders, from the per request records
(10-22, L3's c16 gaps). Order a puts the first arm on gpu0, order b on gpu1.
usage: python3 gap_positions_c20261022.py <pair dir> <cell> [<pair dir> <cell> ...]
"""

import json
import statistics
import sys
from pathlib import Path

ORDERS = (("a", ("gpu0", "gpu1")), ("b", ("gpu1", "gpu0")))


def positions(path: Path) -> dict[str, list[float]]:
    split = {"first": [], "middle": [], "last": [], "all": []}
    for record in json.loads(path.read_text())["per_request"]:
        gaps = (record["inter_chunk_s"] or []) if record["is_success"] else []
        if not gaps:
            continue
        else:
            pass
        split["all"].extend(gaps)
        split["last"].append(gaps[-1])
        if len(gaps) > 1:
            split["first"].append(gaps[0])
            split["middle"].extend(gaps[1:-1])
        else:
            pass
    return split


arguments = sys.argv[1:]
for pair_dir, cell in zip(arguments[0::2], arguments[1::2]):
    arms = [{"first": [], "middle": [], "last": [], "all": []} for _ in range(2)]
    for order, gpus in ORDERS:
        for arm, gpu in zip(arms, gpus):
            for key, values in positions(Path(pair_dir) / order / gpu / cell / "speed_results.json").items():
                arm[key].extend(values)
    cells = []
    for key in ("first", "middle", "last", "all"):
        base, other = arms[0][key], arms[1][key]
        if base and other:
            change = 100 * (statistics.mean(other) - statistics.mean(base)) / statistics.mean(base)
            cells.append(f"{key} n {len(base)}/{len(other)} {statistics.mean(base) * 1e3:.1f}/{statistics.mean(other) * 1e3:.1f} ms ({change:+.1f} %)")
        else:
            cells.append(f"{key} none")
    print(f"{Path(pair_dir).name} {cell}: " + "; ".join(cells))
