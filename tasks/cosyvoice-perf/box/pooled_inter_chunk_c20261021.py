"""Every inter chunk gap of a pair's cell, per arm and order and pooled over both orders, from the per request records:
mean, p95, p99, p99.9, and the requests whose playback underran. Order a puts the first arm on gpu0, order b on gpu1.
usage: python3 pooled_inter_chunk_c20261021.py <pair dir> <cell> <first arm name> <second arm name>"""

import json
import sys
from pathlib import Path

ORDERS = (("a", ("gpu0", "gpu1")), ("b", ("gpu1", "gpu0")))


def records(path):
    with open(path) as handle:
        return [record for record in json.load(handle)["per_request"] if record["is_success"]]


def quantile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def describe(gaps):
    return (f"gaps {len(gaps)} mean {sum(gaps) / len(gaps):.4f} p95 {quantile(gaps, 0.95):.4f} "
            f"p99 {quantile(gaps, 0.99):.4f} p99.9 {quantile(gaps, 0.999):.4f}")


pair_dir, cell, names = Path(sys.argv[1]), sys.argv[2], sys.argv[3:5]
pooled = {name: [] for name in names}
for order, gpus in ORDERS:
    for name, gpu in zip(names, gpus):
        arm = records(pair_dir / order / gpu / cell / "speed_results.json")
        gaps = [gap for record in arm for gap in (record["inter_chunk_s"] or [])]
        underruns = [record["max_playback_underrun_s"] or 0.0 for record in arm]
        pooled[name].extend(gaps)
        print(f"{order} {name} {gpu}: {describe(gaps)} requests underrun {sum(value > 0 for value in underruns)} "
              f"total underrun {sum(underruns):.2f} s")
for name, gaps in pooled.items():
    print(f"pooled {name}: {describe(gaps)}")
