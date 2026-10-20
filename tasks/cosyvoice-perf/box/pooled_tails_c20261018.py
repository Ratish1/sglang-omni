"""Pair rows pooled over both card orders, beside the same rows of A/A pairs: per arm, every request of order a and
order b together (first audio, latency, and every chunk gap), its mean, p95 and p99, and the arm's change. A pair's
base arm is the tree whose head starts with the given prefix; an A/A pair (both arms one head) takes the tree in slot
0 of order a and slot 1 of order b as its base, the way a pair's arm swaps cards.
usage: python3 pooled_tails_c20261018.py <cell> <label>=<pair dir>[:<base head prefix>] ..."""

import json
import statistics
import sys
from pathlib import Path


def percentile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def arms(pair_dir: Path, base_head: str | None, cell: str):
    pooled = {"base": [], "arm": []}
    for order, base_slot in (("a", "gpu0"), ("b", "gpu1")):
        for gpu in ("gpu0", "gpu1"):
            head = (pair_dir / order / f"identity_{gpu}.txt").read_text().split()[0]
            is_base = head.startswith(base_head) if base_head else gpu == base_slot
            records = json.loads((pair_dir / order / gpu / cell / "speed_results.json").read_text())["per_request"]
            pooled["base" if is_base else "arm"] += [r for r in records if r["is_success"]]
    return pooled


def rows(records):
    first = [r["audio_ttfp_s"] for r in records if r["audio_ttfp_s"] is not None]
    latency = [r["latency_s"] for r in records]
    gaps = [g for r in records for g in (r["inter_chunk_s"] or [])]
    out = {}
    for name, values in (("first audio", first), ("latency", latency), ("inter chunk", gaps)):
        out[f"{name} mean"] = statistics.mean(values)
        out[f"{name} p95"] = percentile(values, 0.95)
        out[f"{name} p99"] = percentile(values, 0.99)
    return out


cell = sys.argv[1]
table = {}
for spec in sys.argv[2:]:
    label, _, rest = spec.partition("=")
    directory, _, base_head = rest.partition(":")
    pooled = arms(Path(directory), base_head or None, cell)
    base, arm = rows(pooled["base"]), rows(pooled["arm"])
    table[label] = {key: 100 * (arm[key] / base[key] - 1) for key in base}
keys = list(next(iter(table.values())))
print(f"{cell}, pooled over both orders, arm against base (%)")
print(f"{'metric':18}" + "".join(f"{label:>10}" for label in table))
for key in keys:
    print(f"{key:18}" + "".join(f"{table[label][key]:+10.1f}" for label in table))
