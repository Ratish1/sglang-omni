"""Table of a seedtts_pair.py run: per cell and order, arm a and arm b, then b against a averaged over
the two orders, which cancels the card each arm sat on.

usage: python3 pair_table.py OUT
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

METRICS = (
    ("audio_ttfp_mean_s", 1000.0, "first audio mean ms"),
    ("audio_ttfp_median_s", 1000.0, "first audio median ms"),
    ("audio_ttfp_p95_s", 1000.0, "first audio p95 ms"),
    ("latency_mean_s", 1000.0, "latency mean ms"),
    ("rtf_mean", 1.0, "rtf mean"),
    ("throughput_qps", 1.0, "req/s"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    cells = sorted({path.stem for path in args.out.glob("order*/a/*.json")})
    for cell in cells:
        print(f"== {cell}")
        for key, scale, label in METRICS:
            values = {}
            for order in (0, 1):
                for arm in ("a", "b"):
                    path = args.out / f"order{order}" / arm / f"{cell}.json"
                    if path.exists():
                        value = json.loads(path.read_text()).get(key)
                        values[(order, arm)] = None if value is None else value * scale
                    else:
                        pass
            if not values or any(value is None for value in values.values()):
                continue
            else:
                pass
            row = "  ".join(
                f"o{order} a {values[(order, 'a')]:.4g} b {values[(order, 'b')]:.4g}"
                for order in (0, 1)
                if (order, "a") in values and (order, "b") in values
            )
            pairs = [
                (values[(order, "a")], values[(order, "b")])
                for order in (0, 1)
                if (order, "a") in values and (order, "b") in values
            ]
            delta = sum(b - a for a, b in pairs) / len(pairs)
            relative = sum((b - a) / a for a, b in pairs) / len(pairs)
            print(f"  {label:22s} {row}  | b-a {delta:+.4g} ({relative:+.1%})")
    print("pooled over orders and rounds: a mean, b mean, b against a")
    pooled: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in args.out.glob("order*/*/*.json"):
        cell = re.sub(r"_r\d+$", "", path.stem)
        summary = json.loads(path.read_text())
        for key, scale, label in METRICS:
            if summary.get(key) is not None:
                pooled[(cell, label, path.parent.name)].append(summary[key] * scale)
            else:
                pass
    for cell in sorted({cell for cell, _, _ in pooled}):
        for _, _, label in METRICS:
            a_values = pooled.get((cell, label, "a"))
            b_values = pooled.get((cell, label, "b"))
            if a_values and b_values:
                a_mean = sum(a_values) / len(a_values)
                b_mean = sum(b_values) / len(b_values)
                print(f"  {cell:12s} {label:22s} a {a_mean:.4g} b {b_mean:.4g} ({(b_mean - a_mean) / a_mean:+.1%}, n={len(a_values)})")
            else:
                pass
    for bench_dir in sorted(args.out.glob("order*/*/*/bench")):
        wer_path = bench_dir / "wer_results.json"
        similarity_path = bench_dir / "similarity_results.json"
        if wer_path.exists() and similarity_path.exists():
            wer = json.loads(wer_path.read_text())["summary"]
            similarity = json.loads(similarity_path.read_text())
            similarity = similarity.get("summary", similarity)
            similarity_mean = next(
                value
                for name, value in similarity.items()
                if "mean" in name.lower() and isinstance(value, (int, float))
            )
            print(
                f"{bench_dir.parent.relative_to(args.out)} wer_corpus {wer['wer_corpus']:.4f} "
                f"wer_mean {wer['wer_per_sample_mean']:.4f} above_50 {wer['n_above_50_pct_wer']} "
                f"sim {similarity_mean:.2f}"
            )
        else:
            pass


if __name__ == "__main__":
    main()
