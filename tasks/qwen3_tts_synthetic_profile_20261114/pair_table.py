"""Table of a seedtts_pair.py run: per cell and order, arm a and arm b, then b against a averaged over
the two orders, which cancels the card each arm sat on.

usage: python3 pair_table.py OUT
"""

import argparse
import json
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
