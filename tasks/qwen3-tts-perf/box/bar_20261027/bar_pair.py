"""First byte of the bar's arms paired request by request: for each rate, every arm's measured requests (same seed, so
the same prompts in the same order) joined by phase index against the first arm. Prints per arm the first-byte (first
body less dispatch) quantiles, the paired difference's quantiles against the first arm, and the median difference by
prompt word-count tercile, so a shift that grows with the text separates from a constant one.

usage: python3 bar_pair.py BAR_DIR ARM [ARM ...] [--rates 1 20] [--seed 0]
"""

import argparse
import json
from pathlib import Path

import numpy as np


def first_bytes(path: Path) -> dict[int, tuple[float, int]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("phase") != "measurement" or not record.get("audio_success"):
            continue
        else:
            pass
        ttfb_ms = (record["first_body_elapsed_ns"] - record["dispatch_elapsed_ns"]) / 1e6
        rows[int(record["phase_index"])] = (ttfb_ms, int(record["prompt_word_count"]))
    return rows


def quantiles(values: np.ndarray) -> str:
    return " / ".join(f"{np.quantile(values, q):.2f}" for q in (0.1, 0.5, 0.9, 0.95))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bar", type=Path)
    parser.add_argument("arms", nargs="+")
    parser.add_argument("--rates", nargs="+", default=["1", "20"])
    parser.add_argument("--seed", default="0")
    args = parser.parse_args()
    for rate in args.rates:
        arms = {arm: first_bytes(args.bar / arm / f"seed{args.seed}_rps{rate}" / "requests.jsonl") for arm in args.arms}
        base_name = args.arms[0]
        base = arms[base_name]
        print(f"rate {rate} req/s: first byte ms q10 / q50 / q90 / q95; paired against {base_name}")
        for arm, rows in arms.items():
            shared = sorted(set(rows) & set(base))
            mine = np.array([rows[index][0] for index in rows])
            diff = np.array([rows[index][0] - base[index][0] for index in shared])
            words = np.array([base[index][1] for index in shared])
            edges = np.quantile(words, [1 / 3, 2 / 3])
            terciles = [diff[words <= edges[0]], diff[(words > edges[0]) & (words <= edges[1])], diff[words > edges[1]]]
            by_words = " / ".join(f"{np.median(part):+.2f}" if part.size else "n/a" for part in terciles)
            print(
                f"  {arm}: n {mine.size}, {quantiles(mine)}; paired n {diff.size}, difference {quantiles(diff)}, "
                f"median by word tercile {by_words}"
            )


if __name__ == "__main__":
    main()
