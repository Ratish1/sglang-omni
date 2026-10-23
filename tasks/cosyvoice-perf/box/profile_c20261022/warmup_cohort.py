"""Whether a cell's first cohort carries the rows a pair reads (10-23): the pair harness passes --warmup 1, where the
benchmark's default warms at the cell's concurrency (#1789) so the timed cohort is not the first to run at that
concurrency. Per order and arm, first audio and inter chunk mean, p95 and p99 over every request (the benchmark's
pooled method) and without the first <concurrency> requests (per_request is in dispatch order), then the second
arm's change against the first both ways.
usage: python3 warmup_cohort.py <pair dir> <cell>[,<cell>...]"""

import json
import sys
from pathlib import Path

import numpy as np


def rows(records):
    done = [record for record in records if record["is_success"]]
    ttfp = np.array([record["audio_ttfp_s"] for record in done if record["audio_ttfp_s"] is not None])
    gaps = np.concatenate([np.asarray(record["inter_chunk_s"] or [], dtype=float) for record in done])
    return {
        "first audio mean": ttfp.mean(),
        "first audio p95": np.percentile(ttfp, 95),
        "first audio p99": np.percentile(ttfp, 99),
        "gap mean": gaps.mean(),
        "gap p95": np.percentile(gaps, 95),
        "gap p99": np.percentile(gaps, 99),
    }


def main() -> None:
    pair, cells = Path(sys.argv[1]), sys.argv[2].split(",")
    for cell in cells:
        for order, gpus in (("a", (0, 1)), ("b", (1, 0))):
            results = [json.loads((pair / order / f"gpu{gpu}" / cell / "speed_results.json").read_text()) for gpu in gpus]
            cohort = int(results[0]["config"]["concurrency"])
            warmup = results[0]["config"]["warmup"]
            whole = [rows(result["per_request"]) for result in results]
            rest = [rows(result["per_request"][cohort:]) for result in results]
            first = [rows(result["per_request"][:cohort]) for result in results]
            print(f"{pair.name} {cell} order {order}: warmup {warmup}, first cohort {cohort} requests")
            for key in whole[0]:
                change_whole = 100 * (whole[1][key] - whole[0][key]) / whole[0][key]
                change_rest = 100 * (rest[1][key] - rest[0][key]) / rest[0][key]
                print(
                    f"   {key:17s} arm1 {whole[0][key]:.4f} arm2 {whole[1][key]:.4f} change {change_whole:+.1f} %"
                    f" | without cohort {change_rest:+.1f} % | cohort only arm1 {first[0][key]:.4f}"
                    f" arm2 {first[1][key]:.4f}"
                )


if __name__ == "__main__":
    main()
