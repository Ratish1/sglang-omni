"""The standard SeedTTS EN A/B of two trees on warm server pairs: each card order boots both servers once and runs
non-streaming then streaming c16, unseeded; then each card's runs are scored under one ASR server, then speaker
similarity on that card; then the table: RTF, throughput, first chunk, WER and SIM per run, and SIM paired per request
with its tails.

usage: python3 ab_runs.py --a LABEL=TREE --b LABEL=TREE --root /data/next_20261031/<block> [--port 11000] [--table-only]
"""

import argparse
import concurrent.futures
import json
import re
import statistics
from pathlib import Path

from seeded_tail import per_sample, score_card, serve_phase

RUNS = (("n16", 16, None, False), ("s16", 16, None, True))
METRICS = ("RTF mean", "Throughput (req/s)", "TTFC mean (s)", "TTFC p95 (s)", "Completed requests", "Failed requests",
           "WER (corpus, micro-avg)")


def metrics(out: Path) -> dict[str, str]:
    text = "\n".join(
        path.read_text(errors="replace")
        for path in (out / "bench.log", out / "score--transcribe-only.log", out / "score--similarity-only.log")
        if path.exists()
    )
    found = {}
    for name in METRICS:
        match = re.search(re.escape(name) + r":\s+([0-9.]+)", text)
        found[name] = match.group(1) if match else "-"
    similarity = re.search(r"SeedTTS speaker similarity: ([0-9.]+)", text)
    found["SIM"] = similarity.group(1) if similarity else "-"
    return found


def table(root: Path, first: str, second: str) -> None:
    for run in sorted(path.name for path in root.glob("*_o[12]")):
        row = {label: metrics(root / run / label) for label in (first, second)}
        print(f"{run}: " + " | ".join(f"{label} {json.dumps(values)}" for label, values in row.items()), flush=True)
        a = per_sample(root / run / first, "similarity_results.json", "speaker_similarity")
        b = per_sample(root / run / second, "similarity_results.json", "speaker_similarity")
        common = sorted(set(a) & set(b))
        deltas = [b[k] - a[k] for k in common]
        half_width = 1.96 * statistics.pstdev(deltas) / len(deltas) ** 0.5
        print(f"  {run} paired SIM ({second} minus {first}) {statistics.mean(deltas):+.3f} +- {half_width:.3f}; "
              f"below 40 {sum(a[k] < 40 for k in common)} / {sum(b[k] < 40 for k in common)}; more than 15 under "
              f"the other arm {sum(d > 15 for d in deltas)} / {sum(d < -15 for d in deltas)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args()
    (first, first_tree), (second, second_tree) = (arm.split("=", 1) for arm in (args.a, args.b))
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    if not args.table_only:
        by_card: dict[int, list[Path]] = {0: [], 1: []}
        orders = (("o1", ((first, first_tree, 0), (second, second_tree, 1))),
                  ("o2", ((second, second_tree, 0), (first, first_tree, 1))))
        for index, (order, arms) in enumerate(orders):
            ports = (args.port + 1000 * index, args.port + 1000 * index + 200)
            serve_phase(order, [(label, tree, card, port) for (label, tree, card), port in zip(arms, ports)],
                        [(f"{name}_{order}", concurrency, seed, stream) for name, concurrency, seed, stream in RUNS],
                        root)
            for label, _, card in arms:
                by_card[card] += [root / f"{name}_{order}" / label for name, _, _, _ in RUNS]
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda card: score_card(card, args.port + 2400 + 200 * card, first_tree, by_card[card], root),
                          (0, 1)))
    else:
        pass
    table(root, first, second)


if __name__ == "__main__":
    main()
