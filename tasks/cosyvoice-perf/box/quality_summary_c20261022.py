"""SeedTTS quality of scored pairs, one row per pair, cell and arm with both orders: corpus WER, corpus WER over the
clips at or below 50 %, the clips above 50 %, the requests at the AR's token cap, and SIM, each arm named by its
tree's commit; then each pair's second arm minus its first per order.
usage: python3 quality_summary_c20261022.py <run home> <pair:cells> [<pair:cells> ...]"""

import json
import sys
from pathlib import Path


def load(path):
    return json.loads(path.read_text())["summary"] if path.exists() else None


def main():
    home = Path(sys.argv[1])
    print("| pair | cell | arm | order a: WER, below 50 %, above 50 %, at cap, SIM | order b: same |")
    print("| --- | --- | --- | --- | --- |")
    deltas = []
    for spec in sys.argv[2:]:
        name, cells = spec.split(":")
        pair = home / name
        commits = {
            (order, gpu): (pair / order / f"identity_gpu{gpu}.txt").read_text().split()[0][:9]
            for order in ("a", "b")
            for gpu in (0, 1)
        }
        for cell in cells.split(","):
            rows = {}
            for order in ("a", "b"):
                for gpu in (0, 1):
                    directory = pair / order / f"gpu{gpu}" / cell
                    wer, sim, speed = (
                        load(directory / "wer_results.json"),
                        load(directory / "similarity_results.json"),
                        load(directory / "speed_results.json"),
                    )
                    arm = "first" if (order == "a") == (gpu == 0) else "second"
                    rows[(arm, order)] = (
                        commits[(order, gpu)],
                        wer and 100 * wer["wer_corpus"],
                        wer and 100 * wer["wer_below_50_corpus"],
                        wer and wer["n_above_50_pct_wer"],
                        speed and speed.get("max_token_hits"),
                        sim and sim["speaker_similarity_mean"],
                    )
            for arm in ("first", "second"):
                cells_text = []
                for order in ("a", "b"):
                    _, corpus, below, above, cap, sim = rows[(arm, order)]
                    cells_text.append(
                        "pending" if corpus is None or sim is None else
                        f"{corpus:.3f} %, {below:.3f} %, {above}, {cap}, {sim:.2f}"
                    )
                print(f"| {name} | {cell} | {arm} {rows[(arm, 'a')][0]} | {cells_text[0]} | {cells_text[1]} |")
            for order in ("a", "b"):
                first, second = rows[("first", order)], rows[("second", order)]
                if None not in (first[1], second[1], first[5], second[5]):
                    deltas.append(
                        f"| {name} | {cell} | {order} | {second[1] - first[1]:+.3f} | {second[2] - first[2]:+.3f} | "
                        f"{second[5] - first[5]:+.2f} |"
                    )
    print()
    print("Second arm minus first, per order (WER in points, SIM in its own units):")
    print("| pair | cell | order | corpus WER | WER at or below 50 % | SIM |")
    print("| --- | --- | --- | --- | --- | --- |")
    for line in deltas:
        print(line)


if __name__ == "__main__":
    main()
