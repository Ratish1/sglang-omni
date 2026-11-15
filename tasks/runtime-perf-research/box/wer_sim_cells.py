"""SeedTTS EN WER and SIM per scored cell, from each cell's wer_results.json and similarity_results.json.

usage: python3 wer_sim_cells.py <label>=<bench dir> [<label>=<bench dir> ...]
"""

import json
import sys
from pathlib import Path


def summary(path: Path) -> dict[str, float] | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["summary"]
    else:
        return None


def main() -> None:
    print(
        "| cell | WER evaluated | corpus WER % | below-50 WER % | above 50 % | SIM evaluated | SIM mean |"
    )
    print("|---|---|---|---|---|---|---|")
    for spec in sys.argv[1:]:
        label, bench = spec.split("=")
        wer = summary(Path(bench) / "wer_results.json")
        similarity = summary(Path(bench) / "similarity_results.json")
        if wer is None:
            wer_cells = "- | - | - | -"
        else:
            wer_cells = (
                f"{wer['evaluated']} | {wer['wer_corpus'] * 100:.3f} | "
                f"{wer['wer_below_50_corpus'] * 100:.3f} | {wer['n_above_50_pct_wer']}"
            )
        if similarity is None:
            similarity_cells = "- | -"
        else:
            similarity_cells = f"{similarity['evaluated']} | {similarity['speaker_similarity_mean']:.4f}"
        print(f"| {label} | {wer_cells} | {similarity_cells} |")


if __name__ == "__main__":
    main()
