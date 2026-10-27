"""One row per serve_arm.py arm: RTF mean, request throughput, requests at the token cap, corpus WER, samples over 50 %
WER, SIM mean and the samples below SIM 50 (a lost speaker), with their ids.

usage: python3 ab_table.py ARMS_DIR [ARM ...]
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arms", type=Path)
    parser.add_argument("labels", nargs="*")
    args = parser.parse_args()
    labels = args.labels or sorted(path.name for path in args.arms.iterdir() if path.is_dir())
    print("arm | RTF | req/s | cap hits | WER | over 50 % | SIM | below 50")
    for label in labels:
        bench = args.arms / label / "bench"
        if not (bench / "similarity_results.json").exists():
            print(f"{label} | not scored")
            continue
        else:
            pass
        speed = json.loads((bench / "speed_results.json").read_text())["summary"]
        wer = json.loads((bench / "wer_results.json").read_text())["summary"]
        similarity = json.loads((bench / "similarity_results.json").read_text())
        lost = sorted(
            (round(sample["speaker_similarity"], 1), sample["id"])
            for sample in similarity["per_sample"]
            if sample.get("speaker_similarity") is not None and sample["speaker_similarity"] < 50
        )
        print(
            f"{label} | {speed['rtf_mean']:.4f} | {speed['throughput_qps']:.2f} | {speed['max_token_hits']} | "
            f"{100 * wer['wer_corpus']:.2f} % | {wer['n_above_50_pct_wer']} | "
            f"{similarity['summary']['speaker_similarity_mean']:.2f} | {len(lost)} {lost[:6]}"
        )


if __name__ == "__main__":
    main()
