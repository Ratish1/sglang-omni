"""Scored passes of one server as one run cells_paired.py can read: each pass's samples keep a __rK suffix.

usage: python3 merge_passes.py OUT_RUN PASS_RUN [PASS_RUN ...]   (each pass holds seedtts_en/wer_results.json, sim/seedtts_en/sim_summary.json)
"""

import json
import sys
from pathlib import Path


def main() -> None:
    out = Path(sys.argv[1])
    rows = []
    similarities = []
    for index, name in enumerate(sys.argv[2:]):
        run = Path(name)
        for row in json.loads((run / "seedtts_en" / "wer_results.json").read_text())["per_sample"]:
            rows.append({**row, "id": f"{row['id']}__r{index}"})
        similarities.append(json.loads((run / "sim" / "seedtts_en" / "sim_summary.json").read_text())["similarity"])
    (out / "seedtts_en").mkdir(parents=True, exist_ok=True)
    (out / "sim" / "seedtts_en").mkdir(parents=True, exist_ok=True)
    (out / "seedtts_en" / "wer_results.json").write_text(json.dumps({"per_sample": rows}))
    evaluated = sum(similarity["evaluated"] for similarity in similarities)
    mean = sum(similarity["speaker_similarity_mean"] * similarity["evaluated"] for similarity in similarities) / evaluated
    (out / "sim" / "seedtts_en" / "sim_summary.json").write_text(
        json.dumps({"similarity": {"speaker_similarity_mean": mean, "evaluated": evaluated}})
    )
    print(f"merged {len(rows)} rows from {len(similarities)} passes, similarity {mean:.3f}")


if __name__ == "__main__":
    main()
