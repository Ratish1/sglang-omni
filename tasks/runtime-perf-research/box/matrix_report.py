# SPDX-License-Identifier: Apache-2.0
"""One table of the modality matrix: every cell's quality and speed for each boot.

usage: python3 matrix_report.py /data/runtime-perf/l28 q_a1 q_p1 q_s1 q_a2

Per cell (modality_matrix.py): accuracy and failed requests (text cells; talker cells' accuracy is the text reply's),
text-audio WER of the talker cells (corpus, below 50 %, requests above 50 %), SeedTTS WER against its text and
speaker similarity, and QPS. One column per boot, in the order given.
"""

import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def first(summary: dict, *names: str) -> object:
    for name in names:
        if name in summary:
            return summary[name]
        else:
            pass
    return None


def cell_rows(matrix: Path, cell: str) -> dict[str, object]:
    rows: dict[str, object] = {}
    if cell.startswith("seedtts"):
        wer = load(matrix / cell / "wer_results.json").get("summary", {})
        speed = load(matrix / cell / "speed_results.json").get("summary", {})
        similarity = load(matrix / cell / "similarity_results.json").get("summary", {})
        rows["WER corpus %"] = wer.get("wer_corpus")
        rows["WER below 50 %"] = wer.get("wer_below_50_corpus")
        rows["above 50 %"] = wer.get("n_above_50_pct_wer")
        rows["SIM"] = similarity.get("speaker_similarity_mean")
        rows["failed"] = speed.get("failed_requests")
        rows["QPS"] = speed.get("throughput_qps")
        return rows
    else:
        pass
    result = load(matrix / cell / "result.json")
    summary = result.get("summary") or result.get("accuracy") or {}
    speed = result.get("speed", {})
    rows["accuracy"] = first(summary, "accuracy", "overall_accuracy", "micro_accuracy")
    rows["failed"] = first(summary, "failed", "failed_samples")
    rows["total"] = first(summary, "total_samples", "total")
    rows["QPS"] = speed.get("throughput_qps")
    if cell.endswith("_talker"):
        wer = load(matrix / cell / "wer.json").get("summary", {})
        rows["text-audio WER %"] = wer.get("wer_corpus")
        rows["WER below 50 %"] = wer.get("wer_below_50_corpus")
        rows["above 50 %"] = wer.get("n_above_50_pct_wer")
    else:
        pass
    return rows


def show(value: object, name: str) -> str:
    if value is None:
        return "-"
    elif isinstance(value, float) and "WER" in name:
        return f"{value * 100:.3f}"
    elif isinstance(value, float):
        return f"{value:.4f}"
    else:
        return str(value)


def main() -> None:
    root = Path(sys.argv[1])
    boots = sys.argv[2:]
    cells = [
        "seedtts_voice_clone",
        "seedtts_text",
        "mmmu",
        "mmsu",
        "videomme",
        "videoamme",
        "mmmu_talker",
        "mmsu_talker",
        "videomme_talker",
        "videoamme_talker",
    ]
    print("| cell | row | " + " | ".join(boots) + " |")
    print("|---|---|" + "---|" * len(boots))
    for cell in cells:
        per_boot = [cell_rows(root / boot / "matrix", cell) for boot in boots]
        names = list(dict.fromkeys(name for rows in per_boot for name in rows))
        for name in names:
            print(f"| {cell} | {name} | " + " | ".join(show(rows.get(name), name) for rows in per_boot) + " |")


if __name__ == "__main__":
    main()
