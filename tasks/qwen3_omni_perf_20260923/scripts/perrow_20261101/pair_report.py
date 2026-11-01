"""The per-row pair's rows (queue_pair.sh): per boot and load the client figures, WER and SIM, then per load each arm's
mean over its boots and the branch's change against main's mean.

usage: python3 pair_report.py PAIR_DIR
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ARMS = {"main": ("main_a", "main_b"), "perrow": ("perrow_a", "perrow_b")}
FIELDS = (
    ("throughput_qps", "QPS"),
    ("latency_mean_s", "latency mean"),
    ("latency_p99_s", "latency p99"),
    ("audio_ttfp_mean_s", "first audio mean"),
    ("audio_ttfp_p99_s", "first audio p99"),
    ("inter_chunk_mean_s", "inter chunk mean"),
    ("inter_chunk_p99_s", "inter chunk p99"),
    ("rtf_mean", "RTF mean"),
)


def read(pass_dir: Path) -> dict[str, float]:
    arm_dir = pass_dir / "seedtts_en"
    speed = json.loads((arm_dir / "speed_results.json").read_text())["summary"]
    row = {key: float(speed[key]) for key, _ in FIELDS}
    row["completed"] = float(speed["completed_requests"])
    score = arm_dir / "score_summary.json"
    sim = arm_dir / "sim_summary.json"
    if score.exists():
        row["wer"] = 100 * float(json.loads(score.read_text())["wer"]["wer_corpus"])
    else:
        pass
    if sim.exists():
        row["sim"] = float(
            json.loads(sim.read_text())["similarity"]["speaker_similarity_mean"]
        )
    else:
        pass
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair_dir")
    args = parser.parse_args()
    root = Path(args.pair_dir)
    keys = [key for key, _ in FIELDS] + ["wer", "sim"]
    names = dict(FIELDS) | {"wer": "WER %", "sim": "SIM"}
    for conc in (16, 64):
        rows = {}
        for labels in ARMS.values():
            for label in labels:
                pass_dir = root / label / f"c{conc}"
                if (pass_dir / "seedtts_en" / "speed_results.json").exists():
                    rows[label] = read(pass_dir)
                else:
                    pass
        print(
            f"\nc{conc}: "
            + ", ".join(
                f"{label} {int(row['completed'])} completed"
                for label, row in rows.items()
            )
        )
        print(
            f"{'':18}"
            + "".join(f"{label:>11}" for label in rows)
            + f"{'main':>11}{'perrow':>11}{'change':>9}"
        )
        for key in keys:
            values = {label: row.get(key) for label, row in rows.items()}
            means = {}
            for arm, labels in ARMS.items():
                present = [
                    values[label] for label in labels if values.get(label) is not None
                ]
                means[arm] = statistics.mean(present) if present else None
            change = (
                f"{means['perrow'] / means['main'] - 1:+9.1%}"
                if means["main"] and means["perrow"] is not None
                else f"{'':>9}"
            )
            cells = "".join(
                f"{value:11.4f}" if value is not None else f"{'':>11}"
                for value in values.values()
            )
            mean_cells = "".join(
                f"{value:11.4f}" if value is not None else f"{'':>11}"
                for value in means.values()
            )
            print(f"{names[key]:18}{cells}{mean_cells}{change}")


if __name__ == "__main__":
    main()
