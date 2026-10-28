"""Rows of run_bench_boot.sh pairs: every boot's values per arm, then each tree's mean over the
pair's two card orders and the delta of the second tree against the first.

A round dir holds card0_<tree> and card1_<tree>; the rounds of one pair are given together, the
baseline tree first. When both boots of a round run one tree (an A/A), each is named
<tree>@card<N>. A boot with a FAILED marker is printed and left out of the means.

usage: python pair_rows.py BASE_TREE ROUND_DIR [ROUND_DIR ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

VIDEO_FIELDS = (
    "latency_mean_s",
    "latency_median_s",
    "latency_p95_s",
    "latency_p99_s",
    "throughput_qps",
)
SPEECH_FIELDS = VIDEO_FIELDS + (
    "rtf_mean",
    "audio_throughput_s_per_s",
    "audio_ttfp_mean_s",
    "audio_ttfp_p99_s",
    "inter_chunk_mean_s",
    "inter_chunk_p99_s",
)


def arm_values(boot: Path, arm: str) -> dict[str, float]:
    speech = boot / arm / "speed_results.json"
    if speech.exists():
        summary = json.loads(speech.read_text())["summary"]
        values = {field: summary[field] for field in SPEECH_FIELDS if field in summary}
        values["completed"] = summary["completed_requests"]
        return values
    else:
        pass
    results = json.loads((boot / arm / f"{arm}_results.json").read_text())
    values = {field: results["speed"][field] for field in VIDEO_FIELDS}
    values["accuracy"] = results["summary"]["accuracy"]
    values["completed"] = results["speed"]["completed_requests"]
    values["max_token_hits"] = results["speed"]["max_token_hits"]
    return values


def main() -> None:
    base_tree, rounds = sys.argv[1], [Path(arg) for arg in sys.argv[2:]]
    pooled: dict[tuple[str, str], dict[str, list[float]]] = {}
    for round_dir in rounds:
        boots = sorted(round_dir.glob("card*_*"))
        same_tree = len({boot.name.split("_", 1)[1] for boot in boots}) == 1
        for boot in boots:
            card, tree = boot.name.split("_", 1)
            if same_tree:
                tree = f"{tree}@{card}"
            else:
                pass
            failed = (boot / "FAILED").exists()
            for arm_dir in sorted(path for path in boot.iterdir() if path.is_dir()):
                if arm_dir.name in ("events", "audio"):
                    continue
                else:
                    pass
                values = arm_values(boot, arm_dir.name)
                shown = " ".join(f"{key}={value}" for key, value in values.items())
                print(
                    f"{round_dir.name} {boot.name} {arm_dir.name}"
                    f"{' FAILED' if failed else ''}: {shown}"
                )
                if failed:
                    continue
                else:
                    pass
                key = (arm_dir.name, tree)
                for field, value in values.items():
                    if value is not None:
                        pooled.setdefault(key, {}).setdefault(field, []).append(value)
                    else:
                        pass
    print()
    arms = sorted({arm for arm, _ in pooled})
    for arm in arms:
        trees = [tree for candidate, tree in pooled if candidate == arm]
        others = [tree for tree in trees if tree != base_tree]
        for other in others or [base_tree]:
            print(f"{arm}: {base_tree} / {other} (each the mean of its boots), delta")
            base, compared = pooled[(arm, base_tree)], pooled[(arm, other)]
            for field in base:
                if field not in compared:
                    continue
                else:
                    pass
                base_mean = statistics.mean(base[field])
                other_mean = statistics.mean(compared[field])
                delta = (other_mean / base_mean - 1) * 100 if base_mean else 0.0
                print(
                    f"  {field:26s} {base[field]} / {compared[field]}"
                    f"  mean {base_mean:.4g} / {other_mean:.4g}  {delta:+.1f} %"
                )


if __name__ == "__main__":
    main()
