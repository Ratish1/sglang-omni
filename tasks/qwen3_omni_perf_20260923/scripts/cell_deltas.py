"""Per round deltas: in a pair round the branch boot against the main boot, in an A/A round card 1 against card 0.

usage: python cell_deltas.py BASE_TREE ROUND [ROUND ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, "/data/checks/pair_rows_fix")
from pair_rows import arm_values

base_tree = sys.argv[1]
fields = (
    "latency_mean_s",
    "latency_median_s",
    "latency_p95_s",
    "latency_p99_s",
    "throughput_qps",
    "rtf_mean",
    "audio_ttfp_mean_s",
    "inter_chunk_mean_s",
    "inter_chunk_p99_s",
    "accuracy",
)
for round_name in sys.argv[2:]:
    boots = sorted(Path("/data/ab", round_name).glob("card*_*"))
    if len(boots) != 2 or any((boot / "FAILED").exists() for boot in boots):
        print(f"{round_name}: unusable boots {[boot.name for boot in boots]}")
        continue
    trees = [boot.name.split("_", 1)[1] for boot in boots]
    if trees[0] == trees[1]:
        base, other, label = boots[0], boots[1], "card1 vs card0 (A/A)"
    elif trees[0] == base_tree:
        base, other, label = boots[0], boots[1], "branch on card1"
    else:
        base, other, label = boots[1], boots[0], "branch on card0"
    arms = sorted(
        path.name
        for path in base.iterdir()
        if path.is_dir() and path.name not in ("events",)
    )
    for arm in arms:
        base_values, other_values = arm_values(base, arm), arm_values(other, arm)
        parts = []
        for field in fields:
            if field in base_values and field in other_values and base_values[field]:
                if field == "accuracy":
                    parts.append(
                        f"acc {base_values[field]:.2f}/{other_values[field]:.2f}"
                    )
                else:
                    parts.append(
                        f"{field.replace('latency_', '').replace('_s', '')} {100 * (other_values[field] / base_values[field] - 1):+.1f}%"
                    )
        print(
            f"{round_name} {arm} [{label}] n={base_values.get('completed')}/{other_values.get('completed')}: "
            + ", ".join(parts)
        )
