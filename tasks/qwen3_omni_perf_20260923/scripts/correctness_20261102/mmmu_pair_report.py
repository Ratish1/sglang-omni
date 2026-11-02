"""The MMMU pair answer by answer: correct counts per boot, items correct on one arm only, and a sign test over them.

usage: python3 mmmu_pair_report.py QUEUE_DIR   (QUEUE_DIR/{main_a,main_b,image_a,image_b}/c32/mmmu/mmmu_results.json)
"""

import json
import math
import sys
from pathlib import Path


def per_item(path: Path) -> dict[str, bool]:
    rows = json.loads(path.read_text())["per_sample"]
    return {row["sample_id"]: bool(row["is_correct"]) for row in rows}


def sign_test(wins: int, losses: int) -> float:
    total = wins + losses
    if total == 0:
        return 1.0
    else:
        tail = sum(math.comb(total, k) for k in range(0, min(wins, losses) + 1)) / 2**total
        return min(1.0, 2 * tail)


def main() -> None:
    root = Path(sys.argv[1])
    arms = {name: per_item(root / name / "c32" / "mmmu" / "mmmu_results.json") for name in ("main_a", "main_b", "image_a", "image_b")}
    for name, items in arms.items():
        print(f"{name}: {sum(items.values())} of {len(items)} correct")
    shared = sorted(set.intersection(*(set(items) for items in arms.values())))
    main_score = {item: arms["main_a"][item] + arms["main_b"][item] for item in shared}
    image_score = {item: arms["image_a"][item] + arms["image_b"][item] for item in shared}
    wins = sum(image_score[item] > main_score[item] for item in shared)
    losses = sum(image_score[item] < main_score[item] for item in shared)
    print(f"main mean {sum(main_score.values()) / 2:.1f}, image mean {sum(image_score.values()) / 2:.1f} of {len(shared)}")
    print(f"items better on image {wins}, better on main {losses}, sign test p {sign_test(wins, losses):.3f}")
    print(f"within main, A against B disagree on {sum(arms['main_a'][i] != arms['main_b'][i] for i in shared)} items")
    print(f"within image, A against B disagree on {sum(arms['image_a'][i] != arms['image_b'][i] for i in shared)} items")


if __name__ == "__main__":
    main()
