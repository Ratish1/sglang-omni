"""Codec EOS per frame against each request's outcome: joins `probe/qwen3-tts-eos-forensics`'s per-step lines
(request id, frame, EOS probability under temperature 0.9 and top-k 50, EOS rank, top-1 probability) to a layout.py
serve directory's records by speech id, then reads every runaway against every natural stop: the largest EOS
probability and its frame, the chance the trajectory never samples EOS (the product of 1 - p over its frames), and
EOS over its last frames. A runaway whose trajectory offered EOS and missed it is a sampling miss; one whose EOS
stayed near zero never had a way out.

usage: python3 eos_forensics.py SERVED_DIR PROBE_DIR [--max-new-tokens 300]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

LAST_FRAMES = 50
QUANTILES = (0.1, 0.5, 0.9, 0.99)


def describe(values: list[float]) -> str:
    if not values:
        return "-"
    else:
        pass
    points = np.quantile(np.array(values), QUANTILES)
    return " / ".join(f"{point:.3g}" for point in points)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("served", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()
    trajectories: dict[str, dict[int, tuple[float, int, float]]] = defaultdict(dict)
    for path in sorted(args.probe.glob("eos_*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            request_id, frame, probability, rank, top = line.split()
            trajectories[request_id][int(frame)] = (float(probability), int(rank), float(top))
    records = [
        json.loads(line)
        for line in (args.served / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    unmatched = 0
    for record in records:
        if "speech_id" not in record or "frames" not in record:
            continue
        else:
            pass
        trajectory = next(
            (frames for request_id, frames in trajectories.items() if record["speech_id"] in request_id), None
        ) if record["speech_id"] not in trajectories else trajectories[record["speech_id"]]
        if not trajectory:
            unmatched += 1
            continue
        else:
            pass
        runaway = record.get("finish_reason") == "length" or int(record["frames"]) >= args.max_new_tokens
        frames = sorted(trajectory)
        probabilities = np.array([trajectory[frame][0] for frame in frames])
        peak = int(np.argmax(probabilities))
        tail = probabilities[-LAST_FRAMES:] if runaway else probabilities[:-1][-LAST_FRAMES:]
        rows[("runaway" if runaway else "natural") + " " + str(record["set"])].append({
            "frames": float(record["frames"]),
            "peak": float(probabilities[peak]),
            "peak_frame": float(frames[peak]),
            "never": float(np.prod(1 - probabilities[:-1] if not runaway else 1 - probabilities)),
            "stop": float(probabilities[-1]) if not runaway else float("nan"),
            "tail_mean": float(tail.mean()) if tail.size else float("nan"),
            "tail_rank": float(np.median([trajectory[frame][1] for frame in frames[-LAST_FRAMES:]])),
            "tail_top": float(np.median([trajectory[frame][2] for frame in frames[-LAST_FRAMES:]])),
        })
    print(f"records {len(records)}, trajectories {len(trajectories)}, unmatched {unmatched}")
    print(
        "outcome set | n | frames q10/50/90/99 | peak EOS probability | peak frame | chance of never sampling EOS before "
        "the end | EOS probability at the stop frame | EOS over the last 50 frames | EOS rank, last 50 (median of "
        "medians) | top-1 probability, last 50"
    )
    for key in sorted(rows):
        values = rows[key]
        print(
            f"{key} | {len(values)} | {describe([v['frames'] for v in values])} | {describe([v['peak'] for v in values])} | "
            f"{describe([v['peak_frame'] for v in values])} | {describe([v['never'] for v in values])} | "
            f"{describe([v['stop'] for v in values if v['stop'] == v['stop']])} | "
            f"{describe([v['tail_mean'] for v in values if v['tail_mean'] == v['tail_mean']])} | "
            f"{np.median([v['tail_rank'] for v in values]):.0f} | {np.median([v['tail_top'] for v in values]):.3g}"
        )


if __name__ == "__main__":
    main()
