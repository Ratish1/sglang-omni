"""Each stage's time per request in passes recorded with videoamme_pair.py --request-events, main against a branch.

Run on the box from main's tree (cwd and PYTHONPATH the tree):
    python3 stage_times_read.py --run /data/pr2800/<run> --arms upx,vpq --pass c16
Reads <run>/round<n>/<pass>/<arm>/events for every round, takes sglang_omni.profiler.views' stage and hop breakdowns
and each request's first-to-last event span, and prints every interval's mean per request for each arm (the mean of
the rounds' means, which cancels the card) with the branch's delta, and the audio encoder's mean batch size.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from sglang_omni.profiler.views import hop_breakdown, reconstruct_timelines, stage_breakdown


def arm_means(event_dir: Path) -> dict[str, float]:
    timelines = reconstruct_timelines(event_dir)
    means = {f"{row.stage} {row.interval_name}": row.avg_ms for row in stage_breakdown(timelines)}
    means.update({f"hop {row.src_stage}->{row.dst_stage} {row.kind}": row.avg_ms for row in hop_breakdown(timelines)})
    spans = []
    batch_sizes = []
    for timeline in timelines.values():
        stamps = [int(event["timestamp_ns"]) for event in timeline.events]
        spans.append((max(stamps) - min(stamps)) / 1e6)
        batch_sizes.extend(
            int(event["metadata"]["batch_size"])
            for event in timeline.events
            if event["event_name"] == "encoder_start" and (event.get("metadata") or {}).get("modality") == "audio"
        )
    means["request first to last event"] = statistics.mean(spans)
    means["audio encoder batch size (requests)"] = statistics.mean(batch_sizes) if batch_sizes else float("nan")
    return means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arms", required=True, help="main,branch")
    parser.add_argument("--pass", dest="pass_name", default="c16")
    args = parser.parse_args()
    main_arm, branch_arm = args.arms.split(",")
    rounds = [path for path in sorted(args.run.glob("round*")) if path.is_dir()]
    per_arm = {
        arm: [arm_means(round_dir / args.pass_name / arm / "events") for round_dir in rounds]
        for arm in (main_arm, branch_arm)
    }
    keys = sorted(set.intersection(*(set(means) for runs in per_arm.values() for means in runs)))
    rows = []
    for key in keys:
        main_value = statistics.mean(means[key] for means in per_arm[main_arm])
        branch_value = statistics.mean(means[key] for means in per_arm[branch_arm])
        rounds_text = " ".join(
            f"{main_round[key]:.3f}/{branch_round[key]:.3f}"
            for main_round, branch_round in zip(per_arm[main_arm], per_arm[branch_arm])
        )
        rows.append((main_value, key, branch_value, rounds_text))
    print(f"{args.pass_name}: {len(rounds)} rounds, mean per request, {main_arm} / {branch_arm} (per round)")
    for main_value, key, branch_value, rounds_text in sorted(rows, reverse=True):
        print(f"{key[:78]:78s} {main_value:9.3f} {branch_value:9.3f} {branch_value - main_value:+8.3f}  {rounds_text}")


if __name__ == "__main__":
    main()
