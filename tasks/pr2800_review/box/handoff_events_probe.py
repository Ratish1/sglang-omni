"""Request events around the talker's first chunk, main against #2861, to place #2861's seeded c1 first-audio delta.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/handoff_events_probe.py --arms upk,phu
    --out /data/pr2800/<run>
Two rounds, the arms on cards 0 and 1 at the same time, then swapped. Each arm serves the cookbook's colocated bf16
config (pair_check.serve) and records request events (POST /start_request_profile) while SeedTTS EN streamed voice
clone runs seeded at c1 on the first SAMPLES prompts (pair_check's request shape, warmup 1), then stops them.
The reader rebuilds each request's timeline (sglang_omni.profiler.views), takes every (stage, event) first time as an
offset from the request's first event, and the thinker to talker chunk 0 (the prompt rows) and later chunk hops.
Arms pair by request order, so a request's delta compares the same prompt; each arm's value is the mean over both
rounds, which cancels the card. Writes <out>/r<n>_card<c>_<arm>/{events,c1_seeded}, <out>/handoff.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import pair_check  # noqa: E402

SAMPLES = 200
SERVER_PORT = 29000
BENCH_ARGS = ["--max-concurrency", "1", "--seed", "1234", "--max-samples", str(SAMPLES)]


def run_arm(out: Path, round_index: int, card: int, arm: str) -> None:
    from benchmarks.eval.asr_profiling import (
        start_request_profile,
        stop_request_profile,
    )

    arm_dir = out / f"r{round_index}_card{card}_{arm}"
    arm_dir.mkdir(parents=True, exist_ok=True)
    port = SERVER_PORT + 1000 * round_index + 200 * card
    server = pair_check.serve(
        card,
        pair_check.tree(pair_check.ARMS[arm][0]),
        pair_check.MODEL,
        port,
        ["--config", pair_check.CONFIG, "--colocate"],
        arm_dir / "server.log",
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        start_request_profile(base_url, f"{arm}_r{round_index}", str(arm_dir / "events"))
        rc = pair_check.benchmark(
            card,
            [
                "--host", "127.0.0.1", "--port", str(port), "--model", "qwen3-omni", "--lang", "en",
                "--voice-clone", "--stream", "--warmup", "1", "--generate-only", "--disable-tqdm",
                "--output-dir", str(arm_dir / "c1_seeded"), *BENCH_ARGS,
            ],
            arm_dir / "c1_seeded.log",
        )
        stop_request_profile(base_url, f"{arm}_r{round_index}")
    finally:
        pair_check.stop(card, server)
    pair_check.log(out, f"round {round_index} card {card} {arm} rc {rc}")


def request_offsets(event_dir: Path) -> list[dict[str, float]]:
    """Per request in arrival order: each (stage, event) first offset in ms, and the thinker to talker hops."""
    from sglang_omni.profiler.views import reconstruct_timelines

    requests = []
    for timeline in reconstruct_timelines(event_dir).values():
        events = sorted(timeline.events, key=lambda event: int(event["timestamp_ns"]))
        start_ns = int(events[0]["timestamp_ns"])
        offsets: dict[str, float] = {}
        sent: dict[int, int] = {}
        hops: dict[int, float] = {}
        for event in events:
            key = f"{event.get('stage')}:{event['event_name']}"
            timestamp_ns = int(event["timestamp_ns"])
            offsets.setdefault(key, (timestamp_ns - start_ns) / 1e6)
            metadata = event.get("metadata") or {}
            if event["event_name"] == "stage_stream_chunk_sent" and metadata.get("to_stage") == "talker_ar":
                sent.setdefault(int(metadata["chunk_id"]), timestamp_ns)
            elif event["event_name"] == "stage_stream_chunk_received" and metadata.get("from_stage") == "thinker":
                chunk_id = int(metadata["chunk_id"])
                if chunk_id in sent:
                    hops.setdefault(chunk_id, (timestamp_ns - sent[chunk_id]) / 1e6)
                else:
                    pass
            else:
                pass
        offsets["hop thinker->talker chunk 0"] = hops.get(0, float("nan"))
        later = [value for chunk_id, value in hops.items() if chunk_id > 0]
        offsets["hop thinker->talker later chunks (median)"] = (
            statistics.median(later) if later else float("nan")
        )
        offsets["start_ns"] = start_ns
        requests.append(offsets)
    requests.sort(key=lambda offsets: offsets["start_ns"])
    return requests[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", required=True, help="main,branch")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main_arm, branch_arm = args.arms.split(",")
    sys.path.insert(0, str(pair_check.tree(pair_check.ARMS[main_arm][0])))
    rounds = [((main_arm, 0), (branch_arm, 1)), ((branch_arm, 0), (main_arm, 1))]
    for round_index, placements in enumerate(rounds, start=1):
        threads = [
            threading.Thread(target=run_arm, args=(args.out, round_index, card, arm))
            for arm, card in placements
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    per_arm: dict[str, list[list[dict[str, float]]]] = {main_arm: [], branch_arm: []}
    for round_index, placements in enumerate(rounds, start=1):
        for arm, card in placements:
            per_arm[arm].append(request_offsets(args.out / f"r{round_index}_card{card}_{arm}" / "events"))
    keys = sorted(
        set.intersection(*(set(request) for runs in per_arm.values() for run in runs for request in run))
        - {"start_ns"}
    )
    table = []
    for key in keys:
        means = {
            arm: statistics.mean(
                statistics.mean(request[key] for request in run) for run in runs
            )
            for arm, runs in per_arm.items()
        }
        table.append(
            dict(event=key, main_ms=means[main_arm], branch_ms=means[branch_arm],
                 delta_ms=means[branch_arm] - means[main_arm])
        )
    table.sort(key=lambda row: row["main_ms"])
    for row in table:
        print(f"{row['event'][:70]:70s} main {row['main_ms']:9.2f}  branch {row['branch_ms']:9.2f}  "
              f"delta {row['delta_ms']:+7.2f} ms", flush=True)
    (args.out / "handoff.json").write_text(json.dumps(table, indent=1))
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
