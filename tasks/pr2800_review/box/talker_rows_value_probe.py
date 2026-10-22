"""Whether #2861's slower talker prefill comes from the rows' values: the same head with and without its rows.

Run on the box: python3 /workspace/box/tasks/pr2800_review/box/talker_rows_value_probe.py --arm phu
    --out /data/pr2800/<run>
run224 placed #2861's seeded c1 first-audio delta in the talker's prompt prefill (+3.24 ms on the same token count,
graph replays and no prefix cache). The talker is a 128-expert MoE: main feeds every multimodal row the same zero
vector, so those rows route to the same experts, where real rows spread over many. Both arms here serve the same tree;
the zeroed arm loads talker_zero_rows/sitecustomize.py, which replaces the rows with zeros at the talker only. Two
rounds, the arms on cards 0 and 1 at the same time, then swapped; seeded c1 on the first SAMPLES SeedTTS EN prompts
with request events, read as handoff_events_probe.py reads them. If the zeroed arm's talker prefill returns to main's,
the cost is the rows' values. Writes <out>/r<n>_card<c>_<arm>/, <out>/rows_value.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
from pathlib import Path

BOX = Path(__file__).resolve().parent
sys.path.insert(0, str(BOX))
import handoff_events_probe  # noqa: E402
import pair_check  # noqa: E402

ZERO_ROWS_CARDS: set[int] = set()
CARD_ENV = pair_check.card_env


def card_env(card: int, tree_path: Path) -> dict[str, str]:
    env = CARD_ENV(card, tree_path)
    if card in ZERO_ROWS_CARDS:
        env.update(PYTHONPATH=f"{BOX / 'talker_zero_rows'}:{tree_path}", TALKER_ZERO_ROWS="1")
    else:
        pass
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pair_check.tree("upk")))
    pair_check.card_env = card_env
    handoff_events_probe.PASSES = {"c1_seeded": handoff_events_probe.PASSES["c1_seeded"]}
    rounds = [(("rows", 0), ("zeros", 1)), (("zeros", 0), ("rows", 1))]
    for round_index, placements in enumerate(rounds, start=1):
        ZERO_ROWS_CARDS.clear()
        ZERO_ROWS_CARDS.update(card for name, card in placements if name == "zeros")
        threads = [
            threading.Thread(
                target=handoff_events_probe.run_arm,
                args=(args.out, round_index, card, args.arm, name),
            )
            for name, card in placements
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    per_arm: dict[str, list[list[dict[str, float]]]] = {"rows": [], "zeros": []}
    for round_index, placements in enumerate(rounds, start=1):
        for name, card in placements:
            per_arm[name].append(
                handoff_events_probe.request_offsets(
                    args.out / f"r{round_index}_card{card}_{name}" / "events_c1_seeded"
                )
            )
    intervals = {
        "talker prefill start to first codec chunk": (
            "talker_ar:scheduler_prefill_start", "talker_ar:stage_first_stream_chunk_sent"),
        "talker request build": ("talker_ar:scheduler_request_build_start", "talker_ar:scheduler_request_build_end"),
        "thinker prefill": ("thinker:scheduler_prefill_start", "thinker:scheduler_prefill_end"),
        "request to first audio": ("coordinator:request_admission", "code2wav:code2wav_first_audio"),
    }
    table = {}
    for label, (start, end) in intervals.items():
        means = {
            name: statistics.mean(
                statistics.mean(request[end] - request[start] for request in run) for run in runs
            )
            for name, runs in per_arm.items()
        }
        table[label] = means
        print(f"{label:45s} rows {means['rows']:8.2f}  zeros {means['zeros']:8.2f}  "
              f"rows - zeros {means['rows'] - means['zeros']:+6.2f} ms", flush=True)
    (args.out / "rows_value.json").write_text(json.dumps(table, indent=1))
    pair_check.log(args.out, "done")


if __name__ == "__main__":
    main()
