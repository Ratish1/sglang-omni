"""Our talker's per-step codec EOS (`probe/qwen3-tts-teacher-forced`'s eos lines) against official qwen-tts teacher-forced
on the same frames (teacher_forced_reference.py), joined by speech id and frame. Prints per outcome and set: the steps
compared, the absolute difference of EOS's log-probability (temperature 1, whole vocabulary) where both are finite,
how often the top id agrees and the difference of its log-probability; for runaways, EOS under temperature 0.9 and
top-k 50 at our peak step in both, and the step range around it; for natural stops, both at the stop step; then the
cases whose EOS log-probabilities part most, and the first step where each parts by more than 1 nat.

usage: python3 teacher_forced_compare.py PROBE_DIR OFFICIAL.jsonl
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

SPEECH_ID = re.compile(r"speech-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
WORST_CASES = 15


def ours(probe: Path) -> dict[str, dict[int, list[float]]]:
    steps: dict[str, dict[int, list[float]]] = defaultdict(dict)
    for path in sorted(probe.glob("eos_*.txt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            request_id, frame, *fields = line.split()
            match = SPEECH_ID.search(request_id)
            if match is None:
                continue
            else:
                pass
            steps[match.group(0)][int(frame)] = [float(field) for field in fields]
    return steps


def quantiles(values: list[float]) -> str:
    if not values:
        return "-"
    else:
        pass
    return " / ".join(f"{value:.3g}" for value in np.quantile(np.array(values), (0.5, 0.9, 0.99)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", type=Path)
    parser.add_argument("official", type=Path)
    args = parser.parse_args()
    served = ours(args.probe)
    rows: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    parting: list[tuple[float, str, str, int, int]] = []
    for line in args.official.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        steps = served.get(record["speech_id"], {})
        official = record["official"]
        key = f"{record['finish_reason']} {record['set']}"
        bucket = rows[key]
        bucket["cases"].append(1.0)
        largest, first_part = 0.0, -1
        for frame, mine in sorted(steps.items()):
            if frame >= len(official):
                continue
            else:
                pass
            ours_eos, ours_top_k, _, ours_top, ours_top_log = mine
            official_eos, official_top_k, _, official_top, official_top_log = official[frame]
            bucket["steps"].append(1.0)
            if math.isfinite(ours_eos) and math.isfinite(official_eos):
                difference = abs(ours_eos - official_eos)
                bucket["eos_difference"].append(difference)
                if difference > largest:
                    largest = difference
                else:
                    pass
                if difference > 1.0 and first_part < 0:
                    first_part = frame
                else:
                    pass
            else:
                pass
            bucket["top_agrees"].append(float(int(ours_top) == int(official_top)))
            bucket["top_difference"].append(abs(ours_top_log - official_top_log))
        if steps:
            if record["finish_reason"] == "length":
                peak = max(steps, key=lambda frame: steps[frame][1])
                bucket["ours_at_peak"].append(steps[peak][1])
                bucket["official_at_peak"].append(official[peak][1] if peak < len(official) else float("nan"))
                bucket["official_peak_in_window"].append(
                    max(official[frame][1] for frame in range(max(0, peak - 3), min(len(official), peak + 4)))
                )
            else:
                stop = max(steps)
                bucket["ours_at_stop"].append(steps[stop][1])
                bucket["official_at_stop"].append(official[stop][1] if stop < len(official) else float("nan"))
            parting.append((largest, record["id"], key, first_part, record["frames"]))
        else:
            bucket["missing"].append(1.0)
    print(
        "outcome set | cases (missing ours) | steps | |EOS log-prob difference| q50 / q90 / q99 | top id agrees | "
        "|top log-prob difference| q50 / q90 / q99 | EOS top-k: ours at our peak / official there / official max "
        "within 3 steps (runaways), ours at the stop / official there (natural), medians"
    )
    for key in sorted(rows):
        bucket = rows[key]
        if key.startswith("length"):
            ends = (
                f"{np.median(bucket['ours_at_peak']):.3g} / {np.nanmedian(bucket['official_at_peak']):.3g} / "
                f"{np.median(bucket['official_peak_in_window']):.3g}"
                if bucket["ours_at_peak"]
                else "-"
            )
        else:
            ends = (
                f"{np.median(bucket['ours_at_stop']):.3g} / {np.nanmedian(bucket['official_at_stop']):.3g}"
                if bucket["ours_at_stop"]
                else "-"
            )
        print(
            f"{key} | {len(bucket['cases'])} ({len(bucket['missing'])}) | {len(bucket['steps'])} | "
            f"{quantiles(bucket['eos_difference'])} | "
            f"{100 * np.mean(bucket['top_agrees']) if bucket['top_agrees'] else float('nan'):.1f} % | "
            f"{quantiles(bucket['top_difference'])} | {ends}"
        )
    print("largest |EOS log-prob difference| | id | outcome set | first step parting by more than 1 nat | frames")
    for largest, case_id, key, first_part, frames in sorted(parting, reverse=True)[:WORST_CASES]:
        print(f"{largest:.3g} | {case_id} | {key} | {first_part} | {frames}")


if __name__ == "__main__":
    main()
