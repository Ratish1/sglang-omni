"""Do long references stall every other stream while they are preprocessed? From a pair.py run with --events, per arm:
the LONGS requests with the longest preprocessing (stage input received to stage complete) are the long references;
around each, the window from its preprocessing start to WINDOW_TAIL_S after its end, the longest interval in which no
other request's chunk reached the coordinator. The same statistic over every window of that length in the run's
steady part, away from the long requests, is the baseline (p50, p95, p99, max).

usage: PAIR_ROOT=... python3 arrival_stalls.py LABEL [--longs 5]
"""

import argparse
import json
import os
import statistics
from bisect import bisect_left, bisect_right
from pathlib import Path

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/next1024/pairs"))
WINDOW_TAIL_S = 0.3
STEADY_SKIP_S = 10.0
STEP_S = 0.25
CLEAR_S = 1.0


def load(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def longest_silence(times_s: list[float], start_s: float, end_s: float) -> float:
    inside = times_s[bisect_left(times_s, start_s) : bisect_right(times_s, end_s)]
    edges = [start_s, *inside, end_s]
    return max(later - earlier for earlier, later in zip(edges, edges[1:]))


def arm_report(folder: Path, longs: int) -> tuple[list[tuple[float, float]], list[float]]:
    rows = load(folder)
    received, completed, admitted = {}, {}, []
    chunks = []
    for row in rows:
        seconds = row["timestamp_ns"] / 1e9
        name, stage = row["event_name"], row["stage"]
        if stage == "preprocessing" and name == "stage_input_received":
            received[row["request_id"]] = seconds
        elif stage == "preprocessing" and name == "stage_complete":
            completed[row["request_id"]] = seconds
        elif stage == "coordinator" and name == "request_admission":
            admitted.append(seconds)
        elif stage == "coordinator" and name == "stage_stream_chunk_received":
            chunks.append((seconds, row["request_id"]))
        else:
            pass
    spans = sorted(
        ((completed[key] - received[key], received[key], completed[key], key) for key in received if key in completed),
        reverse=True,
    )[:longs]
    chunks.sort()
    windows = []
    for _, start_s, end_s, key in sorted(spans, key=lambda span: span[1]):
        others = [seconds for seconds, request in chunks if request != key]
        windows.append((1000 * (end_s - start_s), 1000 * longest_silence(others, start_s, end_s + WINDOW_TAIL_S)))
    long_keys = {span[3] for span in spans}
    others = [seconds for seconds, request in chunks if request not in long_keys]
    length_s = statistics.median([span[0] for span in spans]) + WINDOW_TAIL_S
    blocked = [(span[1] - CLEAR_S, span[2] + WINDOW_TAIL_S + CLEAR_S) for span in spans]
    baseline = []
    start_s = min(admitted) + STEADY_SKIP_S
    while start_s + length_s < max(admitted):
        if not any(low < start_s + length_s and start_s < high for low, high in blocked):
            baseline.append(1000 * longest_silence(others, start_s, start_s + length_s))
        else:
            pass
        start_s += STEP_S
    return windows, baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("--longs", type=int, default=5)
    args = parser.parse_args()
    for arm in ("a", "b"):
        windows, baseline = arm_report(ROOT / args.label / arm / "events", args.longs)
        ordered = sorted(baseline)
        if ordered:
            quantile = lambda share: ordered[min(len(ordered) - 1, int(share * len(ordered)))]
            print(f"{args.label} {arm}: baseline windows {len(baseline)}, longest silence p50 {quantile(0.5):.1f} "
                  f"p95 {quantile(0.95):.1f} p99 {quantile(0.99):.1f} max {ordered[-1]:.1f} ms")
        else:
            print(f"{args.label} {arm}: no baseline window in the run's steady part")
        for preprocess_ms, silence_ms in windows:
            print(f"  long reference: preprocessing {preprocess_ms:.1f} ms, longest silence of the others {silence_ms:.1f} ms")


if __name__ == "__main__":
    main()
