"""Do long references stall every other stream while they are preprocessed? From a pair.py run with --events, per arm:
the long references are the requests whose first vocoder decode reads more than LONG_FRAMES frames (no SeedTTS EN
reference reaches 120), less any whose preprocessing took under CACHE_HIT_S (a repeated reference served from the
reference cache); around each, the window from its preprocessing start to WINDOW_TAIL_S after its end, the longest
interval in which no other request's chunk reached the coordinator. The same statistic over every window of that
length in the run after its opening, away from the long requests, is the baseline.

usage: PAIR_ROOT=... python3 arrival_stalls.py LABEL
"""

import argparse
import json
import os
import statistics
from bisect import bisect_left, bisect_right
from pathlib import Path

ROOT = Path(os.environ.get("PAIR_ROOT", "/data/next1024/pairs"))
LONG_FRAMES = 200
CACHE_HIT_S = 0.02
WINDOW_TAIL_S = 0.3
OPENING_S = 2.0
STEP_S = 0.1
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


def arm_report(folder: Path) -> tuple[list[tuple[float, float]], list[float]]:
    rows = load(folder)
    preprocessing: dict[str, dict[str, float]] = {}
    wide, admitted, chunks = set(), [], []
    for row in rows:
        seconds = row["timestamp_ns"] / 1e9
        name, stage, metadata = row["event_name"], row["stage"], row.get("metadata") or {}
        if stage == "preprocessing" and name in ("stage_input_received", "stage_complete"):
            preprocessing.setdefault(row["request_id"], {})[name] = seconds
        elif name == "qwen3_tts_vocoder_decode_launched" and metadata.get("decoded_chunks") == 0:
            if (metadata.get("fresh_frames") or 0) > LONG_FRAMES:
                wide.add(row["request_id"])
            else:
                pass
        elif stage == "coordinator" and name == "request_admission":
            admitted.append(seconds)
        elif stage == "coordinator" and name == "stage_stream_chunk_received":
            chunks.append((seconds, row["request_id"]))
        else:
            pass
    # note (ratish): a long reference whose window reaches the last admission shares it with the run's tail, where the
    # other streams finish rather than stall, so it is read only while requests are still arriving.
    spans = sorted(
        (preprocessing[key]["stage_input_received"], preprocessing[key]["stage_complete"])
        for key in wide
        if preprocessing[key]["stage_complete"] - preprocessing[key]["stage_input_received"] > CACHE_HIT_S
        and preprocessing[key]["stage_complete"] + WINDOW_TAIL_S < max(admitted)
    )
    others = sorted(seconds for seconds, request in chunks if request not in wide)
    windows = [(1000 * (end_s - start_s), 1000 * longest_silence(others, start_s, end_s + WINDOW_TAIL_S)) for start_s, end_s in spans]
    length_s = statistics.median(end_s - start_s for start_s, end_s in spans) + WINDOW_TAIL_S
    blocked = [(start_s - CLEAR_S, end_s + WINDOW_TAIL_S + CLEAR_S) for start_s, end_s in spans]
    baseline = []
    start_s = min(admitted) + OPENING_S
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
    args = parser.parse_args()
    for arm in ("a", "b"):
        tree = (ROOT / args.label / arm / "tree.txt").read_text(encoding="utf-8").split()[0]
        windows, baseline = arm_report(ROOT / args.label / arm / "events")
        ordered = sorted(baseline)
        if ordered:
            quantile = lambda share: ordered[min(len(ordered) - 1, int(share * len(ordered)))]
            print(f"{args.label} {arm} {tree}: baseline windows {len(baseline)}, longest silence p50 {quantile(0.5):.1f} "
                  f"p95 {quantile(0.95):.1f} p99 {quantile(0.99):.1f} max {ordered[-1]:.1f} ms")
        else:
            print(f"{args.label} {arm} {tree}: no baseline window after the opening")
        for preprocess_ms, silence_ms in windows:
            print(f"  long reference: preprocessing {preprocess_ms:.1f} ms, longest silence of the others {silence_ms:.1f} ms")


if __name__ == "__main__":
    main()
