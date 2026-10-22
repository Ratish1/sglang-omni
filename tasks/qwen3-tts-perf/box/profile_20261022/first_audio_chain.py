"""Each request's first-audio chain from a cell's request events: median, p90 and max of every gap between consecutive
milestones, admission to the first chunk at the coordinator (the 10-19 census chain).

usage: python3 first_audio_chain.py EVENT_DIR [EVENT_DIR ...]
"""

import json
import sys
from pathlib import Path

# (stage, event): each request's first occurrence, in first-audio order.
CHAIN = (
    ("coordinator", "request_admission"),
    ("preprocessing", "preprocess_start"),
    ("preprocessing", "preprocess_end"),
    ("tts_engine", "stage_input_received"),
    ("tts_engine", "scheduler_request_build_start"),
    ("tts_engine", "scheduler_request_build_end"),
    ("tts_engine", "scheduler_queue_enter"),
    ("tts_engine", "scheduler_prefill_start"),
    ("tts_engine", "scheduler_prefill_end"),
    ("tts_engine", "scheduler_first_emit"),
    ("tts_engine", "stage_first_stream_chunk_sent"),
    ("vocoder", "stage_stream_chunk_received"),
    ("vocoder", "qwen3_tts_vocoder_decode_enqueued"),
    ("vocoder", "qwen3_tts_vocoder_decode_dispatched"),
    ("vocoder", "qwen3_tts_vocoder_decode_launched"),
    ("vocoder", "qwen3_tts_vocoder_decode_resolved"),
    ("vocoder", "qwen3_tts_vocoder_decode_committed"),
    ("vocoder", "stage_first_stream_chunk_sent"),
    ("coordinator", "stage_stream_chunk_received"),
)


def chain_gaps(event_dir: Path) -> dict[str, list[float]]:
    first: dict[str, dict[tuple[str, str], int]] = {}
    for path in sorted(event_dir.glob("events_*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            milestones = first.setdefault(event["request_id"], {})
            key = (event["stage"], event["event_name"])
            milestones[key] = min(milestones.get(key, event["timestamp_ns"]), event["timestamp_ns"])
    gaps: dict[str, list[float]] = {}
    for milestones in first.values():
        present = [key for key in CHAIN if key in milestones]
        for earlier, later in zip(present, present[1:]):
            name = f"{earlier[0]}.{earlier[1]} -> {later[0]}.{later[1]}"
            gaps.setdefault(name, []).append((milestones[later] - milestones[earlier]) / 1e6)
        if CHAIN[0] in milestones and CHAIN[-1] in milestones:
            gaps.setdefault("TOTAL admission -> first chunk at coordinator", []).append(
                (milestones[CHAIN[-1]] - milestones[CHAIN[0]]) / 1e6
            )
        else:
            pass
    return gaps


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    for event_dir in sys.argv[1:]:
        gaps = chain_gaps(Path(event_dir))
        print(f"{event_dir}")
        print("| segment | n | median ms | p90 ms | max ms |")
        print("| --- | --- | --- | --- | --- |")
        for name, values in gaps.items():
            print(
                f"| {name} | {len(values)} | {quantile(values, 0.5):.2f} | {quantile(values, 0.9):.2f} | "
                f"{max(values):.2f} |"
            )


if __name__ == "__main__":
    main()
