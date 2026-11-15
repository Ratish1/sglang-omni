"""Where a served video request's time goes, per boot, from one events cell: mean and p90 ms of each state between
consecutive marks (admission, preprocessing start and end, its send to the image encoder, image encoder in and out, thinker queue, prefill start and end,
thinker complete, terminal response).

usage: python3 video_states.py <events cell dir> [...]
"""

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MARKS = (
    ("coordinator", "request_admission"),
    ("preprocessing", "preprocess_start"),
    ("preprocessing", "preprocess_end"),
    ("preprocessing", "stage_hop_sent"),
    ("image_encoder", "stage_input_received"),
    ("image_encoder", "stage_complete"),
    ("thinker", "scheduler_queue_enter"),
    ("thinker", "scheduler_prefill_start"),
    ("thinker", "scheduler_prefill_end"),
    ("thinker", "stage_complete"),
    ("coordinator", "terminal_response"),
)
STATES = [f"{end[1]} after {start[1]}" for start, end in zip(MARKS, MARKS[1:])]


def first_marks(cell: Path) -> dict[str, dict[tuple[str, str], int]]:
    marks: dict[str, dict[tuple[str, str], int]] = defaultdict(dict)
    for path in glob.glob(str(cell / "events" / "events_*.jsonl")):
        with open(path) as source:
            events = [json.loads(line) for line in source]
        for event in events:
            key = (event["stage"], event["event_name"])
            if key in MARKS:
                marks[event["request_id"]].setdefault(key, event["timestamp_ns"])
            else:
                pass
    return marks


def main() -> None:
    for argument in sys.argv[1:]:
        marks = first_marks(Path(argument))
        complete = [stamps for stamps in marks.values() if len(stamps) == len(MARKS)]
        print(f"{argument}: {len(complete)} of {len(marks)} requests with every mark")
        for state, start, end in zip(STATES, MARKS, MARKS[1:]):
            spans = sorted((stamps[end] - stamps[start]) / 1e6 for stamps in complete)
            print(
                f"  {state:58s} mean {statistics.mean(spans):8.1f}  "
                f"p90 {spans[int(0.9 * (len(spans) - 1))]:8.1f}"
            )


if __name__ == "__main__":
    main()
