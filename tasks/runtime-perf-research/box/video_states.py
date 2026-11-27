"""Where a video request's time goes in a mixed cell: per stage durations of every video request, by arm.

usage: python3 video_states.py --cell c64_events <run dir> main=boot,boot fix=boot,boot

A video request is one the image encoder ran. From the event files of each boot's pass, per request:
  hop to preprocessing   coordinator request_admission to preprocessing stage_input_received
  preprocessing wait     preprocessing stage_input_received to preprocess_start
  preprocessing          preprocess_start to preprocess_end
  to encoder end         preprocess_end to image_encoder encoder_end
  thinker wait           max(preprocess_end, encoder_end) to thinker scheduler_prefill_start (hops, build, queue)
  thinker prefill        scheduler_prefill_start to scheduler_prefill_end
  decode                 scheduler_prefill_end to thinker stage_complete (a text reply emits no first chunk event)
  end to end             request_admission to coordinator terminal_response
Prints mean, p50, p95 and p99 per state per arm, and the change of the arm means.
"""

import argparse
import collections
import glob
import json
import statistics
from pathlib import Path

STATES = (
    "hop to preprocessing",
    "preprocessing wait",
    "preprocessing",
    "to encoder end",
    "thinker wait",
    "thinker prefill",
    "decode",
    "end to end",
)


def events_of(boot: Path, cell: str) -> dict[str, dict[tuple[str, str], int]]:
    """First timestamp of each (stage, event) per request."""
    seen: dict[str, dict[tuple[str, str], int]] = collections.defaultdict(dict)
    for path in glob.glob(str(boot / cell / "events" / "events_*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (event["stage"], event["event_name"])
                request = seen[event["request_id"]]
                if key not in request:
                    request[key] = event["timestamp_ns"]
                else:
                    pass
    return seen


def durations(request: dict[tuple[str, str], int]) -> dict[str, float] | None:
    get = request.get
    needed = [
        ("coordinator", "request_admission"),
        ("preprocessing", "stage_input_received"),
        ("preprocessing", "preprocess_start"),
        ("preprocessing", "preprocess_end"),
        ("image_encoder", "encoder_end"),
        ("thinker", "scheduler_prefill_start"),
        ("thinker", "scheduler_prefill_end"),
        ("thinker", "stage_complete"),
        ("coordinator", "terminal_response"),
    ]
    if any(get(key) is None for key in needed):
        return None
    else:
        pass
    ms = lambda start, end: (get(end) - get(start)) / 1e6  # noqa: E731
    ready = max(
        get(("preprocessing", "preprocess_end")), get(("image_encoder", "encoder_end"))
    )
    return {
        "hop to preprocessing": ms(
            ("coordinator", "request_admission"),
            ("preprocessing", "stage_input_received"),
        ),
        "preprocessing wait": ms(
            ("preprocessing", "stage_input_received"),
            ("preprocessing", "preprocess_start"),
        ),
        "preprocessing": ms(
            ("preprocessing", "preprocess_start"), ("preprocessing", "preprocess_end")
        ),
        "to encoder end": ms(
            ("preprocessing", "preprocess_end"), ("image_encoder", "encoder_end")
        ),
        "thinker wait": (get(("thinker", "scheduler_prefill_start")) - ready) / 1e6,
        "thinker prefill": ms(
            ("thinker", "scheduler_prefill_start"), ("thinker", "scheduler_prefill_end")
        ),
        "decode": ms(
            ("thinker", "scheduler_prefill_end"), ("thinker", "stage_complete")
        ),
        "end to end": ms(
            ("coordinator", "request_admission"), ("coordinator", "terminal_response")
        ),
    }


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(q * len(ordered)), len(ordered) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", default="c64_events")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("arms", nargs="+")
    args = parser.parse_args()
    means: dict[str, dict[str, float]] = {}
    for spec in args.arms:
        name, boots = spec.split("=")
        rows: list[dict[str, float]] = []
        for boot in boots.split(","):
            for request in events_of(args.run_dir / boot, args.cell).values():
                if ("image_encoder", "encoder_end") not in request:
                    continue
                else:
                    pass
                row = durations(request)
                if row is not None:
                    rows.append(row)
                else:
                    pass
        print(f"{name}: {len(rows)} video requests over {boots}")
        means[name] = {}
        for state in STATES:
            values = [row[state] for row in rows]
            means[name][state] = statistics.fmean(values)
            print(
                f"  {state:22s} mean {statistics.fmean(values):8.1f}  p50 {quantile(values, 0.5):8.1f}  "
                f"p95 {quantile(values, 0.95):8.1f}  p99 {quantile(values, 0.99):8.1f} ms"
            )
    names = list(means)
    for other in names[1:]:
        print(f"{other} against {names[0]}, change of the mean:")
        for state in STATES:
            base, value = means[names[0]][state], means[other][state]
            print(
                f"  {state:22s} {value - base:+8.1f} ms ({(value - base) / base * 100 if base else 0:+.1f} %)"
            )


if __name__ == "__main__":
    main()
