"""Two boots' stage sends side by side from the comm trace (SGLANG_OMNI_COMM_TRACE=1).

Reads every COMM_TRACE line of each serve.log, groups the send events (comm_payload_send,
comm_stream_send) by event, sending stage, receiving stage and transport, and prints per group
the count and the median and p90 of each timing field in ms for both boots, with the median's
difference, so a hop whose write, queue wait or control send moved stands out.

usage: python comm_trace_hops.py SERVE_LOG_A SERVE_LOG_B
"""

from __future__ import annotations

import collections
import json
import statistics
import sys

SEND_EVENTS = ("comm_payload_send", "comm_stream_send")
TIMING_FIELDS = ("queue_wait_ms", "write_ms", "control_send_ms", "elapsed_ms")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def hop_timings(
    serve_log_path: str,
) -> dict[tuple[str, str, str, str], dict[str, list[float]]]:
    hops: dict[tuple[str, str, str, str], dict[str, list[float]]] = (
        collections.defaultdict(lambda: collections.defaultdict(list))
    )
    with open(serve_log_path, errors="replace") as serve_log:
        for line in serve_log:
            marker = line.find("COMM_TRACE ")
            if marker < 0:
                continue
            else:
                pass
            record = json.loads(line[marker + len("COMM_TRACE ") :])
            if record["event"] not in SEND_EVENTS:
                continue
            else:
                pass
            hop = (
                record["event"],
                str(record.get("from_stage")),
                str(record.get("to_stage")),
                str(record.get("transport")),
            )
            for field in TIMING_FIELDS:
                value = record.get(field)
                if isinstance(value, (int, float)) and value >= 0:
                    hops[hop][field].append(float(value))
                else:
                    pass
    return hops


def main() -> None:
    first, second = hop_timings(sys.argv[1]), hop_timings(sys.argv[2])
    for hop in sorted(set(first) | set(second)):
        counts = [
            len(first[hop]["elapsed_ms"]) if hop in first else 0,
            len(second[hop]["elapsed_ms"]) if hop in second else 0,
        ]
        print(f"{' '.join(hop)}  n={counts[0]}/{counts[1]}")
        for field in TIMING_FIELDS:
            values_first = first[hop][field] if hop in first else []
            values_second = second[hop][field] if hop in second else []
            if not values_first or not values_second:
                continue
            else:
                pass
            median_first = statistics.median(values_first)
            median_second = statistics.median(values_second)
            print(
                f"  {field:16s} median {median_first:9.3f} / {median_second:9.3f}"
                f" ({median_second - median_first:+8.3f})"
                f"  p90 {percentile(values_first, 0.9):9.3f} / {percentile(values_second, 0.9):9.3f}"
            )


if __name__ == "__main__":
    main()
