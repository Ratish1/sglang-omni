# SPDX-License-Identifier: Apache-2.0
"""Join full SeedTTS results to native Talker and Code2Wav request events."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BOUNDARIES = {
    "admitted": ("coordinator", "request_admission"),
    "queued": ("talker_ar", "scheduler_queue_enter"),
    "prefill_start": ("talker_ar", "scheduler_prefill_start"),
    "prefill_end": ("talker_ar", "scheduler_prefill_end"),
    "talker_complete": ("talker_ar", "stage_complete"),
    "audio_complete": ("code2wav", "stage_complete"),
}
INTERVALS = {
    "before_talker_queue_ms": ("admitted", "queued"),
    "talker_queue_ms": ("queued", "prefill_start"),
    "talker_first_forward_ms": ("prefill_start", "prefill_end"),
    "talker_after_prefill_ms": ("prefill_end", "talker_complete"),
    "audio_after_talker_ms": ("talker_complete", "audio_complete"),
    "admission_to_audio_ms": ("admitted", "audio_complete"),
}
SAMPLE_COUNT = 1088
TAIL_COUNT = math.ceil(SAMPLE_COUNT * 0.01)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    return ordered[low] + (index - low) * (ordered[math.ceil(index)] - ordered[low])


def measured_tail_ids(history: list[Path]) -> list[str]:
    counts: Counter[tuple[str, str]] = Counter()
    expected_ids: set[str] | None = None
    for parent in history:
        for arm in sorted(parent.glob("[1-4]-*/seedtts/speed_results.json")):
            variant = arm.parent.parent.name.split("-", 1)[1]
            rows = json.loads(arm.read_bytes())["per_request"]
            ids = {row["id"] for row in rows}
            assert len(rows) == len(ids) == SAMPLE_COUNT
            assert all(row["is_success"] for row in rows)
            if expected_ids is None:
                expected_ids = ids
            else:
                assert ids == expected_ids
            ranked = sorted(rows, key=lambda row: (-row["latency_s"], row["id"]))
            counts.update((variant, row["id"]) for row in ranked[:TAIL_COUNT])
    return sorted({rid for (_, rid), count in counts.items() if count >= 2})


def load_run(run: Path) -> dict[str, dict[str, float]]:
    client_rows = json.loads((run / "seedtts/speed_results.json").read_bytes())[
        "per_request"
    ]
    response_ids = json.loads((run / "seedtts/response_ids.json").read_bytes())
    event_dir = run / "events"
    assert len(client_rows) == len(response_ids) == SAMPLE_COUNT
    assert all(row["is_success"] for row in client_rows)
    assert set(response_ids) == {row["id"] for row in client_rows}
    assert all(value.startswith("chatcmpl-") for value in response_ids.values())
    sample_by_rid = {
        value.removeprefix("chatcmpl-"): sample
        for sample, value in response_ids.items()
    }
    assert len(sample_by_rid) == SAMPLE_COUNT
    events_by_rid: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    files = set(event_dir.glob("events_*.jsonl"))
    statuses = [
        json.loads(status.read_bytes())
        for status in sorted(event_dir.glob("run_status_*.json"))
    ]
    assert len(files) == len(statuses) == 7, run
    assert {event_dir / Path(s["event_path"]).name for s in statuses} == files
    for status in statuses:
        assert status["run_id"] == run.name and status["complete"], status
        path = event_dir / Path(status["event_path"]).name
        count = 0
        with path.open() as events:
            for count, line in enumerate(events, 1):
                event = json.loads(line)
                rid = event["request_id"]
                if rid in sample_by_rid:
                    key = (event["stage"], event["event_name"])
                    events_by_rid[rid][key].append(event)
                else:
                    pass
        assert count == status["written_events"], path
    assert set(events_by_rid) == set(sample_by_rid), run
    return {
        row["id"]: request_intervals(
            row,
            events_by_rid[response_ids[row["id"]].removeprefix("chatcmpl-")],
        )
        for row in client_rows
    }


def request_intervals(
    client_row: dict, events: dict[tuple[str, str], list[dict]]
) -> dict[str, float]:
    sample_id = client_row["id"]
    times: dict[str, int] = {}
    for name, key in BOUNDARIES.items():
        records = events.get(key, [])
        assert len(records) == 1, (sample_id, key, len(records))
        times[name] = records[0]["timestamp_ns"]
    assert list(times.values()) == sorted(times.values()), (sample_id, times)
    starts = events[("code2wav", "code2wav_decode_start")]
    ends = events[("code2wav", "code2wav_decode_end")]
    starts_by_frame = {event["metadata"]["start_frame"]: event for event in starts}
    ends_by_frame = {event["metadata"]["start_frame"]: event for event in ends}
    assert starts and len(starts_by_frame) == len(starts) == len(ends), sample_id
    assert starts_by_frame.keys() == ends_by_frame.keys(), sample_id
    decode_host_ns = 0
    for frame, start in starts_by_frame.items():
        end = ends_by_frame[frame]
        assert start["metadata"]["end_frame"] == end["metadata"]["end_frame"]
        elapsed = end["timestamp_ns"] - start["timestamp_ns"]
        assert elapsed >= 0, (sample_id, frame, elapsed)
        decode_host_ns += elapsed
    intervals = {
        name: (times[end] - times[start]) / 1e6
        for name, (start, end) in INTERVALS.items()
    }
    return {
        **intervals,
        "client_latency_ms": client_row["latency_s"] * 1000,
        "audio_duration_s": client_row["audio_duration_s"],
        "codec_frames": sum(event["metadata"]["new_frames"] for event in ends),
        "code2wav_windows": len(ends),
        # Includes dispatch, synchronization and host copies. These windows
        # can overlap Talker work; do not add them to the critical path.
        "code2wav_window_host_ms": decode_host_ns / 1e6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile_root", type=Path)
    parser.add_argument("history", type=Path, nargs="*")
    arguments = parser.parse_args()
    witnesses = measured_tail_ids(arguments.history)
    runs = {}
    for run in sorted(arguments.profile_root.resolve().glob("[1-4]-*")):
        if (run / "seedtts/response_ids.json").exists():
            rows = load_run(run)
        else:
            continue
        fastest = sorted(rows, key=lambda rid: rows[rid]["client_latency_ms"])
        groups = {"all": fastest, "slowest": fastest[-TAIL_COUNT:]}
        if witnesses:
            assert set(witnesses) <= rows.keys()
            groups["historical_witnesses"] = witnesses
        else:
            pass
        runs[run.name] = {
            "joined": len(rows),
            "intervals": {
                group: {
                    field: {
                        "median": statistics.median(values),
                        "p95": percentile(values, 0.95),
                        "p99": percentile(values, 0.99),
                    }
                    for field in next(iter(rows.values()))
                    for values in [[rows[rid][field] for rid in group_ids]]
                }
                for group, group_ids in groups.items()
            },
            "witnesses": {rid: rows[rid] for rid in witnesses},
            "per_request": rows,
        }
    assert runs, arguments.profile_root
    print(json.dumps(runs, sort_keys=True))


if __name__ == "__main__":
    main()
