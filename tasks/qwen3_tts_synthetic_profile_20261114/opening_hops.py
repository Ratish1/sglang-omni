"""Offline digest of recorded request events: engine-to-vocoder hops, first decode launches, and streams by first chunk.

For each event directory: per stream, the engine's send of codec chunk k, the vocoder stage's receipt and the
scheduler's ingest of it; the first decode's launches (one per cohort) with their launch-to-resolve time and total;
and admission to first audio at the coordinator split by the first commit's samples (1,920 is a trimmed or one-frame
first chunk, 3,840 a two-frame chunk whose first frame failed the silence guard).

usage: python3 opening_hops.py EVENT_DIR [EVENT_DIR ...]
"""

import collections
import json
import statistics
import sys
from pathlib import Path

CHUNKS = (0, 1, 2, 3)
GAPS = (
    ("admission", "engine_dispatch"), ("engine_dispatch", "build_end"), ("build_end", "queue_enter"),
    ("queue_enter", "prefill_start"), ("sent0", "received0"), ("received0", "ingested0"), ("sent0", "sent1"),
    ("sent1", "received1"), ("received1", "ingested1"), ("sent1", "sent2"), ("received2", "ingested2"),
    ("sent2", "sent3"), ("received3", "ingested3"), ("committed", "vocoder_sent"), ("vocoder_sent", "coordinator"),
)


def quantiles(values: list[float]) -> str:
    if not values:
        return "none"
    else:
        pass
    values = sorted(values)
    picks = [values[min(len(values) - 1, int(fraction * len(values)))] for fraction in (0.1, 0.5, 0.9)]
    return "/".join(f"{value:.1f}" for value in picks) + f" mean {statistics.mean(values):.1f} n {len(values)}"


def stream_rows(event_dir: Path) -> list[dict]:
    by_request: dict[str, list[dict]] = collections.defaultdict(list)
    for path in event_dir.glob("events_*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            by_request[event["request_id"]].append(event)
    rows = []
    for events in by_request.values():
        events.sort(key=lambda event: event["timestamp_ns"])
        row: dict = {}
        for event in events:
            metadata = event.get("metadata") or {}
            name = event["event_name"]
            stamp = event["timestamp_ns"]
            stage = event.get("stage")
            chunks = metadata.get("decoded_chunks")
            if name == "request_admission":
                row.setdefault("admission", stamp)
            elif name == "stage_dispatch" and stage == "tts_engine":
                row.setdefault("engine_dispatch", stamp)
            elif name == "scheduler_request_build_end":
                row.setdefault("build_end", stamp)
            elif name == "scheduler_queue_enter":
                row.setdefault("queue_enter", stamp)
            elif name == "scheduler_prefill_start":
                row.setdefault("prefill_start", stamp)
            elif name == "stage_stream_chunk_sent" and stage == "tts_engine" and metadata.get("chunk_id") in CHUNKS:
                row.setdefault(f"sent{metadata['chunk_id']}", stamp)
            elif name == "stage_stream_chunk_received" and stage == "vocoder" and metadata.get("chunk_id") in CHUNKS:
                row.setdefault(f"received{metadata['chunk_id']}", stamp)
            elif name == "qwen3_tts_frames_ingested" and metadata.get("generated_frames", 0) - 1 in CHUNKS:
                row.setdefault(f"ingested{metadata['generated_frames'] - 1}", stamp)
            elif name == "qwen3_tts_vocoder_decode_enqueued" and chunks == 0:
                row.setdefault("enqueued", stamp)
            elif name == "qwen3_tts_vocoder_decode_launched" and chunks == 0 and "launched" not in row:
                row["launched"] = stamp
                row["width"] = metadata.get("fresh_frames", metadata.get("path"))
                row["cohort"] = metadata.get("cohort_size")
            elif name == "qwen3_tts_vocoder_decode_resolved" and chunks == 0:
                row.setdefault("resolved", stamp)
            elif name == "qwen3_tts_vocoder_decode_committed" and chunks == 1 and "committed" not in row:
                row["committed"] = stamp
                row["samples"] = metadata.get("samples")
            elif name == "stage_stream_chunk_sent" and stage == "vocoder" and metadata.get("chunk_id") == 0:
                row.setdefault("vocoder_sent", stamp)
            elif name == "coordinator_stream_received":
                row.setdefault("coordinator", stamp)
            else:
                pass
        if "admission" in row and "committed" in row:
            rows.append(row)
        else:
            pass
    rows.sort(key=lambda row: row["admission"])
    return rows


def digest(event_dir: Path) -> str:
    rows = stream_rows(event_dir)
    rows = rows[128 if len(rows) > 500 else (32 if len(rows) > 200 else 2):]
    lines = [f"{event_dir}: streams {len(rows)} (p10/p50/p90 mean n, ms)"]
    for before, after in GAPS:
        gaps = [(row[after] - row[before]) / 1e6 for row in rows if before in row and after in row]
        lines.append(f"  {before:15s} -> {after:13s} {quantiles(gaps)}")
    launches: dict[tuple[int, int], tuple] = {}
    for row in rows:
        if "launched" in row and "resolved" in row:
            launches[(row["launched"], row["resolved"])] = (row["width"], row["cohort"], (row["resolved"] - row["launched"]) / 1e6)
        else:
            pass
    by_width: dict = collections.defaultdict(list)
    for width, _, busy_ms in launches.values():
        by_width[width].append(busy_ms)
    for width, busy in sorted(by_width.items(), key=lambda entry: str(entry[0])):
        lines.append(f"  first decode launches, width {width}: {len(busy)} launches, launch to resolve {quantiles(busy)}, total {sum(busy):.0f} ms")
    by_samples: dict = collections.defaultdict(list)
    for row in rows:
        if "coordinator" in row:
            by_samples[(row.get("width"), row.get("samples"))].append(row)
        else:
            pass
    for (width, samples), group in sorted(by_samples.items(), key=lambda entry: str(entry[0])):
        first_audio = [(row["coordinator"] - row["admission"]) / 1e6 for row in group]
        first_frame = [(row["coordinator"] - row["ingested0"]) / 1e6 for row in group if "ingested0" in row]
        lines.append(
            f"  first decode width {width}, first commit {samples} samples: admission to coordinator {quantiles(first_audio)}, "
            f"frame 1 ingest to coordinator {quantiles(first_frame)}"
        )
    return "\n".join(lines)


def main() -> None:
    for argument in sys.argv[1:]:
        print(digest(Path(argument)), flush=True)


if __name__ == "__main__":
    main()
