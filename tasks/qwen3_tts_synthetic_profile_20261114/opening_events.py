"""Server-side timeline of each stream's first two vocoder chunks under Poisson load, from the request event recorder.

Boots TREE alone on card N, runs Nari's client at RATE req/s, records request events for WINDOW seconds starting
START seconds into the client run, then reads, per stream: when its first and second decodes were enqueued, when the
first left the initial worker's queue (dispatched) and the second was launched, when each resolved, and the generated
frames each covered. Writes OUT/timeline.txt.

usage: python3 opening_events.py --tree TREE --card 0 --out DIR [--rate 80] [--start 60] [--window 10]
"""

import argparse
import collections
import json
import threading
import time
from pathlib import Path

import httpx

from bar_overload_pair import prepare_client, run_rate
from seedtts_pair import serve, stop
from served_baseline import MODELS, wait_healthy

EVENT_CAPACITY = 400000
# the frame gaps reported per request, from frame 1 to 2 up to FRAME_GAPS to FRAME_GAPS + 1
FRAME_GAPS = 5
HERE = Path(__file__).resolve().parent
FIRST_EVENTS = ("qwen3_tts_vocoder_decode_enqueued", "qwen3_tts_vocoder_decode_dispatched", "qwen3_tts_vocoder_decode_resolved")
SECOND_EVENTS = ("qwen3_tts_vocoder_decode_enqueued", "qwen3_tts_vocoder_decode_launched", "qwen3_tts_vocoder_decode_resolved")


def record_events(port: int, delay_s: int, window_s: int, event_dir: Path) -> None:
    time.sleep(delay_s)
    url = f"http://127.0.0.1:{port}"
    body = {"run_id": "opening", "event_dir": str(event_dir), "event_buffer_capacity": EVENT_CAPACITY}
    httpx.post(f"{url}/start_request_profile", json=body, timeout=60).raise_for_status()
    time.sleep(window_s)
    httpx.post(f"{url}/stop_request_profile", json={"run_id": "opening"}, timeout=60).raise_for_status()


def chunk_times(events: list[dict], chunk: int, names: tuple[str, str, str]) -> tuple[int, int, int, int] | None:
    """The first enqueue, start and resolve of the decode that commits chunk, and the generated frames it saw."""
    found = {}
    frames = None
    for event in events:
        metadata = event.get("metadata") or {}
        if metadata.get("decoded_chunks") != chunk or event["event_name"] not in names or event["event_name"] in found:
            continue
        else:
            pass
        found[event["event_name"]] = event["timestamp_ns"]
        if event["event_name"] == names[0]:
            frames = metadata.get("generated_frames")
        else:
            pass
    if len(found) < 3:
        return None
    else:
        pass
    return found[names[0]], found[names[1]], found[names[2]], frames


def quantiles(values: list[float]) -> str:
    values = sorted(values)
    return "/".join(f"{values[min(len(values) - 1, int(fraction * len(values)))]:.1f}" for fraction in (0.1, 0.5, 0.9))


def frame_intervals(by_request: dict[str, list[dict]]) -> str:
    """Per request, the gap from the ingest that first reached k generated frames to the one that reached k + 1."""
    gaps: dict[int, list[float]] = collections.defaultdict(list)
    for events in by_request.values():
        reached: dict[int, int] = {}
        for event in events:
            if event["event_name"] == "qwen3_tts_frames_ingested":
                frames = (event.get("metadata") or {}).get("generated_frames", 0)
                for count in range(1, frames + 1):
                    reached.setdefault(count, event["timestamp_ns"])
            else:
                pass
        for count in range(1, FRAME_GAPS + 1):
            if count in reached and count + 1 in reached:
                gaps[count].append((reached[count + 1] - reached[count]) / 1e6)
            else:
                pass
    return ", ".join(f"frame {count}->{count + 1} {quantiles(values)}" for count, values in sorted(gaps.items()) if values)


def digest(event_dir: Path) -> str:
    by_request: dict[str, list[dict]] = collections.defaultdict(list)
    for path in event_dir.glob("events_*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event_name"].startswith("qwen3_tts_"):
                by_request[event["request_id"]].append(event)
            else:
                pass
    for events in by_request.values():
        events.sort(key=lambda event: event["timestamp_ns"])
    intervals = frame_intervals(by_request)
    rows = []
    for events in by_request.values():
        first = chunk_times(events, 0, FIRST_EVENTS)
        second = chunk_times(events, 1, SECOND_EVENTS)
        if first is not None and second is not None:
            rows.append((first, second))
        else:
            pass
    if not rows:
        return "no stream with both chunks recorded"
    else:
        pass
    milliseconds = 1e6
    lines = [
        f"streams {len(rows)} (p10/p50/p90, ms)",
        f"first chunk frames {dict(collections.Counter(first[3] for first, _ in rows).most_common(3))}, "
        f"second chunk frames {dict(collections.Counter(second[3] for _, second in rows).most_common(3))}",
        f"first: enqueue to dispatch {quantiles([(first[1] - first[0]) / milliseconds for first, _ in rows])}, "
        f"dispatch to resolve {quantiles([(first[2] - first[1]) / milliseconds for first, _ in rows])}",
        f"second: enqueue after first resolve {quantiles([(second[0] - first[2]) / milliseconds for first, second in rows])}, "
        f"enqueue to launch {quantiles([(second[1] - second[0]) / milliseconds for _, second in rows])}, "
        f"launch to resolve {quantiles([(second[2] - second[1]) / milliseconds for _, second in rows])}",
        f"second resolve after first resolve {quantiles([(second[2] - first[2]) / milliseconds for first, second in rows])}",
        f"frames ingested: {intervals}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rate", type=int, default=80)
    parser.add_argument("--duration", type=int, default=90, help="measured seconds of the client run")
    parser.add_argument("--start", type=int, default=60, help="seconds into the client run to start recording")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--port", type=int, default=24000)
    args = parser.parse_args()
    prepare_client()
    model, _ = MODELS["customvoice"]
    args.out.mkdir(parents=True, exist_ok=True)
    server = serve(
        args.tree, model, args.card, args.port, args.out / "serve.log",
        {"PYTHONPATH": f"{HERE / 'frame_probe'}:{args.tree}", "QWEN3_TTS_FRAME_PROBE": "1"},
    )
    try:
        wait_healthy(args.port, server)
        event_dir = args.out / "events"
        recorder = threading.Thread(target=record_events, args=(args.port, args.start, args.window, event_dir), daemon=True)
        recorder.start()
        code = run_rate(args.port, model, args.rate, 0, args.out / f"rps{args.rate}", args.duration, {})
        recorder.join()
        time.sleep(5)
    finally:
        stop(server)
    report = digest(event_dir)
    (args.out / "timeline.txt").write_text(report + "\n")
    print(f"{time.strftime('%T')} opening_events rc {code}\n{report}", flush=True)


if __name__ == "__main__":
    main()
