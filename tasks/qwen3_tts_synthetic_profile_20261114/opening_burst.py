"""The silent opening frame under bursts of streamed CustomVoice requests, for one tree on one card.

Boots TREE on card N and runs each burst: COUNT streamed PCM requests (Ryan, English, default sampling) released
together. Per request it records the total samples and whether the first frame's 1,920 samples pass the
vocoder's own silence guard (RMS 0.001, peak 0.0032). Request events, recorded per burst, give each stream's
first decode width (fresh_frames of its first launch) and whether the engine finished the request before the
vocoder dispatched that first decode. Writes OUT/<burst>/requests.jsonl and OUT/<burst>/events, prints one
line per burst and the vocoder's codec counters at the end.

usage: python3 opening_burst.py --tree TREE --card 0 --out DIR [--port 26000] [--count 48] [--rounds 3]
"""

import argparse
import collections
import concurrent.futures
import json
import threading
import time
from pathlib import Path

import httpx
import numpy as np

from seedtts_pair import serve, stop
from served_baseline import MODELS, wait_healthy
from slot_wait_serving import INPUTS, codec_state

FRAME_SAMPLES = 1920
SILENCE_MAX_RMS = 0.001
SILENCE_MAX_PEAK = 0.0032
EVENT_CAPACITY = 400000
REQUEST_TIMEOUT_S = 300
PHRASES = (
    "Yes, of course.",
    "Thank you so much.",
    "Hello there, how are you?",
    "Please hold for a moment.",
    "That sounds good to me.",
    "I can help with that.",
    "Sure, one second.",
    "Goodbye, and take care.",
)
# per burst: extra request fields, and whether its inputs are the short phrases
BURSTS = (
    ("capped", {"max_new_tokens": 3}, True),
    ("capped_off", {"max_new_tokens": 3, "suppress_bootstrap_silence": False}, True),
    ("one_frame", {"max_new_tokens": 1}, True),
    ("override", {"max_new_tokens": 8, "initial_codec_chunk_frames": 4}, True),
    ("override_off", {"max_new_tokens": 8, "initial_codec_chunk_frames": 4, "suppress_bootstrap_silence": False}, True),
    ("normal", {}, False),
)
REPEATED_BURSTS = ("capped", "override")


def run_request(url: str, body: dict, barrier: threading.Barrier) -> dict:
    barrier.wait()
    start = time.monotonic()
    first_byte_s = None
    audio = bytearray()
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        with client.stream("POST", f"{url}/v1/audio/speech", json=body) as response:
            if response.status_code != 200:
                return {"status": response.status_code, "error": response.read().decode(errors="replace")[:300]}
            else:
                pass
            for piece in response.iter_bytes():
                if first_byte_s is None:
                    first_byte_s = time.monotonic() - start
                else:
                    pass
                audio.extend(piece)
    samples = np.frombuffer(bytes(audio[: len(audio) // 2 * 2]), dtype=np.int16).astype(np.float32) / 32768.0
    head = samples[:FRAME_SAMPLES]
    if head.size:
        rms = float(np.sqrt(np.mean(head * head)))
        peak = float(np.max(np.abs(head)))
    else:
        rms, peak = 0.0, 0.0
    return {
        "status": 200, "samples": int(samples.size), "first_byte_s": first_byte_s,
        "head_rms": rms, "head_peak": peak, "is_silent_opening": rms <= SILENCE_MAX_RMS and peak <= SILENCE_MAX_PEAK,
    }


def stream_rows(event_dir: Path) -> list[dict]:
    """Per server request: its first decode's width, and whether the engine finished before that decode was dispatched."""
    by_request: dict[str, list[dict]] = collections.defaultdict(list)
    for path in event_dir.glob("events_*.jsonl"):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            by_request[event["request_id"]].append(event)
    rows = []
    for request_id, events in by_request.items():
        events.sort(key=lambda event: event["timestamp_ns"])
        first_launch = None
        first_dispatch_ns = None
        engine_done_ns = None
        for event in events:
            metadata = event.get("metadata") or {}
            name = event["event_name"]
            if name == "qwen3_tts_vocoder_decode_launched" and metadata.get("decoded_chunks") == 0 and first_launch is None:
                first_launch = metadata.get("fresh_frames", metadata.get("path"))
            elif name == "qwen3_tts_vocoder_decode_dispatched" and metadata.get("decoded_chunks") == 0 and first_dispatch_ns is None:
                first_dispatch_ns = event["timestamp_ns"]
            elif name == "stage_complete" and event.get("stage") == "tts_engine" and engine_done_ns is None:
                engine_done_ns = event["timestamp_ns"]
            else:
                pass
        if first_launch is None and first_dispatch_ns is None:
            continue
        else:
            pass
        rows.append({
            "request_id": request_id, "first_launch_frames": first_launch,
            "is_done_before_dispatch": engine_done_ns is not None and first_dispatch_ns is not None and engine_done_ns < first_dispatch_ns,
        })
    return rows


def run_burst(url: str, port: int, name: str, extra: dict, is_short: bool, count: int, out: Path) -> str:
    out.mkdir(parents=True, exist_ok=True)
    event_dir = out / "events"
    profile = {"run_id": out.name, "event_dir": str(event_dir), "event_buffer_capacity": EVENT_CAPACITY}
    httpx.post(f"{url}/start_request_profile", json=profile, timeout=60).raise_for_status()
    texts = PHRASES if is_short else INPUTS
    bodies = [
        {
            "model": MODELS["customvoice"][0], "input": texts[index % len(texts)], "voice": "Ryan",
            "language": "English", "response_format": "pcm", "stream": True, **extra,
        }
        for index in range(count)
    ]
    barrier = threading.Barrier(count)
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(lambda body: run_request(url, body, barrier), bodies))
    time.sleep(2)
    httpx.post(f"{url}/stop_request_profile", json={"run_id": out.name}, timeout=60).raise_for_status()
    time.sleep(3)
    with (out / "requests.jsonl").open("w") as handle:
        for body, result in zip(bodies, results):
            handle.write(json.dumps({"input": body["input"], **result}) + "\n")
    served = [result for result in results if result["status"] == 200]
    rows = stream_rows(event_dir)
    late_rows = [row for row in rows if row["is_done_before_dispatch"]]
    return (
        f"{name:13s} ok {len(served)}/{count} silent openings {sum(result['is_silent_opening'] for result in served)} "
        f"samples {dict(collections.Counter(result['samples'] for result in served).most_common(4))} | "
        f"streams {len(rows)} first decode widths {dict(collections.Counter(row['first_launch_frames'] for row in rows))} "
        f"engine done before first dispatch {len(late_rows)}, their widths "
        f"{dict(collections.Counter(row['first_launch_frames'] for row in late_rows))}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=26000)
    parser.add_argument("--count", type=int, default=48)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}"
    server = serve(args.tree, MODELS["customvoice"][0], args.card, args.port, args.out / "serve.log", {})
    try:
        wait_healthy(args.port, server)
        for name, extra, is_short in BURSTS:
            rounds = args.rounds if name in REPEATED_BURSTS else 1
            for index in range(rounds):
                line = run_burst(url, args.port, name, extra, is_short, args.count, args.out / f"{name}_r{index}")
                print(f"{time.strftime('%T')} {line}", flush=True)
        print(f"{time.strftime('%T')} codec counters {json.dumps(codec_state(url))}", flush=True)
    finally:
        stop(server)


if __name__ == "__main__":
    main()
