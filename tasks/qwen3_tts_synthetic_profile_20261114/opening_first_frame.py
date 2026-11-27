"""Whether each CustomVoice size opens Ryan/English speech with a silent codec frame, over SeedTTS EN.

Per model, one server on card N from TREE. Every SeedTTS EN target text is sent streamed (Ryan, English,
default sampling, PCM) with suppress_bootstrap_silence false, so the first frame's samples arrive as decoded.
Per request it reads the first 1,920 samples against the vocoder's own silence guard (RMS 0.001, peak 0.0032).
Then, one request at a time over the first PAIRED texts, the trim on and off alternate, and the time to first
byte and to first audible sound are read for each. The audible sound is the first point where DC-removed RMS
over 20 ms windows (10 ms hop) reaches -45 dBFS for two windows in a row, timed as a listener hears it: played
from the first byte, so any leading silence plays first. Writes OUT/<model>/frames.jsonl and
OUT/<model>/paired.jsonl.

usage: python3 opening_first_frame.py --tree TREE --card 0 --out DIR [--models 0.6b 1.7b] [--paired 100]
       [--skip-corpus]
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
from pathlib import Path

import httpx
import numpy as np

from seedtts_pair import serve, stop
from served_baseline import wait_healthy

MODELS = {
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}
SEEDTTS_META = "zhaochenyang20/seed-tts-eval-arrow"
SAMPLE_RATE = 24000
FRAME_SAMPLES = 1920
SILENCE_MAX_RMS = 0.001
SILENCE_MAX_PEAK = 0.0032
AUDIBLE_DBFS = -45.0
WINDOW_SAMPLES = 480
HOP_SAMPLES = 240
CONCURRENCY = 16
REQUEST_TIMEOUT_S = 300


def body_for(model: str, text: str, is_trimmed: bool) -> dict:
    return {
        "model": model, "input": text, "voice": "Ryan", "language": "English", "response_format": "pcm",
        "stream": True, "suppress_bootstrap_silence": is_trimmed,
    }


def stream_pcm(url: str, body: dict) -> tuple[np.ndarray, list[tuple[float, int]], int]:
    """The decoded samples and, per received piece, its arrival time and the sample count received by then."""
    start = time.monotonic()
    arrivals: list[tuple[float, int]] = []
    audio = bytearray()
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        with client.stream("POST", f"{url}/v1/audio/speech", json=body) as response:
            if response.status_code != 200:
                return np.zeros(0, dtype=np.float32), [], response.status_code
            else:
                pass
            for piece in response.iter_bytes():
                audio.extend(piece)
                arrivals.append((time.monotonic() - start, len(audio) // 2))
    samples = np.frombuffer(bytes(audio[: len(audio) // 2 * 2]), dtype=np.int16).astype(np.float32) / 32768.0
    return samples, arrivals, 200


def audible_onset(samples: np.ndarray) -> int | None:
    """The first sample of the first two consecutive windows at or above the audible level."""
    previous = False
    for start in range(0, max(0, samples.size - WINDOW_SAMPLES + 1), HOP_SAMPLES):
        window = samples[start : start + WINDOW_SAMPLES]
        window = window - float(window.mean())
        rms = float(np.sqrt(np.mean(window * window)))
        is_audible = rms > 0 and 20.0 * np.log10(rms) >= AUDIBLE_DBFS
        if is_audible and previous:
            return start - HOP_SAMPLES
        else:
            previous = is_audible
    return None


def arrival_of(sample: int, arrivals: list[tuple[float, int]]) -> float | None:
    for seconds, received in arrivals:
        if received > sample:
            return seconds
        else:
            pass
    return None


def first_frame(url: str, model: str, text: str) -> dict:
    samples, _, status = stream_pcm(url, body_for(model, text, False))
    if status != 200 or samples.size == 0:
        return {"status": status}
    else:
        pass
    head = samples[:FRAME_SAMPLES]
    rms = float(np.sqrt(np.mean(head * head)))
    peak = float(np.max(np.abs(head)))
    return {
        "status": 200, "samples": int(samples.size), "head_rms": rms, "head_peak": peak,
        "is_silent": rms <= SILENCE_MAX_RMS and peak <= SILENCE_MAX_PEAK,
    }


def paired(url: str, model: str, text: str, is_trimmed: bool) -> dict:
    samples, arrivals, status = stream_pcm(url, body_for(model, text, is_trimmed))
    if status != 200 or not arrivals:
        return {"status": status, "is_trimmed": is_trimmed}
    else:
        pass
    onset = audible_onset(samples)
    first_byte_ms = arrivals[0][0] * 1000.0
    return {
        "status": 200, "is_trimmed": is_trimmed, "first_byte_ms": first_byte_ms,
        "onset_sample": onset,
        "audible_arrival_ms": None if onset is None else arrival_of(onset, arrivals) * 1000.0,
        "audible_ms": None if onset is None else first_byte_ms + 1000.0 * onset / SAMPLE_RATE,
        "samples": int(samples.size),
    }


def quantiles(values: list[float]) -> str:
    values = sorted(value for value in values if value is not None)
    if not values:
        return "none"
    else:
        pass
    return "/".join(f"{values[min(len(values) - 1, int(q * len(values)))]:.1f}" for q in (0.1, 0.5, 0.9))


def report_first_frames(url: str, name: str, model: str, texts: list[str], out: Path) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        frames = list(pool.map(lambda text: first_frame(url, model, text), texts))
    with (out / "frames.jsonl").open("w") as handle:
        for text, row in zip(texts, frames):
            handle.write(json.dumps({"text": text, **row}) + "\n")
    served = [row for row in frames if row["status"] == 200]
    silent = sum(row["is_silent"] for row in served)
    print(
        f"{time.strftime('%T')} {name} first frame silent {silent}/{len(served)} of {len(texts)} "
        f"({100.0 * silent / max(1, len(served)):.2f} %), head rms p50/p99/max "
        f"{statistics.median(row['head_rms'] for row in served):.6f}/"
        f"{sorted(row['head_rms'] for row in served)[int(0.99 * (len(served) - 1))]:.6f}/"
        f"{max(row['head_rms'] for row in served):.6f}, head peak max {max(row['head_peak'] for row in served):.6f}",
        flush=True,
    )


def run_model(
    tree: Path, name: str, card: int, port: int, paired_count: int, texts: list[str], out: Path, is_corpus_run: bool
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model = MODELS[name]
    url = f"http://127.0.0.1:{port}"
    server = serve(tree, model, card, port, out / "serve.log", {})
    try:
        wait_healthy(port, server)
        if is_corpus_run:
            report_first_frames(url, name, model, texts, out)
        else:
            pass
        rows = []
        for index, text in enumerate(texts[:paired_count]):
            order = (True, False) if index % 2 == 0 else (False, True)
            for is_trimmed in order:
                rows.append({"text": text, **paired(url, model, text, is_trimmed)})
        with (out / "paired.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        for is_trimmed in (False, True):
            arm = [row for row in rows if row["is_trimmed"] is is_trimmed and row["status"] == 200]
            print(
                f"{time.strftime('%T')} {name} one request at a time, trim {'on ' if is_trimmed else 'off'}: "
                f"first byte p10/p50/p90 {quantiles([row['first_byte_ms'] for row in arm])} ms, "
                f"audible when played from the first byte {quantiles([row['audible_ms'] for row in arm])} ms, "
                f"audible sample's arrival {quantiles([row['audible_arrival_ms'] for row in arm])} ms, "
                f"onset sample {quantiles([row['onset_sample'] for row in arm])}, n {len(arm)}",
                flush=True,
            )
    finally:
        stop(server)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=27000)
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["0.6b", "1.7b"])
    parser.add_argument("--paired", type=int, default=100)
    parser.add_argument("--skip-corpus", action="store_true", help="run only the paired one-at-a-time requests")
    args = parser.parse_args()
    sys.path.insert(0, str(args.tree))
    from benchmarks.dataset.seedtts import load_seedtts_samples

    texts = [sample.target_text for sample in load_seedtts_samples(SEEDTTS_META, split="en")]
    print(f"{time.strftime('%T')} SeedTTS EN texts {len(texts)}", flush=True)
    for index, name in enumerate(args.models):
        run_model(
            args.tree, name, args.card, args.port + 400 * index, args.paired, texts, args.out / name,
            not args.skip_corpus,
        )


if __name__ == "__main__":
    main()
