#!/usr/bin/env python3
"""Speech load each mixed cell actually served beside the video client, and the video cost per answer token.

A cell's speech client caps requests in flight; when the cap binds, the faster arm serves more speech and the
video client's latency moves with that load, so video is read against the speech each arm served.
"""

import argparse
import glob
import json
import os
import statistics


def speech_row(cell_dir: str) -> dict:
    paths = glob.glob(
        os.path.join(cell_dir, "c*_events", "bench", "speed_results.json")
    )
    if len(paths) != 1:
        return {}
    path = paths[0]
    with open(path) as handle:
        data = json.load(handle)
    summary = data["summary"]
    requests = data["per_request"]
    return {
        "speech_qps": summary["throughput_qps"],
        "slot_waits": summary.get("client_slot_waits"),
        "requests": summary["total_requests"],
        "lateness_p50": summary.get("dispatch_lateness_p50_s"),
        "latency_mean": summary["latency_mean_s"],
        "first_audio_mean": summary.get("audio_ttfp_mean_s"),
        "tokens_mean": statistics.mean(
            r["completion_tokens"] for r in requests if r["is_success"]
        ),
    }


def video_row(cell_dir: str) -> dict:
    passes = sorted(
        glob.glob(os.path.join(cell_dir, "video_pass*", "videomme_results.json"))
    )
    full = []
    for path in passes:
        with open(path) as handle:
            data = json.load(handle)
        samples = [
            s
            for s in data["per_sample"]
            if s.get("is_success", True) and s.get("latency_s")
        ]
        if len(samples) == data["speed"]["total_requests"] and len(samples) >= 50:
            full.append((os.path.basename(os.path.dirname(path)), samples))
    later = [samples for name, samples in full if name != "video_pass0"]
    if not later:
        return {}
    flat = [s for samples in later for s in samples]
    latency = sum(s["latency_s"] for s in flat)
    tokens = sum(s["completion_tokens"] for s in flat)
    return {
        "video_passes": len(later),
        "video_latency_mean": latency / len(flat),
        "video_tokens_mean": tokens / len(flat),
        "video_s_per_100_tokens": 100 * latency / tokens,
        "pass_latency_means": [
            round(statistics.mean(s["latency_s"] for s in samples), 2)
            for samples in later
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("cells", nargs="+")
    args = parser.parse_args()
    header = (
        f"{'cell':8} {'speech/s':>8} {'waits':>9} {'late p50':>8} {'fa mean':>7} {'v passes':>8} "
        f"{'v lat':>6} {'v toks':>6} {'v s/100tok':>10}  pass means"
    )
    print(header)
    for cell in args.cells:
        speech = speech_row(os.path.join(args.root, cell))
        video = video_row(os.path.join(args.root, cell))
        if not speech or not video:
            print(f"{cell:8} missing")
            continue
        waits = f"{speech['slot_waits']}/{speech['requests']}"
        print(
            f"{cell:8} {speech['speech_qps']:8.2f} {waits:>9} {speech['lateness_p50'] or 0:8.2f} "
            f"{speech['first_audio_mean']:7.3f} {video['video_passes']:8d} {video['video_latency_mean']:6.2f} "
            f"{video['video_tokens_mean']:6.1f} {video['video_s_per_100_tokens']:10.3f}  {video['pass_latency_means']}"
        )


if __name__ == "__main__":
    main()
