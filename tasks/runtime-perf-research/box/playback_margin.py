"""Chunk gaps against the audio the listener holds, per pass of a streamed speech cell recorded by omni_seedtts_chunks.py.

usage: python3 playback_margin.py <boot dir> [<boot dir> ...]

Per chunk after the first, the margin is the audio still queued for playback when the chunk arrives: the playback
deadline (first chunk's arrival plus every queued chunk's length, the walk compute_max_playback_underrun_s makes) minus
the arrival. A negative margin is an underrun. Also prints streams in flight by Little's law.
"""

import glob
import json
import statistics
import sys


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def main() -> None:
    print(
        f"{'pass':22s} {'QPS':>6s} {'1st audio':>9s} {'stream':>6s} {'gap mean':>8s} {'p99':>5s} {'max':>5s} "
        f"{'stalls':>6s} {'margin min':>10s} {'p1':>5s} {'p5':>5s} {'chunk s':>7s}"
    )
    for boot in sys.argv[1:]:
        for path in sorted(glob.glob(f"{boot}/c*_*/bench/speed_results.json")):
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            done = [r for r in data["per_request"] if r["is_success"]]
            gaps, margins, lengths, stalls = [], [], [], 0
            for request in done:
                durations = request.get("chunk_audio_duration_s") or []
                arrivals = [0.0]
                for gap in request["inter_chunk_s"] or []:
                    arrivals.append(arrivals[-1] + gap)
                    gaps.append(gap)
                lengths += durations
                if not durations:
                    continue
                else:
                    pass
                deadline = durations[0]
                for arrival, duration in zip(arrivals[1:], durations[1:]):
                    margins.append(deadline - arrival)
                    deadline = max(deadline, arrival) + duration
                stalls += (request.get("max_playback_underrun_s") or 0) > 0
            summary = data["summary"]
            qps = summary["throughput_qps"]
            streaming = qps * statistics.fmean(
                r["latency_s"] - r["audio_ttfp_s"] for r in done
            )
            label = path.split("/bench")[0]
            print(
                f"{label:22s} {qps:6.2f} {summary['audio_ttfp_mean_s'] * 1000:9.0f} {streaming:6.1f} "
                f"{statistics.fmean(gaps) * 1000:8.0f} {pct(gaps, 0.99) * 1000:5.0f} {max(gaps) * 1000:5.0f} "
                f"{stalls:6d} {min(margins) * 1000:10.0f} {pct(margins, 0.01) * 1000:5.0f} "
                f"{pct(margins, 0.05) * 1000:5.0f} {statistics.fmean(lengths):7.3f}"
            )


if __name__ == "__main__":
    main()
