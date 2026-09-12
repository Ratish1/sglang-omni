"""First chunk anatomy from the event recorder files of one or more arms.

Usage: python first_chunk_anatomy.py <label>=<events_dir>[:<speed_results.json>] ...

For every arm: the per segment p50 / p95 from admission to the coordinator's receipt of
the first audio chunk, the first audio segment measured from the first generated frame
and binned by how many other bootstraps were in flight in the initial worker when that
frame arrived, the number of generated code chunks received before first audio, the
prefill duration binned by how many bootstraps overlapped it, the talker's code chunk
cadence, and the client TTFC when a speed_results.json is given.

A stream prefix (reference codes sent before prefill) is detected per request as a code
chunk received before the request's prefill ended; it counts as initial worker work but
not as a generated chunk, and the first generated frame is the first chunk after it.
"""

import collections
import glob
import json
import statistics
import sys

CHAIN = [
    ("coordinator", "request_admission"),
    ("preprocessing", "stage_dispatch"),
    ("preprocessing", "stage_complete"),
    ("tts_engine", "stage_input_received"),
    ("tts_engine", "scheduler_request_build_start"),
    ("tts_engine", "scheduler_request_build_end"),
    ("tts_engine", "scheduler_queue_enter"),
    ("tts_engine", "scheduler_prefill_start"),
    ("tts_engine", "scheduler_prefill_end"),
    ("tts_engine", "scheduler_first_emit"),
    ("tts_engine", "stage_stream_chunk_sent"),
    ("vocoder", "stage_stream_chunk_received"),
    ("vocoder", "stage_stream_chunk_sent"),
    ("coordinator", "stage_stream_chunk_received"),
]
PREFILL_END = ("tts_engine", "scheduler_prefill_end")
CODES_SENT = ("tts_engine", "stage_stream_chunk_sent")
CODES_RECEIVED = ("vocoder", "stage_stream_chunk_received")
AUDIO_SENT = ("vocoder", "stage_stream_chunk_sent")


def load(events_dir):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    meta = {}
    for path in glob.glob(f"{events_dir}/*.jsonl"):
        with open(path) as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (event["stage"], event["event_name"])
                per[event["request_id"]][key].append(event["timestamp_ns"])
                if key == PREFILL_END:
                    meta[event["request_id"]] = event.get("metadata", {})
    for request in per.values():
        for values in request.values():
            values.sort()
    return per, meta


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


def fmt(values):
    if not values:
        return "-"
    return f"{statistics.median(values):7.1f} / {pct(values, 0.95):7.1f}"


def first_generated(request):
    """Timestamps of the first generated code chunk sent and received.

    Chunks received before the prefill ended are the stream prefix.
    """
    prefill_end = request.get(PREFILL_END)
    sent = request.get(CODES_SENT, [])
    received = request.get(CODES_RECEIVED, [])
    if not sent or not received:
        return None, None
    if not prefill_end:
        return sent[0], received[0]
    cutoff = prefill_end[0]
    sent_after = [t for t in sent if t >= cutoff]
    received_after = [t for t in received if t >= cutoff]
    if not sent_after or not received_after:
        return None, None
    return sent_after[0], received_after[0]


def event_time(request, key):
    if key == CODES_SENT:
        return first_generated(request)[0]
    if key == CODES_RECEIVED:
        return first_generated(request)[1]
    values = request.get(key)
    return values[0] if values else None


def segments(per):
    print("  segment (p50 / p95 ms)")
    for a, b in zip(CHAIN, CHAIN[1:]):
        xs = []
        for request in per.values():
            ta, tb = event_time(request, a), event_time(request, b)
            if ta is not None and tb is not None:
                xs.append((tb - ta) / 1e6)
        print(f"    {a[0]}.{a[1]} -> {b[0]}.{b[1]:38s} n={len(xs):4d}  {fmt(xs)}")
    total = []
    for request in per.values():
        ta, tb = event_time(request, CHAIN[0]), event_time(request, CHAIN[-1])
        if ta is not None and tb is not None:
            total.append((tb - ta) / 1e6)
    print(
        f"    admission -> first audio at the coordinator{'':29s} n={len(total):4d}  {fmt(total)}"
    )


def bootstrap_rows(per):
    """One row per request with first audio: worker job start, first generated
    frame receipt, first audio send, and the generated chunks before it."""
    rows = []
    for request in per.values():
        received = request.get(CODES_RECEIVED)
        audio = request.get(AUDIO_SENT)
        _, frame1 = first_generated(request)
        if not received or not audio or frame1 is None:
            continue
        first_audio = audio[0]
        prefix = sum(1 for t in received if t < frame1)
        rows.append(
            {
                "job": received[0],
                "frame1": frame1,
                "audio": first_audio,
                "segment": (first_audio - frame1) / 1e6,
                "prefix": prefix > 0,
                "generated_before": sum(
                    1 for t in received if frame1 <= t < first_audio
                ),
            }
        )
    rows.sort(key=lambda r: r["job"])
    return rows


def queue_table(per):
    rows = bootstrap_rows(per)
    if not rows:
        return
    prefixed = sum(1 for r in rows if r["prefix"])
    print(
        f"  first generated frame received -> first audio sent: n={len(rows)}  "
        f"p50 {statistics.median([r['segment'] for r in rows]):.1f} ms  "
        f"mean {statistics.mean([r['segment'] for r in rows]):.1f}  "
        f"p95 {pct([r['segment'] for r in rows], 0.95):.1f}"
        + (f"  ({prefixed} requests carried a stream prefix)" if prefixed else "")
    )
    by_depth = collections.defaultdict(list)
    chunks_by_depth = collections.defaultdict(list)
    for row in rows:
        depth = sum(
            1
            for other in rows
            if other is not row and other["job"] < row["frame1"] < other["audio"]
        )
        by_depth[min(depth, 6)].append(row["segment"])
        chunks_by_depth[min(depth, 6)].append(row["generated_before"])
    print("  first audio segment by other bootstraps in flight when the frame arrived")
    for depth in sorted(by_depth):
        label = f"{depth}+" if depth == 6 else f"{depth} "
        print(
            f"    ahead {label}: n={len(by_depth[depth]):4d}  p50 {statistics.median(by_depth[depth]):6.1f} ms"
            f"  generated chunks before audio p50 {statistics.median(chunks_by_depth[depth]):.0f}"
        )
    print(
        f"  generated code chunks received before first audio: p50 "
        f"{statistics.median([r['generated_before'] for r in rows]):.0f}"
    )
    span = rows[-1]["audio"] - rows[0]["job"]
    if span > 0:
        occupancy = sum(r["audio"] - r["job"] for r in rows) / span
        print(
            f"  bootstrap intervals (job start -> first audio) summed over the window: "
            f"{occupancy:.2f} x the window, mean {statistics.mean([(r['audio'] - r['job']) / 1e6 for r in rows]):.1f} ms per request"
        )


def prefill_table(per, meta):
    rows = bootstrap_rows(per)
    by_overlap = collections.defaultdict(list)
    for request_id, request in per.items():
        start = request.get(("tts_engine", "scheduler_prefill_start"))
        end = request.get(PREFILL_END)
        if not start or not end:
            continue
        if meta.get(request_id, {}).get("batch_size") != 1:
            continue
        overlap = sum(1 for r in rows if r["job"] < end[0] and r["audio"] > start[0])
        by_overlap[min(overlap, 3)].append((end[0] - start[0]) / 1e6)
    if not by_overlap:
        return
    print("  prefill duration at batch size 1 by bootstraps overlapping it")
    for overlap in sorted(by_overlap):
        label = f"{overlap}+" if overlap == 3 else f"{overlap} "
        values = by_overlap[overlap]
        print(
            f"    overlap {label}: n={len(values):4d}  p50 {statistics.median(values):6.1f} ms  mean {statistics.mean(values):6.1f}"
        )


def cadence(per):
    gaps = []
    for request in per.values():
        sent = request.get(CODES_SENT, [])
        cutoff = request.get(PREFILL_END, [0])[0]
        sent = [t for t in sent if t >= cutoff]
        if len(sent) > 3:
            gaps.extend((b - a) / 1e6 for a, b in zip(sent[1:], sent[2:]))
    if gaps:
        print(
            f"  talker code chunk cadence: n={len(gaps)}  p50 {statistics.median(gaps):.1f} ms"
            f"  mean {statistics.mean(gaps):.1f}  p95 {pct(gaps, 0.95):.1f}"
        )


def client(path):
    with open(path) as handle:
        summary = json.load(handle)["summary"]
    with open(path) as handle:
        per_request = json.load(handle)["per_request"]
    longest = max(r.get("audio_duration_s", 0) for r in per_request)
    print(
        f"  client: req/s {summary.get('throughput_qps')}  TTFC mean {summary.get('audio_ttfp_mean_s')} s"
        f"  p99 {summary.get('audio_ttfp_p99_s')} s  inter chunk mean {summary.get('inter_chunk_mean_s')} s"
        f"  latency mean {summary.get('latency_mean_s')} s  longest audio {longest:.1f} s"
        + ("  RUNAWAY, pass not quotable" if longest > 100 else "")
    )


def main():
    for arg in sys.argv[1:]:
        label, _, rest = arg.partition("=")
        events_dir, _, speed = rest.partition(":")
        per, meta = load(events_dir)
        print(f"== {label}: {len(per)} requests in {events_dir}")
        if speed:
            client(speed)
        segments(per)
        queue_table(per)
        prefill_table(per, meta)
        cadence(per)


if __name__ == "__main__":
    main()
