"""First chunk anatomy from the event recorder files of one or more arms.

Usage: python first_chunk_anatomy.py <label>=<events_dir>[:<speed_results.json>] ...

For every arm: the per segment p50 / p95 from admission to the coordinator's receipt of
the first audio chunk, the first audio segment binned by how many other requests were
between their first code receipt and their first audio send when this request's first
code arrived, the number of code chunks received before first audio, and the client
TTFC when a speed_results.json is given.
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


def load(events_dir):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in glob.glob(f"{events_dir}/*.jsonl"):
        with open(path) as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                per[event["request_id"]][(event["stage"], event["event_name"])].append(
                    event["timestamp_ns"]
                )
    return per


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


def fmt(values):
    if not values:
        return "-"
    return f"{statistics.median(values):7.1f} / {pct(values, 0.95):7.1f}"


def segments(per):
    print("  segment (p50 / p95 ms)")
    for a, b in zip(CHAIN, CHAIN[1:]):
        xs = [
            (min(x[b]) - min(x[a])) / 1e6 for x in per.values() if x.get(a) and x.get(b)
        ]
        print(f"    {a[0]}.{a[1]} -> {b[0]}.{b[1]:38s} n={len(xs):4d}  {fmt(xs)}")
    total = [
        (min(x[CHAIN[-1]]) - min(x[CHAIN[0]])) / 1e6
        for x in per.values()
        if x.get(CHAIN[0]) and x.get(CHAIN[-1])
    ]
    print(
        f"    admission -> first audio at the coordinator{'':29s} n={len(total):4d}  {fmt(total)}"
    )


def queue_table(per):
    rows = []
    for x in per.values():
        received = x.get(("vocoder", "stage_stream_chunk_received"))
        sent = x.get(("vocoder", "stage_stream_chunk_sent"))
        codes = x.get(("tts_engine", "stage_stream_chunk_sent"), [])
        if not received or not sent:
            continue
        first_code = min(received)
        first_audio = min(sent)
        rows.append(
            (
                first_code,
                first_audio,
                (first_audio - first_code) / 1e6,
                sum(1 for t in codes if t < first_audio),
            )
        )
    rows.sort()
    by_depth = collections.defaultdict(list)
    for first_code, first_audio, segment, _ in rows:
        depth = sum(1 for r in rows if r[0] < first_code < r[1])
        by_depth[min(depth, 6)].append(segment)
    print("  first audio segment by requests ahead in the initial worker")
    for depth in sorted(by_depth):
        label = f"{depth}+" if depth == 6 else f"{depth} "
        print(
            f"    ahead {label}: n={len(by_depth[depth]):4d}  p50 {statistics.median(by_depth[depth]):6.1f} ms"
        )
    chunks = [r[3] for r in rows]
    if chunks:
        print(
            f"  code chunks received before first audio: p50 {statistics.median(chunks):.0f}"
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
        per = load(events_dir)
        print(f"== {label}: {len(per)} requests in {events_dir}")
        if speed:
            client(speed)
        segments(per)
        queue_table(per)


if __name__ == "__main__":
    main()
