"""Vocoder-side streaming timings from one request event recorder directory.

Usage: python vocoder_event_timings.py <event_dir>
"""

import bisect
import glob
import json
import math
import os
import sys
from collections import defaultdict


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    return ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1]


def stats(name: str, values: list[float]) -> None:
    if not values:
        print(f"  {name:<46} n=0")
        return
    mean = sum(values) / len(values)
    print(
        f"  {name:<46} n={len(values):<6} mean={mean:8.2f} p50={pct(values, 50):8.2f} "
        f"p90={pct(values, 90):8.2f} p95={pct(values, 95):8.2f} "
        f"p99={pct(values, 99):8.2f} max={max(values):8.2f} ms"
    )


def main() -> None:
    event_dir = sys.argv[1]
    rows = []
    for path in glob.glob(os.path.join(event_dir, "events_*.jsonl")):
        with open(path) as handle:
            rows.extend(json.loads(line) for line in handle)
    rows.sort(key=lambda row: row["timestamp_ns"])

    def select(stage: str, event_name: str) -> list[dict]:
        return [
            r for r in rows if r["stage"] == stage and r["event_name"] == event_name
        ]

    vocoder_recv = select("vocoder", "stage_stream_chunk_received")
    vocoder_sent = select("vocoder", "stage_stream_chunk_sent")
    vocoder_first = select("vocoder", "stage_first_stream_chunk_sent")
    engine_sent = select("tts_engine", "stage_stream_chunk_sent")
    print(
        f"vocoder chunk_received={len(vocoder_recv)} chunk_sent={len(vocoder_sent)} "
        f"first_sent={len(vocoder_first)} tts_engine chunk_sent={len(engine_sent)}"
    )

    recv_by_request: dict[str, dict[int, int]] = defaultdict(dict)
    for r in vocoder_recv:
        recv_by_request[r["request_id"]][r["metadata"]["chunk_id"]] = r["timestamp_ns"]
    sent_by_request: dict[str, list[int]] = defaultdict(list)
    for r in vocoder_sent:
        sent_by_request[r["request_id"]].append(r["timestamp_ns"])
    first_sent = {}
    for r in vocoder_first:
        first_sent.setdefault(r["request_id"], r["timestamp_ns"])

    print("\nA. first audio in the vocoder process")
    after_recv0, after_recv1, recv1_after_recv0 = [], [], []
    recv1_wait: dict[str, tuple[int, float]] = {}
    for request_id, sent_ns in first_sent.items():
        received = recv_by_request[request_id]
        after_recv0.append((sent_ns - received[0]) / 1e6)
        if 1 in received:
            wait_ms = (sent_ns - received[1]) / 1e6
            after_recv1.append(wait_ms)
            recv1_wait[request_id] = (received[1], wait_ms)
            recv1_after_recv0.append((received[1] - received[0]) / 1e6)
    stats("first audio sent - chunk 0 received", after_recv0)
    stats("first audio sent - chunk 1 received", after_recv1)
    stats("chunk 1 received - chunk 0 received", recv1_after_recv0)

    print("\nB. later audio chunks: sent - latest earlier receive of the request")
    later: dict[str, list[float]] = defaultdict(list)
    for request_id, sent_list in sent_by_request.items():
        received = sorted(recv_by_request[request_id].values())
        for index, sent_ns in enumerate(sorted(sent_list)[1:], start=2):
            latest = bisect.bisect_right(received, sent_ns) - 1
            if latest >= 0:
                key = "2nd" if index == 2 else "3rd" if index == 3 else "4th+"
                later[key].append((sent_ns - received[latest]) / 1e6)
    for key in ("2nd", "3rd", "4th+"):
        stats(key, later[key])

    print("\nC. vocoder backlog: each receive -> next audio sent of any request")
    start_ns = rows[0]["timestamp_ns"]
    sent_ns_all = [r["timestamp_ns"] for r in vocoder_sent]
    recv_ns_all = [r["timestamp_ns"] for r in vocoder_recv]
    backlog = []
    for r in vocoder_recv:
        index = bisect.bisect_left(sent_ns_all, r["timestamp_ns"])
        if index < len(sent_ns_all):
            arrived = bisect.bisect_right(
                recv_ns_all, sent_ns_all[index]
            ) - bisect.bisect_right(recv_ns_all, r["timestamp_ns"])
            backlog.append(
                (
                    (sent_ns_all[index] - r["timestamp_ns"]) / 1e6,
                    r["timestamp_ns"],
                    arrived,
                )
            )
    stats("receive -> next sent", [b[0] for b in backlog])
    print("  top 10 stalls: stall_ms at_ms receives_during_stall")
    for stall_ms, at_ns, arrived in sorted(backlog, reverse=True)[:10]:
        print(f"    {stall_ms:8.2f} {(at_ns - start_ns) / 1e6:10.1f} {arrived:5d}")

    print(
        "\nD. first audio sent - chunk 1 received, by other first-audio requests in flight"
    )
    first_recv = {q: min(d.values()) for q, d in recv_by_request.items() if d}
    first_sent_any = {q: min(l) for q, l in sent_by_request.items()}
    bins: dict[str, list[float]] = defaultdict(list)
    for request_id, (recv1_ns, wait_ms) in recv1_wait.items():
        in_flight = sum(
            1
            for other, other_recv in first_recv.items()
            if other != request_id
            and other_recv <= recv1_ns
            and first_sent_any.get(other, math.inf) > recv1_ns
        )
        bins[str(in_flight) if in_flight < 4 else "4+"].append(wait_ms)
    for key in ("0", "1", "2", "3", "4+"):
        stats(f"{key} in flight", bins[key])

    print("\nE. transport: vocoder receive - tts_engine sent, matched by chunk")
    engine_ns = {
        (r["request_id"], r["metadata"]["chunk_id"]): r["timestamp_ns"]
        for r in engine_sent
    }
    transport = [
        (t - engine_ns[(q, c)]) / 1e6
        for q, received in recv_by_request.items()
        for c, t in received.items()
        if (q, c) in engine_ns
    ]
    stats("vocoder receive - tts_engine sent", transport)

    print(
        "\nF. ten equal windows: receives, sends, first audio sent - chunk 1 received"
    )
    first_ns, last_ns = recv_ns_all[0], max(recv_ns_all[-1], sent_ns_all[-1])
    width = (last_ns - first_ns) / 10
    for window in range(10):
        low = first_ns + window * width
        high = first_ns + (window + 1) * width if window < 9 else last_ns + 1
        waits = [w for t, w in recv1_wait.values() if low <= t < high]
        print(
            f"  {window} recv={sum(low <= t < high for t in recv_ns_all):5d} "
            f"sent={sum(low <= t < high for t in sent_ns_all):5d} n={len(waits):4d} "
            f"p50={pct(waits, 50):8.2f} p95={pct(waits, 95):8.2f} ms"
        )


if __name__ == "__main__":
    main()
