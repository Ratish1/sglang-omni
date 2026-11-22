#!/usr/bin/env python3
"""Client first audio split into the server's part and the rest, per arm, from one cell's client results and its
pipeline events (boots run with events).

usage: python3 first_audio_delivery.py <run dir> --cell c64_events <arm>=<boot>,<boot> [<arm>=<boot>,...]

Server first audio is admission (coordinator request_admission) to code2wav's first stream chunk sent; delivery is the
client's first audio minus it (send to admission, and the chunk's way to the client). Per arm, boots pooled: how many
client requests joined their events by id, then mean, p50, p90 and p99 of client, server and delivery, and over the
arm's slowest 1 % by client first audio the mean of each.
"""

import argparse
import glob
import json
import os
import statistics

ADMISSION = ("coordinator", "request_admission")
FIRST_AUDIO = ("code2wav", "stage_first_stream_chunk_sent")


def quantile(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def boot_rows(run, boot, cell):
    first = {}
    for path in glob.glob(os.path.join(run, boot, cell, "events", "*.jsonl")):
        with open(path, encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                key = (record["stage"], record["event_name"])
                if key in (ADMISSION, FIRST_AUDIO):
                    seen = first.setdefault(record["request_id"], {})
                    seen[key] = min(
                        seen.get(key, record["timestamp_ns"]), record["timestamp_ns"]
                    )
                else:
                    pass
    with open(
        os.path.join(run, boot, cell, "bench", "speed_results.json"), encoding="utf-8"
    ) as source:
        requests = json.load(source)["per_request"]
    rows, unjoined = [], 0
    for request in requests:
        client = request.get("audio_ttfp_s")
        stamps = first.get(request.get("id"), {})
        if client is None:
            continue
        elif ADMISSION in stamps and FIRST_AUDIO in stamps:
            server = (stamps[FIRST_AUDIO] - stamps[ADMISSION]) / 1e9
            rows.append((client * 1e3, server * 1e3, (client - server) * 1e3))
        else:
            unjoined += 1
    return rows, unjoined, len(first)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--cell", default="c64_events")
    parser.add_argument("arms", nargs="+")
    args = parser.parse_args()
    for arm in args.arms:
        name, _, boots = arm.partition("=")
        rows, unjoined, event_requests = [], 0, 0
        for boot in boots.split(","):
            boot_joined, boot_unjoined, boot_events = boot_rows(
                args.run, boot, args.cell
            )
            rows += boot_joined
            unjoined += boot_unjoined
            event_requests += boot_events
        print(
            f"{name}: {len(rows)} joined, {unjoined} unjoined, {event_requests} requests with events"
        )
        if not rows:
            continue
        else:
            pass
        for index, label in enumerate(("client", "server", "delivery")):
            values = [row[index] for row in rows]
            print(
                f"  {label:9s} mean {statistics.fmean(values):7.1f} p50 {quantile(values, 0.5):7.1f} "
                f"p90 {quantile(values, 0.9):7.1f} p99 {quantile(values, 0.99):7.1f} ms"
            )
        slowest = sorted(rows, key=lambda row: -row[0])[: max(1, len(rows) // 100)]
        print(
            f"  slowest 1 % by client ({len(slowest)}): client {statistics.fmean(r[0] for r in slowest):.1f}, "
            f"server {statistics.fmean(r[1] for r in slowest):.1f}, delivery {statistics.fmean(r[2] for r in slowest):.1f} ms"
        )


if __name__ == "__main__":
    main()
