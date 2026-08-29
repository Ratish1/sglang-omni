# SPDX-License-Identifier: Apache-2.0
"""Per-request stage timeline from a request profiler event directory.

Usage:
    python stage_events.py EVENT_DIR [--sort total|queue] [--json OUT]

Reads the events_<stage>_<pid>.jsonl files written after POST
/start_request_profile and prints, per request, the time from the first event
to preprocess_start (the preprocessing queue), the preprocess duration, the
encoder durations, and the time to terminal_response, then a summary. The
stage breakdown of python -m sglang_omni.profiler covers the same events per
stage, this view keeps them per request.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

from sglang_omni.profiler.views import reconstruct_timelines


def _first(events, name, stage=None):
    for ev in events:
        if ev.get("event_name") == name and (stage is None or ev.get("stage") == stage):
            return ev["timestamp_ns"]
    return None


def _durations(events, opener, closer):
    out = {}
    open_at = {}
    for ev in events:
        key = ev.get("stage")
        if ev.get("event_name") == opener:
            open_at[key] = ev["timestamp_ns"]
        elif ev.get("event_name") == closer and key in open_at:
            out[key] = (ev["timestamp_ns"] - open_at.pop(key)) / 1e9
    return out


def rows_for(event_dir: str) -> list[dict]:
    rows = []
    for rid, tl in reconstruct_timelines(event_dir).items():
        evs = tl.events
        t0 = evs[0]["timestamp_ns"]
        pre_start = _first(evs, "preprocess_start")
        pre_end = _first(evs, "preprocess_end")
        terminal = _first(evs, "terminal_response")
        encoders = _durations(evs, "encoder_start", "encoder_end")
        builds = _durations(
            evs, "scheduler_request_build_start", "scheduler_request_build_end"
        )
        rows.append(
            {
                "request_id": rid,
                "t0_ns": t0,
                "queue_s": (pre_start - t0) / 1e9 if pre_start else None,
                "preprocess_s": (
                    (pre_end - pre_start) / 1e9 if pre_start and pre_end else None
                ),
                "encoders_s": {k: round(v, 3) for k, v in encoders.items()},
                "request_build_s": {k: round(v, 3) for k, v in builds.items()},
                "total_s": (terminal - t0) / 1e9 if terminal else tl.total_ms / 1e3,
                "events": len(evs),
                "first_event": evs[0].get("event_name"),
            }
        )
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("event_dir")
    p.add_argument("--sort", choices=("total", "queue", "t0"), default="t0")
    p.add_argument("--json", default=None)
    args = p.parse_args(argv)
    rows = rows_for(args.event_dir)
    if not rows:
        print("no events found", file=sys.stderr)
        return 1
    t_min = min(r["t0_ns"] for r in rows)
    for r in rows:
        r["submit_rel_s"] = (r["t0_ns"] - t_min) / 1e9
    key = {"total": "total_s", "queue": "queue_s", "t0": "t0_ns"}[args.sort]
    rows.sort(key=lambda r: (r[key] is None, r[key]))
    print(
        "request | submit(s) | queue(s) | preprocess(s) | encoders | request_build | total(s) | events | first_event"
    )
    for r in rows:
        q = f"{r['queue_s']:.2f}" if r["queue_s"] is not None else "-"
        pp = f"{r['preprocess_s']:.2f}" if r["preprocess_s"] is not None else "-"
        print(
            f"{r['request_id'][:8]} | {r['submit_rel_s']:6.2f} | {q:>6} | {pp:>6} | "
            f"{r['encoders_s']} | {r['request_build_s']} | {r['total_s']:.2f} | {r['events']} | {r['first_event']}"
        )
    queues = [r["queue_s"] for r in rows if r["queue_s"] is not None]
    pres = [r["preprocess_s"] for r in rows if r["preprocess_s"] is not None]
    totals = [r["total_s"] for r in rows]
    print(
        f"\nrequests={len(rows)} queue median={statistics.median(queues):.2f}s max={max(queues):.2f}s"
        if queues
        else f"\nrequests={len(rows)} (no preprocess_start events)"
    )
    if pres:
        print(
            f"preprocess median={statistics.median(pres):.3f}s max={max(pres):.3f}s sum={sum(pres):.1f}s"
        )
    print(f"total median={statistics.median(totals):.2f}s max={max(totals):.2f}s")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
