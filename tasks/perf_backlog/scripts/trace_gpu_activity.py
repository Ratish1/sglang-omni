"""GR active and SM level proxies from a kineto trace.

Usage: python trace_gpu_activity.py <trace.json.gz>

GR active proxy: fraction of the window with at least one kernel running on any stream.
SM proxies: kernel duration weighted mean of est. achieved occupancy and blocks per SM,
and the duration weighted mean number of kernels running concurrently.
"""

import sys

from trace_streams import parse, raw_events


def main():
    path = sys.argv[1]
    intervals = []
    occ_w = blocks_w = dur_sum = 0.0
    per_stream = {}
    for text in raw_events(path):
        if '"cat": "kernel"' not in text:
            continue
        ev = parse(text)
        if ev is None:
            continue
        ts, dur = ev["ts"], ev.get("dur", 0.0)
        args = ev.get("args") or {}
        intervals.append((ts, ts + dur))
        occ = args.get("est. achieved occupancy %")
        bps = args.get("blocks per SM")
        if occ is not None:
            occ_w += occ * dur
        if bps is not None:
            blocks_w += min(bps, 8.0) * dur
        dur_sum += dur
        s = args.get("stream")
        per_stream[s] = per_stream.get(s, 0.0) + dur
    intervals.sort()
    window = intervals[-1][1] - intervals[0][0]
    union = 0.0
    concurrency_w = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s > cur_e:
            union += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    union += cur_e - cur_s
    print(f"{path}")
    print(f"  window {window/1e3:.0f} ms, kernels {len(intervals)}")
    print(f"  GR active proxy (any kernel resident): {100*union/window:.1f} %")
    print(
        f"  summed kernel time / window (mean concurrent kernels): {dur_sum/window:.2f}"
    )
    print(f"  duration weighted est. achieved occupancy: {occ_w/dur_sum:.1f} %")
    print(f"  duration weighted blocks per SM (capped at 8): {blocks_w/dur_sum:.2f}")
    for s, d in sorted(per_stream.items(), key=lambda kv: -kv[1])[:4]:
        print(f"  stream {s}: busy {100*d/window:.1f} % of window")


if __name__ == "__main__":
    main()
