"""Device idle between two talker steps and the host path that causes it, per pair of consecutive decode steps.

usage: python3 step_gaps.py <formal trace.json.gz>
For decode step N (an omni.step span on the scheduler thread) and the decode step after it:
  wait end       the end of step N's event synchronize (finalize waiting for the sampled ids)
  host path      wait end -> the start of step N+1's first cudaGraphLaunch (the talker)
  predictor left step N's predictor graph end (its second launch) minus wait end
  idle           time on the talker's stream between step N's predictor end and step N+1's talker start with no
                 device work on that stream (eager kernels launched in between count as busy)
  launch lag     step N+1's talker start on the device minus its launch call's end
Medians and means per step N's batch size. A step without exactly one synchronize and two graph launches is skipped
and counted. Check: idle should be about max(0, host path + launch call + launch lag - predictor left).
"""

import bisect
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict

STEP = re.compile(r"omni\.step (\w+) (\w+) fwd=(\d+) bs=(\d+)")
SYNC = ("cudaEventSynchronize", "cudaStreamSynchronize")


def main() -> None:
    with gzip.open(sys.argv[1], "rt", encoding="utf-8") as handle:
        events = json.load(handle)["traceEvents"]
    steps = sorted(
        (e for e in events if e.get("cat") == "user_annotation" and STEP.match(str(e.get("name")))), key=lambda e: e["ts"]
    )
    thread = steps[0]["tid"]
    runtime = sorted(
        (e for e in events if e.get("cat") == "cuda_runtime" and e["tid"] == thread), key=lambda e: e["ts"]
    )
    runtime_starts = [e["ts"] for e in runtime]
    device_by_corr: dict[int, list[dict]] = defaultdict(list)
    talker_stream_spans: list[tuple[float, float]] = []
    for e in events:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            device_by_corr[e.get("args", {}).get("correlation")].append(e)
        else:
            pass

    def calls_in(step: dict) -> list[dict]:
        return runtime[bisect.bisect_left(runtime_starts, step["ts"]): bisect.bisect_right(runtime_starts, step["ts"] + step["dur"])]

    parsed = []
    skipped = 0
    for step in steps:
        calls = calls_in(step)
        graphs = [c for c in calls if str(c["name"]).startswith("cudaGraphLaunch")]
        syncs = [c for c in calls if c["name"] in SYNC]
        if len(graphs) != 2 or len(syncs) != 1:
            parsed.append(None)
            skipped += 1
            continue
        else:
            pass
        talker = device_by_corr.get(graphs[0].get("args", {}).get("correlation"), [])
        predictor = device_by_corr.get(graphs[1].get("args", {}).get("correlation"), [])
        if not talker or not predictor:
            parsed.append(None)
            skipped += 1
            continue
        else:
            pass
        stream = talker[0].get("args", {}).get("stream")
        parsed.append({
            "kind": STEP.match(step["name"]).group(1),
            "bs": int(STEP.match(step["name"]).group(4)),
            "stream": stream,
            "wait_end": syncs[0]["ts"] + syncs[0]["dur"],
            "talker_call": graphs[0],
            "talker_start": min(k["ts"] for k in talker),
            "predictor_end": max(k["ts"] + k["dur"] for k in predictor),
        })
    stream_ids = {p["stream"] for p in parsed if p}
    for e in events:
        if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("args", {}).get("stream") in stream_ids:
            talker_stream_spans.append((e["ts"], e["ts"] + e["dur"]))
        else:
            pass
    talker_stream_spans.sort()
    span_starts = [s for s, _ in talker_stream_spans]

    def busy_us(start: float, end: float) -> float:
        total, cursor = 0.0, start
        for s, e in talker_stream_spans[max(0, bisect.bisect_left(span_starts, start) - 64): bisect.bisect_right(span_starts, end)]:
            s, e = max(s, cursor), min(e, end)
            if e > s:
                total += e - s
                cursor = e
            else:
                pass
        return total

    rows: dict[int, list[dict[str, float]]] = defaultdict(list)
    for current, following in zip(parsed, parsed[1:]):
        if current is None or following is None or current["kind"] != "decode" or following["kind"] != "decode":
            continue
        else:
            pass
        gap_start, gap_end = current["predictor_end"], following["talker_start"]
        call = following["talker_call"]
        rows[current["bs"]].append({
            "host path ms": (call["ts"] - current["wait_end"]) / 1e3,
            "predictor left ms": (current["predictor_end"] - current["wait_end"]) / 1e3,
            "idle ms": max(0.0, (gap_end - gap_start) - busy_us(gap_start, gap_end)) / 1e3,
            "launch call ms": call["dur"] / 1e3,
            "launch lag ms": (following["talker_start"] - (call["ts"] + call["dur"])) / 1e3,
        })
    print(f"scheduler thread {thread}: {len(steps)} steps, {skipped} skipped (not one sync and two graph launches)")
    columns = ("host path ms", "predictor left ms", "idle ms", "launch call ms", "launch lag ms")
    print("| bs | pairs | " + " | ".join(f"{c} (median / mean)" for c in columns) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in columns) + " |")
    for batch, items in sorted(rows.items()):
        if len(items) < 5:
            continue
        else:
            pass
        cells = [f"{statistics.median(i[c] for i in items):.3f} / {statistics.fmean(i[c] for i in items):.3f}" for c in columns]
        print(f"| {batch} | {len(items)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
