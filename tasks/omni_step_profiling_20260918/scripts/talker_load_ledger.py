"""What the engine thread does between entering a talker forward and launching its graph, per step mode, from one
pipeline_nvtx nsys report: the wall before and after the graph launch, the launches before it, and those launches' kernels.

usage: python3 talker_load_ledger.py serve.sqlite --bench-log bench.log
"""

import argparse
import bisect
import collections
import sqlite3
import statistics

from nsys_metrics import bench_window
from pipeline_census import PUSH_POP

ENGINE_THREAD = "scheduler-tts_engine"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--bench-log", required=True)
    args = parser.parse_args()
    db = sqlite3.connect(args.sqlite)
    strings = dict(db.execute("select id, value from StringIds"))
    t0, t1 = bench_window(db, args.bench_log)
    talkers: list[tuple[int, int, int]] = []
    steps: list[tuple[int, int, str]] = []
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType = ? and start >= ? and start <= ? order by start",
        (PUSH_POP, t0, t1),
    ):
        if end is None:
            continue
        label = text if text is not None else strings.get(text_id, "")
        if label == "mr.talker":
            talkers.append((start, end, tid))
        elif label.startswith("sched.batch "):
            steps.append((start, end, label.split(" ")[1]))
    engine_tid = talkers[0][2]
    step_starts = [start for start, _, _ in steps]
    calls = list(
        db.execute(
            "select start, end, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ? and start >= ? and start <= ? order by start",
            (engine_tid, t0, t1),
        )
    )
    call_starts = [call[0] for call in calls]
    kernels = {
        corr: (strings.get(name_id, str(name_id)), end - start)
        for corr, name_id, start, end in db.execute(
            "select correlationId, demangledName, start, end from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start <= ?",
            (t0, t1),
        )
    }
    before_ms: dict[str, list[float]] = collections.defaultdict(list)
    after_ms: dict[str, list[float]] = collections.defaultdict(list)
    launches: dict[str, list[int]] = collections.defaultdict(list)
    by_kernel: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    device_ns: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    api_ns: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for start, end, _ in talkers:
        index = bisect.bisect_right(step_starts, start) - 1
        mode = steps[index][2] if index >= 0 and start < steps[index][1] else "other"
        first = bisect.bisect_left(call_starts, start)
        last = bisect.bisect_left(call_starts, end)
        graph_start = None
        count = 0
        for call_start, call_end, corr, name_id in calls[first:last]:
            name = strings.get(name_id, "")
            if name.startswith("cudaGraphLaunch"):
                graph_start = call_start
                break
            elif name.startswith(("cudaLaunchKernel", "cuLaunchKernel")):
                count += 1
                kernel = kernels.get(corr, ("?", 0))
                by_kernel[mode][kernel[0][:88]] += 1
                device_ns[mode][kernel[0][:88]] += kernel[1]
                api_ns[mode][kernel[0][:88]] += call_end - call_start
            else:
                pass
        if graph_start is None:
            continue
        else:
            pass
        before_ms[mode].append((graph_start - start) / 1e6)
        after_ms[mode].append((end - graph_start) / 1e6)
        launches[mode].append(count)
    for mode in before_ms:
        steps_in_mode = len(before_ms[mode])
        print(
            f"{mode}: {steps_in_mode} forwards; before the graph launch {statistics.fmean(before_ms[mode]):.3f} ms, "
            f"total {sum(before_ms[mode]):.0f} ms; after it {statistics.fmean(after_ms[mode]):.3f} ms; "
            f"launches before it {statistics.fmean(launches[mode]):.1f}"
        )
        for name, value in by_kernel[mode].most_common(14):
            print(
                f"  {value / steps_in_mode:5.1f} a step, device {device_ns[mode][name] / max(value, 1) / 1e3:6.1f} us, "
                f"launch call {api_ns[mode][name] / max(value, 1) / 1e3:5.1f} us  {name}"
            )


if __name__ == "__main__":
    main()
