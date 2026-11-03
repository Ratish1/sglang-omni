"""Where the engine's device work sits idle and what the engine thread was doing meanwhile, from one pipeline_nvtx nsys
report. A gap the next launch closes late is host time a trim can convert; the rest of a gap is the launched work waiting
on the device, which no host trim can convert. So this bounds any host item at the load traced, and names its range.

usage: python3 engine_idle_ledger.py serve.sqlite --bench-log bench.log
"""

import argparse
import collections

from pipeline_census import Report, ms

ENGINE_THREAD = "scheduler-tts_engine"
SAMPLE_NS = 50_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite")
    parser.add_argument("--bench-log", required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    report = Report(args.sqlite, args.bench_log)
    engine_tid = next(tid for tid, name in report.thread_names.items() if name == ENGINE_THREAD)
    engine_work = sorted(
        (start, end, corr)
        for start, end, _, _, _, _, corr, _ in report.device
        if report.api_tid.get(corr) == engine_tid and start >= report.t0 and end <= report.t1
    )
    busy_ns = 0
    current_start, current_end = engine_work[0][0], engine_work[0][1]
    host_late_ns = 0
    device_wait_ns = 0
    host_by_range: collections.Counter[str] = collections.Counter()
    host_by_step: collections.Counter[str] = collections.Counter()
    late_by_next: collections.Counter[str] = collections.Counter()
    gap_count = 0
    for start, end, corr in engine_work[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        busy_ns += current_end - current_start
        gap_count += 1
        launch = min(max(report.api_start[corr], current_end), start)
        host_late_ns += launch - current_end
        device_wait_ns += start - launch
        next_owner = report.owner_name(report.owner.get(corr))
        late_by_next[next_owner] += launch - current_end
        for moment in range(current_end, launch, SAMPLE_NS):
            item = report.innermost(report.ranges, report.starts, engine_tid, moment)
            kind = item.kind if item is not None else "(no range)"
            step = "outside a step"
            for parent_kind in item.chain() if item is not None else ():
                if parent_kind.startswith("sched.batch") or parent_kind.startswith("sched.result"):
                    step = parent_kind
                    break
            host_by_range[kind] += SAMPLE_NS
            host_by_step[step] += SAMPLE_NS
        current_start, current_end = start, end
    busy_ns += current_end - current_start
    window_ns = report.t1 - report.t0
    print(f"window {ms(window_ns):.0f} ms; engine work on the device {ms(busy_ns):.0f} ms, {100 * busy_ns / window_ns:.1f} %")
    print(f"idle gaps {gap_count}, {ms(host_late_ns + device_wait_ns):.0f} ms")
    print(f"  host late, the next launch had not been made: {ms(host_late_ns):.0f} ms, {100 * host_late_ns / window_ns:.1f} % of the window")
    print(f"  launched and waiting on the device: {ms(device_wait_ns):.0f} ms, {100 * device_wait_ns / window_ns:.1f} %")
    print(f"bound if every late launch were on time: {window_ns / max(window_ns - host_late_ns, 1):.3f}x")
    print("host late by what launched next:")
    for name, value in late_by_next.most_common():
        print(f"  {name:<28}{ms(value):8.0f} ms")
    print("host late by the step the engine thread was in:")
    for name, value in host_by_step.most_common():
        print(f"  {name:<28}{ms(value):8.0f} ms")
    print("host late by innermost range:")
    for name, value in host_by_range.most_common(args.top):
        print(f"  {name:<28}{ms(value):8.0f} ms")


if __name__ == "__main__":
    main()
