"""Read hop_probe records of one baseline.py run and print per cell: queue waits, the stream chain per edge, event
loop lateness, GC pauses and the stacks taken at each loop stall.

usage: python3 hop_report.py /data/runtime-perf/b3_tts [--cells c16_clean,c64_clean]

Queue waits pair the i-th put with the i-th get of each queue (stage inbox and outbox are FIFO with one producer
thread); a pair whose request ids differ is counted as a mismatch and skipped. The stream chain joins, per request and
target, the n-th stream message on the sender's outbox with chunk id n on the target's inbox:
  put        scheduler thread puts the step's messages on its outbox
  waiter     the outbox get returns on the waiter thread (first of a drain) or the loop thread (rest of the batch)
  route      the loop routes it and puts the stream item on the target scheduler's inbox
  pickup     the target scheduler's get returns
Lag is the lateness of a 5 ms sleep on each event loop. GC is collections of generation 1 or 2, or any over 1 ms.
"""

import argparse
import collections
import json
import statistics
from pathlib import Path


def pct(values, q):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else float("nan")


def line(label, values_ms):
    if not values_ms:
        return f"    {label}: none"
    else:
        pass
    return (f"    {label}: p50 {statistics.median(values_ms):.3f} p90 {pct(values_ms, 0.9):.3f} "
            f"p99 {pct(values_ms, 0.99):.3f} max {max(values_ms):.3f} ms n={len(values_ms)}")


def load(run: Path):
    queue_records, other = [], []
    for path in sorted((run / "probe").glob("p.*.jsonl")):
        pid = path.name.split(".")[1]
        for raw in path.open():
            item = json.loads(raw)
            if isinstance(item, list):
                queue_records.append(item)
            else:
                item["pid"] = pid
                other.append(item)
    return queue_records, other


def report_cell(name, start_ns, end_ns, queue_records, other):
    print(f"== {name}")
    in_window = [r for r in queue_records if start_ns <= r[0] <= end_ns]
    by_queue = collections.defaultdict(lambda: ([], []))
    for record in sorted(in_window, key=lambda r: r[0]):
        puts, gets = by_queue[record[1]]
        (puts if record[2] == "put" else gets).append(record)
    waits = collections.defaultdict(list)
    for queue, (puts, gets) in sorted(by_queue.items()):
        mismatched = 0
        for put, get in zip(puts, gets):
            if put[4] != get[4]:
                mismatched += 1
                continue
            else:
                pass
            waits[(queue, put[5], get[3])].append((get[0] - put[0]) / 1e6)
        print(f"  queue {queue}: puts {len(puts)} gets {len(gets)} mismatched {mismatched}")
    for (queue, kind, thread), values in sorted(waits.items(), key=lambda item: -len(item[1]))[:16]:
        print(line(f"{queue} {kind} got by {thread}", values))
    outbox_gets = collections.defaultdict(list)
    outbox_puts = collections.defaultdict(list)
    inbox_puts, inbox_gets = {}, {}
    for record in sorted(in_window, key=lambda r: r[0]):
        ns, queue, op, thread, rid, kind, target, chunk, source = record
        stage, direction = queue.rsplit(".", 1)
        if direction == "out" and kind == "stream" and target:
            (outbox_puts if op == "put" else outbox_gets)[(rid, stage, target)].append(ns)
        elif direction == "in" and kind == "stream_chunk" and chunk is not None:
            (inbox_puts if op == "put" else inbox_gets)[(rid, source, stage, chunk)] = ns
        else:
            pass
    segments = collections.defaultdict(lambda: collections.defaultdict(list))
    for (rid, stage, target), gets in outbox_gets.items():
        puts = outbox_puts.get((rid, stage, target), [])
        for index, got in enumerate(gets):
            key = (rid, stage, target, index)
            edge = f"{stage}->{target}"
            if index < len(puts):
                segments[edge]["put to waiter"].append((got - puts[index]) / 1e6)
            else:
                pass
            if key in inbox_puts:
                segments[edge]["waiter to route"].append((inbox_puts[key] - got) / 1e6)
                if key in inbox_gets:
                    segments[edge]["route to pickup"].append((inbox_gets[key] - inbox_puts[key]) / 1e6)
                    if index < len(puts):
                        segments[edge]["put to pickup"].append((inbox_gets[key] - puts[index]) / 1e6)
                    else:
                        pass
                else:
                    pass
            else:
                pass
    for edge, parts in segments.items():
        print(f"  stream chain {edge} (put, outbox get, target inbox put, target pick-up; cross-process edges include the transport in waiter to route)")
        for part in ("put to waiter", "waiter to route", "route to pickup", "put to pickup"):
            print(line(part, parts.get(part, [])))
    lags = collections.defaultdict(list)
    worst = collections.defaultdict(float)
    late = collections.Counter()
    for item in other:
        if not start_ns <= item["ns"] <= end_ns:
            continue
        else:
            pass
        if item.get("k") == "lag":
            key = (item["pid"], item["loop"])
            lags[key].append(item["sum_ms"] / max(item["n"], 1))
            worst[key] = max(worst[key], item["max_ms"])
        elif item.get("k") == "lag1":
            late[(item["pid"], item["loop"])] += 1
        else:
            pass
    span_s = (end_ns - start_ns) / 1e9
    for key, means in sorted(lags.items()):
        print(f"  loop {key[1]} pid {key[0]}: mean lateness {statistics.mean(means):.3f} ms, worst {worst[key]:.1f} ms, "
              f"wakeups later than 2 ms {late[key]} ({late[key] / span_s:.1f}/s)")
    collections_by_pid = collections.defaultdict(list)
    for item in other:
        if item.get("k") == "gc" and start_ns <= item["ns"] <= end_ns:
            collections_by_pid[(item["pid"], item["gen"], item["th"])].append(item["dur_ns"] / 1e6)
        else:
            pass
    for (pid, gen, thread), durations in sorted(collections_by_pid.items()):
        print(f"  gc pid {pid} gen {gen} on {thread}: {len(durations)} collections, total {sum(durations):.1f} ms, "
              f"max {max(durations):.2f} ms")
    for item in other:
        if item.get("k") == "stall" and start_ns <= item["ns"] <= end_ns:
            print(f"  stall loop {item['loop']} pid {item['pid']} quiet {item['quiet_ms']:.0f} ms at capture; "
                  f"loop thread {item['loop_thread']}")
            for thread, frames in item["stacks"].items():
                if thread.startswith("hop-probe"):
                    continue
                else:
                    pass
                print(f"    {thread}: {' <- '.join(reversed(frames[-5:]))[:400]}")
        else:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--cells", default="")
    args = parser.parse_args()
    run = Path(args.run)
    queue_records, other = load(run)
    for item in other:
        if item.get("k") == "heap":
            print(f"heap pid {item['pid']}: {item['tracked']} tracked objects, full collection "
                  f"{item['full_collect_ns'] / 1e6:.1f} ms, frozen after {item['frozen']}")
            print(f"  types {item['top_types'][:12]}")
            print(f"  packages {item['top_packages'][:12]}")
        else:
            pass
    wanted = set(args.cells.split(",")) if args.cells else None
    for cell in sorted(path for path in run.iterdir() if (path / "window.json").exists()):
        if wanted and cell.name not in wanted:
            continue
        else:
            pass
        window = json.loads((cell / "window.json").read_text())
        report_cell(cell.name, int(window["start"] * 1e9), int(window["end"] * 1e9), queue_records, other)


if __name__ == "__main__":
    main()
