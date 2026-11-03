"""Qwen3-Omni talker scheduler loop in a fine marks trace: every pass between two inbox reads, by what the pass ran.

usage: python3 talker_loop.py <main boot,boot> <branch boot,boot> [--cell c16_nsys]

A pass runs from one omni.loop_admit start to the next. Its kind is the mode of the forward it ran (EXTEND, DECODE) or
none, and its arrivals are the requests the next pass admitted: what reached the inbox while this pass ran. Printed per
arm: passes per second, each kind's share of the wall, its length and arrivals, prefill step sizes, and each mark's
mean and p50 per pass of a kind. Needs only NVTX; a trace with CUDA adds nothing here (span_table.py reads that).
"""

import argparse
import bisect
import re
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path

STARTUP = re.compile(r"sglang_ar_startup stage=(\w+) gpu_id=\d+ .*pid=(\d+)")
ADMIT = re.compile(r"omni\.loop_admit rows=(\d+)")
STEP = re.compile(r"omni\.forward step\[(\w+) bs=(\d+)")
MARKS = (
    "omni.loop_admit",
    "omni.build_request",
    "scheduler.get_next_batch_to_run",
    "omni.build_forward_batch",
    "omni.before_prefill",
    "omni.before_decode",
    "omni.prepare_decode_buffers",
    "omni.write_feedback",
    "omni.forward",
    "omni.sample",
    "omni.suppress_write",
    "omni.sampling_logits",
    "omni.sampler",
    "omni.post_prefill",
    "omni.post_decode",
    "omni.predictor",
    "omni.stage_tokens",
    "omni.emit",
    "omni.publish_next_tokens",
    "omni.finalize",
    "omni.host_tokens_wait",
    "scheduler.process_batch_result",
)


def talker_pid(boot: Path) -> int:
    with open(boot / "serve.log", encoding="utf-8", errors="replace") as log:
        for line in log:
            if (match := STARTUP.search(line)) and match[1] == "talker_ar":
                return int(match[2])
            else:
                pass
    raise RuntimeError(f"no talker_ar startup line in {boot}/serve.log")


def read_boot(boot: Path, cell: str) -> dict[str, list]:
    pid = talker_pid(boot)
    db = sqlite3.connect(boot / cell / "trace.sqlite")
    text = "coalesce(text, (SELECT value FROM StringIds WHERE id = textId))"
    admits, steps, marks = [], [], defaultdict(list)
    for start, end, name, gtid in db.execute(
        f"SELECT start, end, {text}, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL"
    ):
        if not name or (gtid >> 24) & 0xFFFFFF != pid:
            continue
        elif match := ADMIT.match(name):
            admits.append((start, int(match[1])))
            marks["omni.loop_admit"].append((start, end))
        elif match := STEP.match(name):
            steps.append((start, end, match[1], int(match[2])))
            marks["omni.forward"].append((start, end))
        elif name in MARKS:
            marks[name].append((start, end))
        else:
            pass
    admits.sort()
    steps.sort()
    step_starts = [step[0] for step in steps]
    passes = []
    for (start, _), (next_start, arrivals) in zip(admits, admits[1:]):
        index = bisect.bisect_left(step_starts, start)
        kind, rows = "none", 0
        if index < len(steps) and steps[index][0] < next_start:
            kind, rows = steps[index][2], steps[index][3]
        else:
            pass
        passes.append((start, next_start, kind, rows, arrivals))
    per_mark = defaultdict(lambda: defaultdict(list))
    starts = [item[0] for item in passes]
    for name, spans in marks.items():
        totals = defaultdict(float)
        for start, end in spans:
            index = bisect.bisect_right(starts, start) - 1
            if 0 <= index and start < passes[index][1]:
                totals[index] += (end - start) / 1e6
            else:
                pass
        for index, item in enumerate(passes):
            per_mark[item[2]][name].append(totals.get(index, 0.0))
    return {"passes": passes, "per_mark": per_mark}


def print_arm(label: str, boots: list[str], cell: str) -> None:
    passes, per_mark = [], defaultdict(lambda: defaultdict(list))
    wall = 0.0
    for boot in boots:
        data = read_boot(Path(boot), cell)
        passes += data["passes"]
        wall += (data["passes"][-1][1] - data["passes"][0][0]) / 1e9
        for kind, names in data["per_mark"].items():
            for name, values in names.items():
                per_mark[kind][name] += values
    print(
        f"== {label}: {len(boots)} boots, {wall:.1f} s traced, {len(passes) / wall:.1f} passes/s"
    )
    for kind in ("EXTEND", "DECODE", "none"):
        items = [item for item in passes if item[2] == kind]
        if not items:
            continue
        else:
            pass
        lengths = [(item[1] - item[0]) / 1e6 for item in items]
        arrivals = [item[4] for item in items]
        print(
            f"  {kind:7s} {len(items) / wall:7.2f}/s  share {100 * sum(lengths) / 1e3 / wall:5.1f} %  "
            f"length mean {statistics.mean(lengths):6.2f} p50 {statistics.median(lengths):6.2f} ms  "
            f"arrivals/pass {statistics.mean(arrivals):.3f}  arrivals/s {sum(arrivals) / wall:.2f}  "
            f"rows mean {statistics.mean(item[3] for item in items):.2f}"
        )
    sizes = Counter(item[3] for item in passes if item[2] == "EXTEND")
    total = sum(sizes.values())
    requests = sum(size * count for size, count in sizes.items())
    print(
        f"  prefill sizes {dict(sorted(sizes.items()))}, requests a step {requests / max(total, 1):.3f}"
    )
    for kind in ("EXTEND", "DECODE"):
        print(f"  {kind} pass, per mark, mean (p50) ms")
        for name in MARKS:
            values = per_mark[kind].get(name, [])
            if values and max(values) > 0:
                print(
                    f"    {name:34s} {statistics.mean(values):7.3f} ({statistics.median(values):7.3f})"
                )
            else:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("main_boots")
    parser.add_argument("branch_boots")
    parser.add_argument("--cell", default="c16_nsys")
    args = parser.parse_args()
    print_arm("main", args.main_boots.split(","), args.cell)
    print_arm("branch", args.branch_boots.split(","), args.cell)


if __name__ == "__main__":
    main()
