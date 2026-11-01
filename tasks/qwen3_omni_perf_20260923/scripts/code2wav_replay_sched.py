# note(ratish): usage python3 code2wav_replay_sched.py <serve.sqlite> --window <window.txt>. Needs a report traced with
# --cpuctxsw=process-tree. Splits code2wav's graph replays (fast under 2 ms, slow over 10 ms) into the time the
# code2wav thread ran on a CPU and the time it was off one, the off time by its state when switched out (preempted while
# runnable, or blocked), and names the threads that ran on that CPU while it was preempted.
import argparse
import bisect
import collections
import sqlite3
import statistics
import sys

import omni_census


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--window")
    args = parser.parse_args()
    sys.argv = sys.argv[:1]
    r = omni_census.Report(args.report, args.window)
    replays = [
        x
        for tid, items in r.ranges.items()
        if r.stage_of_tid(tid) == "code2wav"
        for x in items
        if x.kind == "c2w.replay" and r.t0 <= x.start <= r.t1
    ]
    if not replays:
        print("no code2wav replays in the window")
        return
    db = sqlite3.connect(args.report)
    states = dict(db.execute("select id, name from ENUM_SAMPLING_THREAD_STATE"))
    tid = replays[0].tid
    events = db.execute(
        "select start, cpu, isSchedIn, threadState from SCHED_EVENTS where globalTid = ? order by start",
        (tid,),
    ).fetchall()
    if not events:
        print(
            "no scheduling events for the code2wav thread: trace with --cpuctxsw=process-tree"
        )
        return
    off = []
    last_out = None
    for start, cpu, sched_in, state in events:
        if sched_in == 0:
            last_out = (start, cpu, states.get(state, str(state)))
        elif last_out is not None:
            off.append((last_out[0], start, last_out[1], last_out[2]))
            last_out = None
        else:
            pass
    off_starts = [item[0] for item in off]
    cpu_rows = collections.defaultdict(list)
    for start, cpu, sched_in, other in db.execute(
        "select start, cpu, isSchedIn, globalTid from SCHED_EVENTS where isSchedIn = 1 and start between ? and ? order by start",
        (r.t0, r.t1),
    ):
        cpu_rows[cpu].append((start, other))
    for name, chosen in (
        ("fast replays (< 2 ms)", [x for x in replays if x.wall < 2e6]),
        ("slow replays (> 10 ms)", [x for x in replays if x.wall > 10e6]),
    ):
        if not chosen:
            continue
        by_state = collections.Counter()
        runners = collections.Counter()
        for x in chosen:
            low = bisect.bisect_left(off_starts, x.start - 1_000_000_000)
            for out_start, in_start, cpu, state in off[low:]:
                if out_start >= x.end:
                    break
                overlap = max(0, min(in_start, x.end) - max(out_start, x.start))
                if overlap:
                    by_state[state] += overlap
                    if state.lower().startswith(("runnable", "running", "preempt")):
                        rows = cpu_rows.get(cpu, [])
                        at = bisect.bisect_left(rows, (max(out_start, x.start), -1))
                        for when, other in rows[max(at - 1, 0) :]:
                            if when >= min(in_start, x.end):
                                break
                            runners[
                                f"{r.stage_of_pid(omni_census.pid_of(other))}/{r.thread_names.get(other, other)}"
                            ] += 1
                    else:
                        pass
                else:
                    pass
        wall = statistics.fmean(x.wall for x in chosen)
        off_total = sum(by_state.values()) / len(chosen)
        print(
            f"{name}: n {len(chosen)}, wall {wall / 1e6:.2f} ms, off a CPU {off_total / 1e6:.2f} ms: "
            + ", ".join(
                f"{state} {ns / len(chosen) / 1e6:.2f}"
                for state, ns in by_state.most_common()
            )
        )
        if runners:
            print(
                "  ran on the CPU while it was preempted: "
                + ", ".join(f"{k} {v}" for k, v in runners.most_common(8))
            )
        else:
            pass


main()
