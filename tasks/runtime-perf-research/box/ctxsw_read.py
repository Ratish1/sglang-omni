"""Who owns the card in an nsys trace taken with --gpuctxsw, and what the thinker's steps and the talker's host waits look like.

usage: python3 ctxsw_read.py trace.sqlite [trace.sqlite ...]

Context residency runs from a context's RESTORE_START (or CONTEXT_START) to its next SAVE_END. A hardware context maps to the
process whose kernels start inside its residencies. The thinker and talker are told apart by the code2wav kernels the talker
process runs. A thinker step is one cudaGraphLaunch in the thinker process; its kernels carry the launch's correlation id.
"""

import bisect
import collections
import sqlite3
import statistics
import sys

SAVE_END, RESTORE_START, CONTEXT_START = 7, 8, 9


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def quantiles(values: list[float]) -> str:
    if not values:
        return "none"
    else:
        pass
    ordered = sorted(values)
    return (
        f"n {len(ordered)} mean {statistics.fmean(ordered):.3f} p50 {ordered[len(ordered) // 2]:.3f} "
        f"p90 {ordered[int(0.9 * len(ordered))]:.3f} p99 {ordered[int(0.99 * len(ordered))]:.3f}"
    )


def residencies(connection: sqlite3.Connection) -> dict[int, list[tuple[int, int]]]:
    open_since: dict[int, int] = {}
    spans: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for tag, context, stamp in connection.execute(
        "select tag, contextId, timestamp from GPU_CONTEXT_SWITCH_EVENTS order by timestamp"
    ):
        if tag in (RESTORE_START, CONTEXT_START):
            open_since[context] = stamp
        elif tag == SAVE_END and context in open_since:
            spans[context].append((open_since.pop(context), stamp))
        else:
            pass
    return spans


def read(path: str) -> None:
    connection = sqlite3.connect(path)
    names = dict(connection.execute("select id, value from StringIds"))
    kernels = connection.execute(
        "select start, end, globalPid, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    spans = residencies(connection)
    window_start = max(
        min(span[0] for values in spans.values() for span in values), kernels[0][0]
    )
    window_end = min(
        max(span[1] for values in spans.values() for span in values), kernels[-1][1]
    )
    window_s = (window_end - window_start) / 1e9

    owner_votes: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for context, values in spans.items():
        starts = [span[0] for span in values]
        for start, _, global_pid, _, _ in kernels[:: max(len(kernels) // 200000, 1)]:
            index = bisect.bisect_right(starts, start) - 1
            if index >= 0 and values[index][0] <= start <= values[index][1]:
                owner_votes[context][pid_of(global_pid)] += 1
            else:
                pass
    context_pid = {
        context: votes.most_common(1)[0][0]
        for context, votes in owner_votes.items()
        if votes
    }

    thread_names: dict[int, set[str]] = collections.defaultdict(set)
    for global_tid, name_id in connection.execute(
        "select globalTid, nameId from ThreadNames"
    ):
        thread_names[pid_of(global_tid)].add(names[name_id])
    kernel_names: dict[int, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    for _, _, global_pid, short_name, _ in kernels:
        kernel_names[pid_of(global_pid)][names[short_name]] += 1
    roles: dict[int, str] = {}
    for pid in kernel_names:
        if "scheduler-image" in thread_names[pid]:
            roles[pid] = "image"
        elif "scheduler-audio" in thread_names[pid]:
            roles[pid] = "audio"
        elif any(
            "conv" in name.lower() or "snake" in name.lower()
            for name in kernel_names[pid]
        ):
            roles[pid] = "talker"
        else:
            roles[pid] = "thinker"
    print(f"== {path}  window {window_s:.2f} s")
    for pid, role in sorted(roles.items(), key=lambda item: item[1]):
        top = ", ".join(
            f"{name[:28]} {count}" for name, count in kernel_names[pid].most_common(3)
        )
        print(f"  pid {pid} {role:8s} kernels {sum(kernel_names[pid].values())}: {top}")

    resident_s: dict[str, float] = collections.defaultdict(float)
    residency_ms: dict[str, list[float]] = collections.defaultdict(list)
    for context, values in spans.items():
        role = roles.get(context_pid.get(context, -1), f"ctx{context}")
        for start, end in values:
            clipped = min(end, window_end) - max(start, window_start)
            if clipped > 0:
                resident_s[role] += clipped / 1e9
                residency_ms[role].append((end - start) / 1e6)
            else:
                pass
    switches = sorted(
        (stamp, tag, context)
        for tag, context, stamp in connection.execute(
            "select tag, contextId, timestamp from GPU_CONTEXT_SWITCH_EVENTS order by timestamp"
        )
        if window_start <= stamp <= window_end
    )
    restores = [
        stamp for stamp, tag, _ in switches if tag in (RESTORE_START, CONTEXT_START)
    ]
    gaps_us = []
    last_save = None
    for stamp, tag, _ in switches:
        if tag == SAVE_END:
            last_save = stamp
        elif tag in (RESTORE_START, CONTEXT_START) and last_save is not None:
            gaps_us.append((stamp - last_save) / 1e3)
            last_save = None
        else:
            pass
    print(
        f"  switches {len(restores) / window_s:.0f}/s, gap us {quantiles(gaps_us)}, switching share {sum(gaps_us) / 1e6 / window_s:.3f}"
    )
    for role in sorted(resident_s):
        print(
            f"  {role:8s} resident {resident_s[role] / window_s:.3f} of the card, {len(residency_ms[role]) / window_s:.0f} residencies/s, ms {quantiles(residency_ms[role])}"
        )

    by_correlation: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for start, end, global_pid, _, correlation in kernels:
        by_correlation[correlation].append((start, end))
    runtime = connection.execute(
        "select start, end, globalTid, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where start between ? and ?",
        (window_start, window_end),
    ).fetchall()
    role_of_pid = roles
    calls: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    thinker_steps = []
    for start, end, global_tid, name_id, correlation in runtime:
        role = role_of_pid.get(pid_of(global_tid), "other")
        name = names[name_id].split("_v")[0]
        calls[(role, name)].append((end - start) / 1e6)
        if (
            role == "thinker"
            and name == "cudaGraphLaunch"
            and correlation in by_correlation
        ):
            graph_kernels = by_correlation[correlation]
            first = min(kernel[0] for kernel in graph_kernels)
            last = max(kernel[1] for kernel in graph_kernels)
            others = sum(1 for stamp in restores if first < stamp < last)
            busy = sum(kernel[1] - kernel[0] for kernel in graph_kernels)
            thinker_steps.append(
                ((first - end) / 1e6, (last - first) / 1e6, others, busy / 1e6)
            )
        else:
            pass
    if thinker_steps:
        print(
            f"  thinker graph launch to first kernel ms {quantiles([step[0] for step in thinker_steps])}"
        )
        print(
            f"  thinker graph span ms {quantiles([step[1] for step in thinker_steps])}"
        )
        print(
            f"  thinker graph kernel busy ms {quantiles([step[3] for step in thinker_steps])}"
        )
        print(
            f"  context restores inside a thinker graph span {quantiles([float(step[2]) for step in thinker_steps])}"
        )
    else:
        pass
    for role in ("talker", "thinker"):
        for name in (
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
            "cudaMemcpyAsync",
            "cudaEventQuery",
            "cudaGraphLaunch",
        ):
            values = calls.get((role, name), [])
            if values:
                print(
                    f"  {role:8s} {name:22s} {len(values) / window_s:7.0f}/s  {sum(values) / window_s:7.1f} ms/s  ms {quantiles(values)}"
                )
            else:
                pass


def main() -> None:
    for path in sys.argv[1:]:
        read(path)


if __name__ == "__main__":
    main()
