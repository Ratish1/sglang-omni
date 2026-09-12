"""Per thread readout of an nsys SQLite export.

Usage: python nsys_threads.py <trace.sqlite> <out.json>

For every thread: name, launch calls (count, mean and p90 duration), kernels attributed
through the correlation id (count, launch to start mean and p90, execution sum), the
kernel names that identify the role, sync calls by API, and the GIL wait and hold totals
from the NVTX ranges.
"""

import json
import sqlite3
import sys
from collections import defaultdict


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return values[int(q * (len(values) - 1))]


def main():
    path, out = sys.argv[1], sys.argv[2]
    db = sqlite3.connect(path)
    db.execute("PRAGMA temp_store=MEMORY")
    names = {row[0]: row[1] for row in db.execute("select id, value from StringIds")}
    tnames = {
        row[0]: names.get(row[1])
        for row in db.execute("select globalTid, nameId from ThreadNames")
    }
    threads = defaultdict(
        lambda: {
            "launch": [],
            "graph_launch": [],
            "sync": defaultdict(list),
            "kernels": 0,
            "q": [],
            "exec_us": 0.0,
            "kernel_names": defaultdict(int),
            "gil_wait_us": 0.0,
            "gil_wait_n": 0,
            "gil_hold_us": 0.0,
            "gil_hold_n": 0,
            "memcpy": [],
        }
    )
    launch_map = {}
    for tid, start, end, name_id, corr in db.execute(
        "select globalTid, start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        name = names.get(name_id, "")
        dur = (end - start) / 1e3
        th = threads[tid]
        if name.startswith("cudaGraphLaunch"):
            th["graph_launch"].append(dur)
            launch_map[corr] = (tid, start)
        elif name.startswith("cudaLaunchKernel") or name.startswith("cuLaunchKernel"):
            th["launch"].append(dur)
            launch_map[corr] = (tid, start)
        elif name.startswith("cudaMemcpyAsync"):
            th["memcpy"].append(dur)
            launch_map[corr] = (tid, start)
        elif "Synchronize" in name or "StreamWaitEvent" in name or "EventQuery" in name:
            th["sync"][name].append(dur)
    for start, end, corr, name_id, graph in db.execute(
        "select start, end, correlationId, demangledName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        hit = launch_map.get(corr)
        if hit is None:
            continue
        tid, lstart = hit
        th = threads[tid]
        th["kernels"] += 1
        th["exec_us"] += (end - start) / 1e3
        if not graph:
            th["q"].append((start - lstart) / 1e3)
        th["kernel_names"][names.get(name_id, "")[:70]] += 1
    for tid, start, end, text_id, text in db.execute(
        "select globalTid, start, end, textId, text from NVTX_EVENTS where textId is not null or text is not null"
    ):
        label = text if text else names.get(text_id, "")
        if "Waiting for GIL" in label:
            threads[tid]["gil_wait_us"] += (end - start) / 1e3
            threads[tid]["gil_wait_n"] += 1
        elif "Holding GIL" in label:
            threads[tid]["gil_hold_us"] += (end - start) / 1e3
            threads[tid]["gil_hold_n"] += 1
    summary = {}
    for tid, th in threads.items():
        if len(th["launch"]) + len(th["graph_launch"]) + th["gil_wait_n"] < 50:
            continue
        top = sorted(th["kernel_names"].items(), key=lambda kv: -kv[1])[:6]
        summary[str(tid)] = {
            "name": tnames.get(tid),
            "launch_n": len(th["launch"]),
            "launch_mean_us": (
                (sum(th["launch"]) / len(th["launch"])) if th["launch"] else None
            ),
            "launch_p90_us": pct(th["launch"], 0.9),
            "launch_sum_ms": sum(th["launch"]) / 1e3,
            "graph_launch_n": len(th["graph_launch"]),
            "graph_launch_mean_us": (
                (sum(th["graph_launch"]) / len(th["graph_launch"]))
                if th["graph_launch"]
                else None
            ),
            "memcpy_n": len(th["memcpy"]),
            "memcpy_mean_us": (
                (sum(th["memcpy"]) / len(th["memcpy"])) if th["memcpy"] else None
            ),
            "kernels": th["kernels"],
            "exec_sum_ms": th["exec_us"] / 1e3,
            "q_n": len(th["q"]),
            "q_mean_us": (sum(th["q"]) / len(th["q"])) if th["q"] else None,
            "q_p50_us": pct(th["q"], 0.5),
            "q_p90_us": pct(th["q"], 0.9),
            "q_p99_us": pct(th["q"], 0.99),
            "sync": {
                k: {"n": len(v), "mean_us": sum(v) / len(v), "sum_ms": sum(v) / 1e3}
                for k, v in th["sync"].items()
            },
            "gil_wait_ms": th["gil_wait_us"] / 1e3,
            "gil_wait_n": th["gil_wait_n"],
            "gil_hold_ms": th["gil_hold_us"] / 1e3,
            "gil_hold_n": th["gil_hold_n"],
            "top_kernels": top,
        }
    with open(out, "w") as handle:
        json.dump(summary, handle, indent=1)
    print("done", path, "threads", len(summary))


if __name__ == "__main__":
    main()
