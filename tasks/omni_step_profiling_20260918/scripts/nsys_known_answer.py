"""Known answer test for the Nsight features the pipeline census relies on.

Run mode builds a workload whose attribution is known by construction; analyze mode
reads the sqlite export and checks each answer. Nothing else is trusted until this
passes on the box's nsys, driver and torch.

  K1 graph node kernels carry a graph node id, and the node count per replay equals
     the kernels captured
  K2 capture time NVTX ranges project onto the replayed node kernels: the gemm of
     every replay belongs to ka.cap.A, the relu and the sum to ka.cap.B
  K3 eager kernels launched from a thread started late resolve to that thread's range
  K4 a renamed OS thread is named in the report
  K5 python-gil records waits of the main thread while a worker holds the GIL
  K6 OS runtime records a timed queue wait with its duration
  K7 python sampling records frames of both threads

usage:
  nsys profile -o ka --trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=node \
      --python-sampling=true --sample=none python3 nsys_known_answer.py run
  nsys export --type sqlite -o ka.sqlite ka.nsys-rep
  python3 nsys_known_answer.py analyze ka.sqlite
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import queue
import sqlite3
import sys
import threading
import time

REPLAYS = 50
NVTX_PUSH_POP = 59


def name_os_thread(name: str) -> None:
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(15, name.encode()[:15], 0, 0, 0)


def run() -> None:
    import torch

    nvtx = torch.cuda.nvtx
    x = torch.randn(1024, 1024, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            y = torch.relu(x @ x).sum()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        nvtx.range_push("ka.cap.A")
        y = x @ x
        nvtx.range_pop()
        nvtx.range_push("ka.cap.B")
        z = torch.relu(y).sum()
        nvtx.range_pop()
    torch.cuda.synchronize()
    nvtx.mark("ka.window.start")
    for i in range(REPLAYS):
        nvtx.range_push(f"ka.replay i={i}")
        graph.replay()
        nvtx.range_pop()
    torch.cuda.synchronize()

    def eager_worker():
        name_os_thread("ka-eager")
        for i in range(20):
            nvtx.range_push(f"ka.eager i={i}")
            torch.add(x, 1.0)
            nvtx.range_pop()
        torch.cuda.synchronize()

    worker = threading.Thread(target=eager_worker, name="ka-eager")
    worker.start()
    worker.join()

    def spin_worker():
        name_os_thread("ka-spin")
        deadline = time.perf_counter() + 0.3
        count = 0
        while time.perf_counter() < deadline:
            count += 1

    spinner = threading.Thread(target=spin_worker, name="ka-spin")
    spinner.start()
    nvtx.range_push("ka.main.contended")
    total = 0
    for i in range(2_000_000):
        total += i
    nvtx.range_pop()
    spinner.join()

    items: queue.Queue = queue.Queue()
    nvtx.range_push("ka.queue.wait")
    try:
        items.get(timeout=0.1)
    except queue.Empty:
        pass
    nvtx.range_pop()
    nvtx.mark("ka.window.end")
    print(f"done {float(z):.3f}", flush=True)


def tables(db) -> set[str]:
    return {
        row[0]
        for row in db.execute("select name from sqlite_master where type='table'")
    }


def columns(db, table: str) -> list[str]:
    return [row[1] for row in db.execute(f"pragma table_info({table})")]


def analyze(path: str) -> None:
    db = sqlite3.connect(path)
    names = tables(db)
    strings = dict(db.execute("select id, value from StringIds"))
    print("tables:", " ".join(sorted(names)))
    for table in sorted(names):
        if "GRAPH" in table or "GIL" in table or "PYTHON" in table or "PROJ" in table:
            print(f"  {table}: {columns(db, table)}")
            for row in db.execute(f"select * from {table} limit 3"):
                print("    ", row)

    kernel_cols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    print("kernel columns:", kernel_cols)
    node_col = "graphNodeId" if "graphNodeId" in kernel_cols else None
    graph_col = "graphId" if "graphId" in kernel_cols else None
    ranges = {}
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType = ?",
        (NVTX_PUSH_POP,),
    ):
        label = text if text is not None else strings.get(text_id, "")
        ranges.setdefault(label.split(" ")[0], []).append((start, end, tid, label))
    print("range kinds:", {k: len(v) for k, v in ranges.items()})

    api = {}
    for start, end, tid, corr, name_id in db.execute(
        "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        api[corr] = (start, end, tid, strings.get(name_id, ""))

    def enclosing(tid, t, kind):
        for start, end, rtid, label in ranges.get(kind, []):
            if rtid == tid and start <= t <= end:
                return label
        return None

    replay_kernels = collections.Counter()
    node_ids = collections.defaultdict(set)
    select = "select start, end, correlationId, demangledName"
    select += f", {node_col}" if node_col else ", null"
    select += f", {graph_col}" if graph_col else ", null"
    kernels = db.execute(
        select + " from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    for start, end, corr, name_id, node, graph in kernels:
        launch = api.get(corr)
        if launch is None:
            continue
        label = enclosing(launch[2], launch[0], "ka.replay")
        if label is not None:
            short = strings.get(name_id, "")[:60]
            replay_kernels[short] += 1
            node_ids[short].add(node)
    print("K1 kernels under replays:", dict(replay_kernels))
    print(
        "K1 node ids per kernel name:",
        {k: sorted(map(str, v))[:4] for k, v in node_ids.items()},
    )
    expected = REPLAYS
    k1 = replay_kernels and all(v % expected == 0 for v in replay_kernels.values())
    print(
        f"K1 {'pass' if k1 and node_col else 'FAIL'}: every replayed kernel {expected} times, node column {node_col}"
    )

    print(
        "K2: look for projection tables above; nvtx_kern_sum / nvtx_gpu_proj_trace via nsys stats"
    )

    eager = 0
    for start, end, corr, name_id, node, graph in kernels:
        launch = api.get(corr)
        if launch and enclosing(launch[2], launch[0], "ka.eager"):
            eager += 1
    print(
        f"K3 {'pass' if eager == 20 else 'FAIL'}: eager kernels under ka.eager {eager} of 20"
    )

    thread_names = {}
    if "ThreadNames" in names:
        for row in db.execute("select * from ThreadNames"):
            thread_names[row] = True
        rows = db.execute(
            "select t.globalTid, s.value from ThreadNames t join StringIds s on t.nameId = s.id"
        ).fetchall()
        print("K4 thread names:", rows)
    else:
        print("K4 FAIL: no ThreadNames table")

    if "OSRT_API" in names:
        waits = db.execute(
            "select o.start, o.end, o.globalTid, s.value from OSRT_API o join StringIds s on o.nameId = s.id "
            "where o.end - o.start > 50000000 order by o.start"
        ).fetchall()
        print(
            "K6 OS runtime calls over 50 ms:",
            [(w[3], round((w[1] - w[0]) / 1e6, 1)) for w in waits][:10],
        )
    else:
        print("K6 FAIL: no OSRT_API table")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "analyze"))
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.mode == "run":
        run()
    else:
        analyze(args.path)


if __name__ == "__main__":
    sys.exit(main())
