"""Known answer test for every Nsight feature omni_census.py relies on (run on the box).

Run mode builds workloads whose attribution is known by construction; analyze mode reads
the sqlite export and prints pass or FAIL per check with the measured values. Nothing in a
census is trusted until this passes on the node's nsys, driver, container and torch.

  K1 node trace: every replayed kernel appears once per replay with a graph node id
  K2 capture labels: through originalGraphNodeId and the node creation events, every
     replayed kernel resolves to the capture range it was created in (the matmul to
     ka.cap.A, the relu and the sum to ka.cap.B), each kernel name to one range (the
     census G join, verbatim)
  K3 eager kernels launched from a late thread resolve to that thread's range
  K5 python-gil: the main thread waits for the GIL while a worker spins (at least one 5 ms
     switch interval of waiting inside the contended range)
  K6 OS runtime: the main thread's 100 ms queue timeout inside ka.queue.wait is recorded
     with its duration
  K7 node tracing inflates graph gaps: gaps between replayed nodes against the eager chain
     of the same kernels (read, not gated: gaps are never read as launch cost)
  K8 two processes: correlation ids repeat across processes, and keying by (pid, id)
     attributes every child kernel to its own process's range
  K9 time slicing: two processes each run the same chain of 100 bf16 8192 cube matmuls at
     once on one GPU, after the parent ran the chain alone; the context residency
     (--gpuctxsw) of each child is about its work (100 times the matmul alone), the two
     chains take about twice the work in wall time, and a matmul that ran while the other
     process had one in flight is longer than alone, so a kernel's duration includes time
     its context was switched out; prints the residency slice length (the timeslice). A
     clock spin kernel (torch.cuda._sleep) cannot answer this: it ends on elapsed cycles
     whether or not its context is resident
  K10 read, stream priority: one process runs two matmul streams at once, first both at the
     default priority (ka.equal), then one at the device's highest priority (ka.prio); prints
     how many hardware contexts (switch-record context ids) hold the process's kernel starts in
     each, so a priority stream that the GPU schedules as its own time-sliced context shows up

usage:
  nsys profile -o ka --trace=cuda,nvtx,osrt,python-gil --cuda-graph-trace=node \\
      --gpuctxsw=true --sample=none --cpuctxsw=none python3 omni_known_answer.py run
  nsys export --type sqlite -o ka.sqlite ka.nsys-rep
  python3 omni_known_answer.py analyze ka.sqlite
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import multiprocessing
import queue
import sqlite3
import statistics
import threading
import time

REPLAYS = 50
PUSH_POP = 59
MATMULS = 100
MATMUL_SIZE = 8192
RESTORE_START = 8
SAVE_END = 7


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def name_os_thread(name: str) -> None:
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(15, name.encode()[:15], 0, 0, 0)


def matmul_chain(label: str) -> None:
    import torch

    nvtx = torch.cuda.nvtx
    a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    nvtx.range_push(label)
    for _ in range(MATMULS):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    nvtx.range_pop()


def matmul_pair(label: str, high_priority: bool) -> None:
    import torch

    nvtx = torch.cuda.nvtx
    least, greatest = torch.cuda.Stream.priority_range()
    first = torch.cuda.Stream(priority=least)
    second = torch.cuda.Stream(priority=greatest if high_priority else least)
    a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    nvtx.range_push(label)
    for _ in range(MATMULS // 2):
        with torch.cuda.stream(first):
            torch.matmul(a, a)
        with torch.cuda.stream(second):
            torch.matmul(a, a)
    torch.cuda.synchronize()
    nvtx.range_pop()


def child(index: int, barrier) -> None:
    import torch

    nvtx = torch.cuda.nvtx
    x = torch.randn(256, 256, device="cuda")
    for i in range(30):
        nvtx.range_push(f"ka.proc{index} i={i}")
        torch.add(x, float(i))
        nvtx.range_pop()
    torch.matmul(x.bfloat16(), x.bfloat16())
    torch.cuda.synchronize()
    barrier.wait()
    matmul_chain(f"ka.slice{index}")


def run() -> None:
    import torch

    nvtx = torch.cuda.nvtx
    x = torch.randn(1024, 1024, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            torch.relu(x @ x).sum()
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
    for i in range(REPLAYS):
        nvtx.range_push(f"ka.replay i={i}")
        graph.replay()
        nvtx.range_pop()
    torch.cuda.synchronize()
    for i in range(REPLAYS):
        nvtx.range_push(f"ka.chain i={i}")
        torch.relu(x @ x).sum()
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
        while time.perf_counter() < deadline:
            pass

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

    matmul_chain("ka.warm")
    matmul_chain("ka.alone")
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    children = [context.Process(target=child, args=(i, barrier)) for i in range(2)]
    for process in children:
        process.start()
    for process in children:
        process.join()
    matmul_pair("ka.equal", high_priority=False)
    matmul_pair("ka.prio", high_priority=True)
    print(f"done {float(z):.3f}", flush=True)


def analyze(path: str) -> None:
    db = sqlite3.connect(path)
    tables = {
        r[0] for r in db.execute("select name from sqlite_master where type='table'")
    }
    strings = dict(db.execute("select id, value from StringIds"))
    ranges = collections.defaultdict(list)
    for start, end, tid, text, text_id in db.execute(
        "select start, end, globalTid, text, textId from NVTX_EVENTS where eventType = ?",
        (PUSH_POP,),
    ):
        label = text if text is not None else strings.get(text_id, "")
        if end is not None:
            ranges[label.split(" ")[0]].append((start, end, tid, label))
    api = {}
    for start, end, tid, corr, name_id in db.execute(
        "select start, end, globalTid, correlationId, nameId from CUPTI_ACTIVITY_KIND_RUNTIME"
    ):
        api[(pid_of(tid), corr)] = (start, end, tid, strings.get(name_id, ""))
    corr_pids = collections.defaultdict(set)
    for pid, corr in api:
        corr_pids[corr].add(pid)

    def enclosing(tid, t, kind):
        for start, end, range_tid, label in ranges.get(kind, []):
            if range_tid == tid and start <= t <= end:
                return label
        return None

    kernels = db.execute(
        "select start, end, correlationId, demangledName, graphNodeId, globalPid "
        "from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ).fetchall()
    replayed = collections.Counter()
    for start, end, corr, name_id, node, global_pid in kernels:
        launch = api.get((pid_of(global_pid), corr))
        if launch and enclosing(launch[2], launch[0], "ka.replay"):
            replayed[(strings.get(name_id, "")[:50], node is not None)] += 1
    k1 = bool(replayed) and all(
        v == REPLAYS and has_node for (_, has_node), v in replayed.items()
    )
    print(f"K1 {'pass' if k1 else 'FAIL'}: kernels under replays {dict(replayed)}")

    original, created = {}, {}
    for start, tid, node, orig in (
        db.execute(
            "select start, globalTid, graphNodeId, originalGraphNodeId from CUDA_GRAPH_NODE_EVENTS"
        )
        if "CUDA_GRAPH_NODE_EVENTS" in tables
        else []
    ):
        if orig is not None:
            original[(pid_of(tid), node)] = orig
        elif (pid_of(tid), node) not in created:
            created[(pid_of(tid), node)] = (start, tid)
    resolved = collections.Counter()
    for start, end, corr, name_id, node, global_pid in kernels:
        pid = pid_of(global_pid)
        launch = api.get((pid, corr))
        if (
            node is None
            or not launch
            or not enclosing(launch[2], launch[0], "ka.replay")
        ):
            continue
        orig = node
        while (pid, orig) in original:
            orig = original[(pid, orig)]
        origin = created.get((pid, orig))
        label = None
        if origin is not None:
            label = enclosing(origin[1], origin[0], "ka.cap.A") or enclosing(
                origin[1], origin[0], "ka.cap.B"
            )
        resolved[(strings.get(name_id, "")[:40], label)] += 1
    labels_by_name = collections.defaultdict(set)
    for (name, label), _ in resolved.items():
        labels_by_name[name].add(label)
    k2 = (
        bool(resolved)
        and all(label is not None for (_, label) in resolved)
        and {label for (_, label) in resolved} == {"ka.cap.A", "ka.cap.B"}
        and all(len(labels) == 1 for labels in labels_by_name.values())
    )
    print(
        f"K2 {'pass' if k2 else 'FAIL'}: replayed kernel -> capture range {dict(resolved)}"
    )

    eager = sum(
        1
        for start, end, corr, name_id, node, global_pid in kernels
        if (launch := api.get((pid_of(global_pid), corr)))
        and enclosing(launch[2], launch[0], "ka.eager")
    )
    print(
        f"K3 {'pass' if eager == 20 else 'FAIL'}: eager kernels under ka.eager {eager} of 20"
    )

    contended = ranges.get("ka.main.contended", [])
    waits = 0
    if contended:
        start, end, tid, _ = contended[0]
        for w_start, w_end, w_tid, text, text_id in db.execute(
            "select start, end, globalTid, text, textId from NVTX_EVENTS "
            "where end is not null and eventType = ?",
            (PUSH_POP,),
        ):
            label = text if text is not None else strings.get(text_id, "")
            if (
                label.startswith("Waiting for GIL")
                and w_tid == tid
                and w_start < end
                and w_end > start
            ):
                waits += w_end - w_start
    print(
        f"K5 {'pass' if waits > 5e6 else 'FAIL'}: main thread GIL wait inside the contended range {waits / 1e6:.1f} ms"
    )

    waits_range = ranges.get("ka.queue.wait", [])
    queue_wait = []
    if waits_range and "OSRT_API" in tables:
        w_start, w_end, w_tid, _ = waits_range[0]
        queue_wait = [
            (end - start) / 1e6
            for start, end, tid, name_id in db.execute(
                "select start, end, globalTid, nameId from OSRT_API"
            )
            if tid == w_tid
            and start >= w_start
            and end <= w_end
            and 90e6 < end - start < 200e6
        ]
    print(
        f"K6 {'pass' if queue_wait else 'FAIL'}: OS runtime wait of 90 to 200 ms on the main "
        f"thread inside ka.queue.wait {queue_wait[:4]}"
    )

    def span_gaps(kind):
        gaps = []
        for start, end, tid, label in ranges.get(kind, []):
            inside = sorted(
                (k_start, k_end)
                for k_start, k_end, corr, name_id, node, global_pid in kernels
                if (launch := api.get((pid_of(global_pid), corr)))
                and launch[2] == tid
                and start <= launch[0] <= end
            )
            gaps += [b[0] - a[1] for a, b in zip(inside, inside[1:])]
        return gaps

    replay_gaps, chain_gaps = span_gaps("ka.replay"), span_gaps("ka.chain")
    if replay_gaps and chain_gaps:
        print(
            f"K7 read: gap between kernels, node-traced replay p50 {statistics.median(replay_gaps) / 1e3:.2f} us, "
            f"eager chain p50 {statistics.median(chain_gaps) / 1e3:.2f} us"
        )

    collisions = sum(1 for pids in corr_pids.values() if len(pids) > 1)
    children = {}
    for label_kind in ("ka.proc0", "ka.proc1"):
        for start, end, tid, label in ranges.get(label_kind, []):
            children.setdefault(label_kind, pid_of(tid))
    own = wrong = 0
    for start, end, corr, name_id, node, global_pid in kernels:
        pid = pid_of(global_pid)
        if pid not in children.values():
            continue
        launch = api.get((pid, corr))
        kind = "ka.proc0" if pid == children.get("ka.proc0") else "ka.proc1"
        if launch and enclosing(launch[2], launch[0], kind):
            own += 1
        elif launch and any(
            enclosing(launch[2], launch[0], k) for k in ("ka.proc0", "ka.proc1")
        ):
            wrong += 1
    k8 = len(children) == 2 and own == 60 and wrong == 0
    print(
        f"K8 {'pass' if k8 else 'FAIL'}: correlation ids shared by several processes {collisions}; "
        f"child kernels in their own process's range {own} of 60, in the other's {wrong}"
    )

    if "GPU_CONTEXT_SWITCH_EVENTS" not in tables:
        print("K9 FAIL: no GPU_CONTEXT_SWITCH_EVENTS (profile with --gpuctxsw=true)")
        return

    def kernels_in(kind):
        found = collections.defaultdict(list)
        for start, end, corr, name_id, node, global_pid in kernels:
            launch = api.get((pid_of(global_pid), corr))
            label = launch and enclosing(launch[2], launch[0], kind)
            if label:
                found[pid_of(global_pid)].append((start, end))
        return found

    alone = [e - s for spans in kernels_in("ka.alone").values() for s, e in spans]
    chains = {}
    for kind in ("ka.slice0", "ka.slice1"):
        for pid, spans in kernels_in(kind).items():
            chains[pid] = spans
    if len(chains) != 2 or not alone:
        print(
            f"K9 FAIL: matmul chains found for {len(chains)} children, alone {len(alone)}"
        )
        return
    matmul_alone = statistics.median(alone)
    expected = MATMULS * matmul_alone
    t0 = min(s for v in chains.values() for s, _ in v)
    t1 = max(e for v in chains.values() for _, e in v)
    open_since, raw = {}, collections.defaultdict(list)
    for timestamp, tag, context, gpu in db.execute(
        "select timestamp, tag, contextId, gpuId from GPU_CONTEXT_SWITCH_EVENTS order by gpuId, timestamp, seqNo"
    ):
        key = (gpu, context)
        if tag == RESTORE_START:
            open_since[key] = timestamp
        elif tag == SAVE_END and key in open_since:
            start = open_since.pop(key)
            if timestamp > t0 and start < t1:
                raw[key].append((max(start, t0), min(timestamp, t1)))
    residency, slice_lengths = {}, []
    for key, slices in raw.items():
        best, best_count = None, 0
        for pid, spans in chains.items():
            count = sum(1 for s, e in slices for ss, _ in spans if s <= ss <= e)
            if count > best_count:
                best, best_count = pid, count
        if best is not None:
            residency[best] = residency.get(best, 0) + sum(e - s for s, e in slices)
            slice_lengths += [e - s for s, e in slices]
    other = {
        pid: [iv for p, v in chains.items() if p != pid for iv in v] for pid in chains
    }
    contended = []
    for pid, spans in chains.items():
        for s, e in spans:
            overlap = sum(max(0, min(e, oe) - max(s, os_)) for os_, oe in other[pid])
            if overlap > 0.5 * (e - s):
                contended.append(e - s)
    k9 = (
        len(residency) == 2
        and all(
            0.85 * expected <= value <= 1.25 * expected for value in residency.values()
        )
        and (t1 - t0) >= 1.7 * expected
        and bool(contended)
        and statistics.median(contended) > 1.5 * matmul_alone
    )
    print(
        f"K9 {'pass' if k9 else 'FAIL'}: matmul alone {matmul_alone / 1e6:.3f} ms, work per child "
        f"{expected / 1e6:.1f} ms; residency per child {[round(v / 1e6, 1) for v in residency.values()]} ms; "
        f"both chains {(t1 - t0) / 1e6:.1f} ms of wall; contended matmul p50 "
        f"{statistics.median(contended) / 1e6 if contended else float('nan'):.3f} ms (n {len(contended)}); "
        f"residency slice p50 {statistics.median(slice_lengths) / 1e6 if slice_lengths else float('nan'):.3f} ms, "
        f"max {max(slice_lengths) / 1e6 if slice_lengths else float('nan'):.3f} ms"
    )

    switch_rows = db.execute(
        "select timestamp, tag, contextId, gpuId, (globalPid >> 24) & 16777215 "
        "from GPU_CONTEXT_SWITCH_EVENTS order by gpuId, timestamp, seqNo"
    ).fetchall()
    for label in ("ka.equal", "ka.prio"):
        spans = ranges.get(label, [])
        if not spans:
            print(f"K10 read: no {label} range")
            continue
        w_start, w_end, w_tid, _ = spans[0]
        starts = sorted(
            start
            for start, end, corr, name_id, node, global_pid in kernels
            if pid_of(global_pid) == pid_of(w_tid) and w_start <= start <= w_end
        )
        open_since, holders = {}, collections.Counter()
        for timestamp, tag, context, gpu, host_pid in switch_rows:
            key = (gpu, context, host_pid)
            if tag == RESTORE_START:
                open_since[key] = timestamp
            elif tag == SAVE_END and key in open_since:
                begin = open_since.pop(key)
                if timestamp > w_start and begin < w_end:
                    inside = sum(
                        1 for k_start in starts if begin <= k_start <= timestamp
                    )
                    if inside:
                        holders[key] += inside
        print(
            f"K10 read: {label}: {len(starts)} kernel starts held by "
            f"{len(holders)} contexts {[(k[1], k[2], v) for k, v in holders.most_common(4)]}"
        )


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
    main()
