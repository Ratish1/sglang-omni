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
  K5 python-gil: the main thread waits for the GIL while a worker spins
  K6 OS runtime: a 100 ms queue timeout is recorded with its duration
  K7 node tracing inflates graph gaps: gaps between replayed nodes against the eager chain
     of the same kernels (read, not gated: gaps are never read as launch cost)
  K8 two processes: correlation ids repeat across processes, and keying by (pid, id)
     attributes every child kernel to its own process's range
  K9 time slicing: two processes each run 150 sleep kernels of 1 ms at once on one GPU;
     the context residency (--gpuctxsw) of each is about its 150 ms of work, and kernels
     that ran while the other process had a kernel in flight are longer than 1 ms, so a
     kernel's duration includes time its context was switched out

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
SLEEP_KERNELS = 150
RESTORE_START = 8
SAVE_END = 7


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def name_os_thread(name: str) -> None:
    libc = ctypes.CDLL("libc.so.6")
    libc.prctl(15, name.encode()[:15], 0, 0, 0)


def cycles_per_ms() -> int:
    import torch

    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    torch.cuda.synchronize()
    return int(10_000_000 / start.elapsed_time(end))


def child(index: int, barrier, cycles: int) -> None:
    import torch

    nvtx = torch.cuda.nvtx
    x = torch.randn(256, 256, device="cuda")
    for i in range(30):
        nvtx.range_push(f"ka.proc{index} i={i}")
        torch.add(x, float(i))
        nvtx.range_pop()
    torch.cuda.synchronize()
    barrier.wait()
    nvtx.range_push(f"ka.slice{index}")
    for _ in range(SLEEP_KERNELS):
        torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    nvtx.range_pop()


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

    cycles = cycles_per_ms()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    children = [
        context.Process(target=child, args=(i, barrier, cycles)) for i in range(2)
    ]
    for process in children:
        process.start()
    for process in children:
        process.join()
    print(f"done {float(z):.3f} cycles/ms {cycles}", flush=True)


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
            "select start, end, globalTid, text, textId from NVTX_EVENTS where end is not null"
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
        f"K5 {'pass' if waits > 50e6 else 'FAIL'}: main thread GIL wait inside the contended range {waits / 1e6:.1f} ms"
    )

    queue_wait = (
        [
            (end - start) / 1e6
            for start, end, tid, name_id in db.execute(
                "select start, end, globalTid, nameId from OSRT_API"
            )
            if 90e6 < end - start < 200e6
        ]
        if "OSRT_API" in tables
        else []
    )
    print(
        f"K6 {'pass' if queue_wait else 'FAIL'}: OS runtime waits of 90 to 200 ms {queue_wait[:4]}"
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
    sleeps = collections.defaultdict(list)
    for start, end, corr, name_id, node, global_pid in kernels:
        if (
            "sleep" in strings.get(name_id, "").lower()
            and pid_of(global_pid) in children.values()
        ):
            sleeps[pid_of(global_pid)].append((start, end))
    if len(sleeps) != 2:
        print(f"K9 FAIL: sleep kernels found for {len(sleeps)} processes")
        return
    t0 = min(s for v in sleeps.values() for s, _ in v)
    t1 = max(e for v in sleeps.values() for _, e in v)
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
    residency = {}
    for key, slices in raw.items():
        best, best_ns = None, 0
        for pid, spans in sleeps.items():
            covered = sum(
                max(0, min(e, se) - max(s, ss)) for s, e in slices for ss, se in spans
            )
            if covered > best_ns:
                best, best_ns = pid, covered
        if best is not None:
            residency[best] = residency.get(best, 0) + sum(e - s for s, e in slices)
    other = {
        pid: [iv for p, v in sleeps.items() if p != pid for iv in v] for pid in sleeps
    }
    stretched, alone = [], []
    for pid, spans in sleeps.items():
        for s, e in spans:
            overlap = sum(max(0, min(e, oe) - max(s, os_)) for os_, oe in other[pid])
            (stretched if overlap > 0.5 * (e - s) else alone).append((e - s) / 1e6)
    expected_ms = SLEEP_KERNELS * 1.0
    k9 = len(residency) == 2 and all(
        0.85 * expected_ms <= value / 1e6 <= 1.25 * expected_ms
        for value in residency.values()
    )
    print(
        f"K9 {'pass' if k9 else 'FAIL'}: residency per process "
        f"{[round(v / 1e6, 1) for v in residency.values()]} ms against {expected_ms:.0f} ms of work each; "
        f"window {(t1 - t0) / 1e6:.1f} ms; sleep kernel ms alone p50 "
        f"{statistics.median(alone) if alone else float('nan'):.3f} (n {len(alone)}), with the other in "
        f"flight p50 {statistics.median(stretched) if stretched else float('nan'):.3f} (n {len(stretched)})"
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
