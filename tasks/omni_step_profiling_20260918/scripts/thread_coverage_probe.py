"""Which threads a torch profiler started on one thread records.

sglang-omni starts TorchProfiler on the stage control thread while the scheduler,
vocoder and reference threads already run. This probe starts the profiler on the
main thread, runs the same work on a thread that existed before the start and on
one created after it, and counts per thread what the trace holds.

usage: python thread_coverage_probe.py --out DIR
"""

from __future__ import annotations

import argparse
import gzip
import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

ITERS = 20
GRAPH_NODES = 8


def build_graph(device):
    static = torch.zeros(1024, device=device)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(2):
            out = static
            for _ in range(GRAPH_NODES):
                out = out + 1
    torch.cuda.current_stream(device).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = static
        for _ in range(GRAPH_NODES):
            out = out + 1
    torch.cuda.synchronize(device)
    return graph


def work(tag, device, graph, go, done, seen_enabled):
    go.wait()
    seen_enabled[tag] = torch.autograd._profiler_enabled()
    a = torch.randn(256, 256, device=device)
    for _ in range(ITERS):
        with torch.profiler.record_function(f"span_{tag}"):
            a = a @ a
            a = a / a.norm()
            graph.replay()
    torch.cuda.synchronize(device)
    done.set()


def run(config_name, profile_kwargs, device, graph, out_dir):
    go_pre = threading.Event()
    done_pre = threading.Event()
    seen = {}
    pre = threading.Thread(
        target=work,
        args=("pre", device, graph, go_pre, done_pre, seen),
        name="probe-pre",
    )
    pre.start()
    time.sleep(0.2)
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], **profile_kwargs
    )
    prof.start()
    seen["main"] = torch.autograd._profiler_enabled()
    go_post = threading.Event()
    done_post = threading.Event()
    post = threading.Thread(
        target=work,
        args=("post", device, graph, go_post, done_post, seen),
        name="probe-post",
    )
    post.start()
    go_pre.set()
    go_post.set()
    done_pre.wait()
    done_post.wait()
    pre.join()
    post.join()
    main_tag = "main"
    with torch.profiler.record_function(f"span_{main_tag}"):
        a = torch.randn(256, 256, device=device)
        for _ in range(ITERS):
            a = a @ a
            graph.replay()
    torch.cuda.synchronize(device)
    prof.stop()
    path = out_dir / f"{config_name}.trace.json"
    prof.export_chrome_trace(str(path))
    return path, seen, {"pre": pre.native_id, "post": post.native_id}


def summarize(path, native_ids):
    trace = json.loads(Path(path).read_text())
    events = trace.get("traceEvents", [])
    by_tid = defaultdict(Counter)
    launch_tid = {}
    kernels = []
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        tid = str(event.get("tid"))
        args = event.get("args") or {}
        if cat == "kernel":
            kernels.append(args.get("correlation"))
            continue
        if cat == "cuda_runtime" and "aunch" in name:
            launch_tid[args.get("correlation")] = tid
            by_tid[tid][name] += 1
            continue
        if cat == "user_annotation" and name.startswith("span_"):
            by_tid[tid][name] += 1
            continue
        if cat in ("cpu_op", "python_function"):
            by_tid[tid][cat] += 1
    kernel_tids = Counter(launch_tid.get(c, "no-launch-event") for c in kernels)
    ext_tid = {}
    launch_ext = {}
    kernel_ext = []
    for event in events:
        if event.get("ph") != "X":
            continue
        cat = str(event.get("cat", ""))
        args = event.get("args") or {}
        ext = args.get("External id")
        if cat in ("cpu_op", "user_annotation") and ext is not None:
            ext_tid.setdefault(ext, str(event.get("tid")))
        elif cat == "cuda_runtime" and "aunch" in str(event.get("name", "")):
            launch_ext[args.get("correlation")] = ext
        elif cat == "kernel":
            kernel_ext.append(
                ext if ext is not None else launch_ext.get(args.get("correlation"))
            )
    owner = Counter(
        ext_tid.get(ext, "no-op-for-ext") if ext is not None else "no-external-id"
        for ext in kernel_ext
    )
    print(f"kernels by owner via External id -> cpu op tid {dict(owner)}")
    names = {
        str(e.get("tid")): (e.get("args") or {}).get("name")
        for e in events
        if e.get("ph") == "M" and e.get("name") == "thread_name"
    }
    return by_tid, kernel_tids, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", 0)
    graph = build_graph(device)
    print(f"torch {torch.__version__} main native_id {threading.get_native_id()}")
    print(f"graph nodes per replay {GRAPH_NODES}, iters per thread {ITERS}")
    configs = {
        "default": {},
        "stack": {"with_stack": True},
        "all_threads": {
            "experimental_config": torch._C._profiler._ExperimentalConfig(
                profile_all_threads=True
            )
        },
        "all_threads_stack": {
            "with_stack": True,
            "experimental_config": torch._C._profiler._ExperimentalConfig(
                profile_all_threads=True
            ),
        },
    }
    for config_name, profile_kwargs in configs.items():
        path, seen, native_ids = run(config_name, profile_kwargs, device, graph, out_dir)
        by_tid, kernel_tids, names = summarize(path, native_ids)
        print(f"\n== {config_name}")
        print(f"thread native ids {native_ids}")
        print(f"_profiler_enabled seen by thread {seen}")
        for tid, counter in sorted(by_tid.items()):
            print(f"tid {tid} name {names.get(tid)}: {dict(sorted(counter.items()))}")
        print(f"kernels by launching tid {dict(kernel_tids)}")
        with open(path, "rb") as src, gzip.open(f"{path}.gz", "wb") as dst:
            dst.write(src.read())
        path.unlink()


if __name__ == "__main__":
    main()
