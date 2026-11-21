#!/usr/bin/env python3
"""One card shared by a talker-like step loop and a thinker-like decode loop: two processes (the GPU time-slices their
contexts) against one process (one context, a stream each, with or without the talker on the top priority).

The talker step replays a short graph of small-batch matmuls, waits for it on an event, then runs Python for about the
talker's measured share of the interpreter lock. The thinker replays a longer graph back to back with Python between.
Each graph is captured on its own stream. Reported: the talker's step wall time (launch to its sync, and the whole step)
and the thinker's steps a second, per mode, over --seconds after a warm second.

usage: python3 context_share_bench.py --device 1 [--seconds 20] [--talker-kernels 500] [--thinker-kernels 1200]
"""

import argparse
import json
import multiprocessing as mp
import statistics
import threading
import time

import torch

CAPTURE_LOCK = threading.Lock()


def build_graph(
    kernels: int, batch: int, hidden: int, stream: torch.cuda.Stream, seed: int
):
    with CAPTURE_LOCK:
        return capture_graph(kernels, batch, hidden, stream, seed)


def capture_graph(
    kernels: int, batch: int, hidden: int, stream: torch.cuda.Stream, seed: int
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    weights = [
        torch.randn(
            hidden, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.02
        for _ in range(8)
    ]
    x = torch.randn(
        batch, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        y = x
        for index in range(kernels):
            y = torch.tanh(y @ weights[index % len(weights)])
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            y = x
            for index in range(kernels):
                y = torch.tanh(y @ weights[index % len(weights)])
    return graph, (weights, x, y)


def spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    total = 0
    while time.perf_counter() < end:
        total += 1


def quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(ordered[len(ordered) // 2], 3),
        "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 3),
        "p99": round(ordered[int(0.99 * (len(ordered) - 1))], 3),
    }


def talker_loop(args, priority: int, start: float, stop: float, out: dict) -> None:
    torch.cuda.set_device(args.device)
    stream = torch.cuda.Stream(priority=priority)
    graph, keep = build_graph(
        args.talker_kernels, args.talker_rows, args.talker_hidden, stream, 1
    )
    event = torch.cuda.Event()
    while time.time() < start:
        time.sleep(0.001)
    waits, steps = [], []
    with torch.cuda.stream(stream):
        while time.time() < stop:
            began = time.perf_counter()
            graph.replay()
            event.record(stream)
            event.synchronize()
            launched_and_done = time.perf_counter()
            spin(args.talker_host_ms / 1e3)
            if time.time() > start + args.warm:
                waits.append((launched_and_done - began) * 1e3)
                steps.append((time.perf_counter() - began) * 1e3)
            else:
                pass
    out["talker_device_wait_ms"] = quantiles(waits)
    out["talker_step_ms"] = quantiles(steps)
    del keep


def thinker_loop(args, priority: int, start: float, stop: float, out: dict) -> None:
    torch.cuda.set_device(args.device)
    stream = torch.cuda.Stream(priority=priority)
    graph, keep = build_graph(
        args.thinker_kernels, args.thinker_rows, args.thinker_hidden, stream, 2
    )
    event = torch.cuda.Event()
    while time.time() < start:
        time.sleep(0.001)
    count, durations = 0, []
    with torch.cuda.stream(stream):
        while time.time() < stop:
            began = time.perf_counter()
            graph.replay()
            event.record(stream)
            event.synchronize()
            spin(args.thinker_host_ms / 1e3)
            if time.time() > start + args.warm:
                count += 1
                durations.append((time.perf_counter() - began) * 1e3)
            else:
                pass
    out["thinker_steps_per_s"] = round(count / (args.seconds - args.warm), 1)
    out["thinker_step_ms"] = quantiles(durations)
    del keep


def process_entry(role: str, args, start: float, stop: float, queue) -> None:
    out: dict = {}
    if role == "talker":
        talker_loop(args, 0, start, stop, out)
    else:
        thinker_loop(args, 0, start, stop, out)
    queue.put(out)


def run_mode(mode: str, args) -> dict:
    torch.cuda.set_device(args.device)
    low, high = torch.cuda.Stream.priority_range()
    result: dict = {"mode": mode}
    start = time.time() + args.setup
    stop = start + args.seconds
    if mode == "talker_alone":
        talker_loop(args, 0, start, stop, result)
    elif mode == "thinker_alone":
        thinker_loop(args, 0, start, stop, result)
    elif mode == "two_processes":
        context = mp.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(target=process_entry, args=(role, args, start, stop, queue))
            for role in ("talker", "thinker")
        ]
        for process in processes:
            process.start()
        for _ in processes:
            result.update(queue.get())
        for process in processes:
            process.join()
    else:
        talker_priority = high if mode == "one_process_talker_first" else low
        threads = [
            threading.Thread(
                target=talker_loop, args=(args, talker_priority, start, stop, result)
            ),
            threading.Thread(
                target=thinker_loop, args=(args, low, start, stop, result)
            ),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--warm", type=float, default=2.0)
    parser.add_argument("--setup", type=float, default=25.0)
    parser.add_argument("--talker-kernels", type=int, default=500)
    parser.add_argument("--talker-rows", type=int, default=4)
    parser.add_argument("--talker-hidden", type=int, default=1024)
    parser.add_argument("--talker-host-ms", type=float, default=0.8)
    parser.add_argument("--thinker-kernels", type=int, default=1200)
    parser.add_argument("--thinker-rows", type=int, default=4)
    parser.add_argument("--thinker-hidden", type=int, default=2048)
    parser.add_argument("--thinker-host-ms", type=float, default=1.5)
    parser.add_argument(
        "--modes",
        default="talker_alone,thinker_alone,two_processes,one_process_equal,one_process_talker_first",
    )
    args = parser.parse_args()
    for mode in args.modes.split(","):
        print(json.dumps(run_mode(mode, args)), flush=True)


if __name__ == "__main__":
    main()
