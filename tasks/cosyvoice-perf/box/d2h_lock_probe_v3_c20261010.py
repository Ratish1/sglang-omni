# SPDX-License-Identifier: Apache-2.0
"""Does a thread waiting for its stream hold up another thread's kernel launches, when it waits for
HiFT's real step graph? Bare, on the tree it runs from and the real checkpoint.

A vocoder thread replays work on its stream and waits for it through one of two copies of the
samples:
  pageable   samples.cpu(), what hift_group does
  sync_first the stream synchronized, then samples.cpu()
The work is one of
  kernels    a graph of 1,000 small one-block kernels, about 20 ms
  memnodes   the same graph with a 4 byte memset and a 4 byte device copy after every 100th kernel
  hift R F   HiFT's captured step graph for the tier of R rows and F window frames, as the factory
             builds the runner
An AR thread meanwhile launches one small kernel at a time on its own stream (synchronized every 64
launches) and times each launch call on the host. Per case: the vocoder's wall per round, AR
launches, the launch time spent over 1 ms, the longest launch.

    PYTHONPATH=/workspace/t_hg python d2h_lock_probe_v3_c20261010.py
"""

from __future__ import annotations

import sys
import threading
import time

import torch

sys.path.insert(0, "/data/c20261010h/in")
from hift_graph_bench_c20261010 import build_runner  # noqa: E402

from sglang_omni.models.fun_cosyvoice3 import stages  # noqa: E402

ROUNDS = 40
GRAPH_KERNELS = 1000


def sleep_cycles(milliseconds: float) -> int:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    end.synchronize()
    return int(10_000_000 * milliseconds / start.elapsed_time(end))


def sleep_graph(cycles: int, *, memory_nodes: bool) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    source = torch.ones(1, device="cuda")
    target = torch.empty(1, device="cuda")
    with torch.cuda.graph(graph):
        for index in range(GRAPH_KERNELS):
            torch.cuda._sleep(cycles // GRAPH_KERNELS)
            if memory_nodes and index % 100 == 0:
                target.zero_()
                target.copy_(source)
    torch.cuda.synchronize()
    return graph


def run_case(name: str, replay, samples: torch.Tensor, variant: str) -> None:
    device = samples.device
    vocoder_stream, ar_stream = torch.cuda.Stream(), torch.cuda.Stream()
    launches: list[int] = []
    stop = threading.Event()

    def ar_loop() -> None:
        counter = torch.zeros(1, device=device)
        with torch.cuda.stream(ar_stream):
            while not stop.is_set():
                for _ in range(64):
                    started = time.perf_counter_ns()
                    counter.add_(1)
                    launches.append(time.perf_counter_ns() - started)
                ar_stream.synchronize()

    ar_thread = threading.Thread(target=ar_loop)
    ar_thread.start()
    time.sleep(0.2)
    walls = []
    with torch.cuda.stream(vocoder_stream):
        for _ in range(ROUNDS):
            started = time.perf_counter_ns()
            replay()
            if variant == "sync_first":
                vocoder_stream.synchronize()
            host = samples.cpu()
            walls.append(time.perf_counter_ns() - started)
            assert host.shape == samples.shape
    stop.set()
    ar_thread.join()
    slow = [t for t in launches if t > 1_000_000]
    print(
        f"{name:14s} {variant:10s} vocoder wall p50 {sorted(walls)[len(walls) // 2] / 1e6:5.1f} ms | "
        f"AR launches {len(launches):7d} over 1 ms {len(slow):4d} "
        f"sum {sum(slow) / 1e6:7.1f} ms max {max(launches) / 1e6:6.2f} ms",
        flush=True,
    )


def main() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    cycles = sleep_cycles(20)
    checkpoint_dir = stages.resolve_checkpoint("FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    flow, hift = stages.load_cosyvoice3_flow_hift(
        checkpoint_dir, device="cuda", autocast_dtype=torch.bfloat16
    )
    vocoder = stages.CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    print(f"tree={stages.__file__}", flush=True)
    with torch.inference_mode():
        runner = build_runner(vocoder, max_rows=16)
        cases = []
        for memory_nodes, name in ((False, "kernels"), (True, "memnodes")):
            graph = sleep_graph(cycles, memory_nodes=memory_nodes)
            cases.append((name, graph.replay, torch.randn(1, 96 * 480, device=device)))
        for rows, frames in ((1, 96), (16, 240)):
            step = runner.captured[(rows, frames)]
            cases.append((f"hift {rows} {frames}", step.graph.replay, step.samples[:rows]))
        for name, replay, samples in cases:
            for variant in ("pageable", "sync_first"):
                run_case(name, replay, samples, variant)


if __name__ == "__main__":
    main()
