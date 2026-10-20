"""Whether CUDA green contexts (sgl_kernel.spatial, as SGLang's PD multiplexing uses them) isolate a latency bound
graph from a concurrent compute bound stream on one H100, and what each side then costs.

A: a decode like CUDA graph, 24 layers of small GEMMs at 32 rows (the AR's shape: hidden 896, MLP 4864), replayed
   in a loop; its replay time is the latency of interest.
B: a vocoder like load, large bf16 GEMMs (persistent cuBLAS kernels that hold every SM they get) looped on its
   own stream.

Arms, each measured as A's mean replay time over N replays and B's GEMMs per second:
  alone          A on a plain stream, B idle (and B alone on a plain stream)
  shared         A and B on two plain streams
  priority       A on the highest priority stream, B on a plain one
  green a/b      A and B on the two streams of a green context pair splitting the SMs a/b, the graph captured on
                 a plain stream and replayed on the green stream; and captured on the green stream itself

usage: CUDA_VISIBLE_DEVICES=3 python3 green_context_probe_c20261020.py [--replays 400]
"""

from __future__ import annotations

import argparse
import threading
import time

import torch

HIDDEN, MLP, LAYERS, ROWS = 896, 4864, 24, 32
B_SIZE = 8192


def decode_like_weights(device: torch.device) -> list[tuple[torch.Tensor, ...]]:
    return [
        (
            torch.randn(HIDDEN, 3 * HIDDEN // 2, device=device, dtype=torch.bfloat16) * 0.02,
            torch.randn(HIDDEN, HIDDEN, device=device, dtype=torch.bfloat16) * 0.02,
            torch.randn(HIDDEN, 2 * MLP, device=device, dtype=torch.bfloat16) * 0.02,
            torch.randn(MLP, HIDDEN, device=device, dtype=torch.bfloat16) * 0.02,
        )
        for _ in range(LAYERS)
    ]


def decode_like_step(x: torch.Tensor, weights: list[tuple[torch.Tensor, ...]]) -> torch.Tensor:
    for qkv, out, gate_up, down in weights:
        h = torch.nn.functional.rms_norm(x, (HIDDEN,))
        h = (h @ qkv)[:, :HIDDEN] @ out
        x = x + h
        g, u = (torch.nn.functional.rms_norm(x, (HIDDEN,)) @ gate_up).chunk(2, dim=-1)
        x = x + (torch.nn.functional.silu(g) * u) @ down
    return x


def capture(stream: torch.cuda.Stream, x: torch.Tensor, weights) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            decode_like_step(x, weights)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = decode_like_step(x, weights)
    torch.cuda.synchronize()
    return graph, out


def replay_ms(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream, replays: int) -> float:
    with torch.cuda.stream(stream):
        for _ in range(20):
            graph.replay()
        stream.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        for _ in range(replays):
            graph.replay()
        end.record(stream)
        end.synchronize()
    return start.elapsed_time(end) / replays


class Load:
    """B: large GEMMs looped on its stream until stopped; counts the GEMMs done."""

    def __init__(self, stream: torch.cuda.Stream, device: torch.device) -> None:
        self.stream = stream
        self.a = torch.randn(B_SIZE, B_SIZE, device=device, dtype=torch.bfloat16)
        self.b = torch.randn(B_SIZE, B_SIZE, device=device, dtype=torch.bfloat16)
        self.stop = threading.Event()
        self.count = 0
        self.seconds = 0.0

    def run(self) -> None:
        started = time.perf_counter()
        with torch.cuda.stream(self.stream):
            while not self.stop.is_set():
                for _ in range(4):
                    torch.mm(self.a, self.b)
                self.stream.synchronize()
                self.count += 4
        self.seconds = time.perf_counter() - started


def with_load(load_stream: torch.cuda.Stream, device, measure) -> tuple[float, float]:
    load = Load(load_stream, device)
    thread = threading.Thread(target=load.run)
    thread.start()
    time.sleep(0.5)
    value = measure()
    load.stop.set()
    thread.join()
    return value, load.count / load.seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays", type=int, default=400)
    args = parser.parse_args()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    x = torch.randn(ROWS, HIDDEN, device=device, dtype=torch.bfloat16)
    weights = decode_like_weights(device)
    plain_a = torch.cuda.Stream(device)
    plain_b = torch.cuda.Stream(device)
    high = torch.cuda.Stream(device, priority=-3)
    graph, _ = capture(plain_a, x, weights)

    alone_ms = replay_ms(graph, plain_a, args.replays)
    b_alone = Load(plain_b, device)
    thread = threading.Thread(target=b_alone.run)
    thread.start()
    time.sleep(3.0)
    b_alone.stop.set()
    thread.join()
    b_alone_rate = b_alone.count / b_alone.seconds
    print(f"sm count {torch.cuda.get_device_properties(device).multi_processor_count}, torch {torch.__version__}")
    print(f"alone: A {alone_ms:.3f} ms a replay; B alone {b_alone_rate:.1f} GEMMs/s")

    shared_ms, shared_rate = with_load(plain_b, device, lambda: replay_ms(graph, plain_a, args.replays))
    print(f"shared: A {shared_ms:.3f} ms ({shared_ms / alone_ms:.2f}x), B {shared_rate:.1f} GEMMs/s ({shared_rate / b_alone_rate:.2f}x)")
    high_graph, _ = capture(high, x, weights)
    high_ms, high_rate = with_load(plain_b, device, lambda: replay_ms(high_graph, high, args.replays))
    print(f"priority: A {high_ms:.3f} ms ({high_ms / alone_ms:.2f}x), B {high_rate:.1f} GEMMs/s ({high_rate / b_alone_rate:.2f}x)")

    try:
        from sgl_kernel import spatial
    except ImportError as error:
        print(f"sgl_kernel.spatial unavailable: {error}")
        return
    total = spatial.get_sm_available(0)
    print(f"sgl_kernel.spatial: {total} SMs available")
    for a_sms in (16, 24, 32, 48):
        b_sms = total - a_sms
        green_a, green_b = spatial.create_greenctx_stream_by_value(a_sms, b_sms, 0)
        for capture_on in ("plain", "green"):
            try:
                if capture_on == "plain":
                    run_graph = graph
                else:
                    run_graph, _ = capture(green_a, x, weights)
                green_alone = replay_ms(run_graph, green_a, args.replays)
                green_ms, green_rate = with_load(
                    green_b, device, lambda: replay_ms(run_graph, green_a, args.replays)
                )
                print(
                    f"green {a_sms}/{b_sms}, captured on {capture_on}: A alone {green_alone:.3f} ms, "
                    f"under B {green_ms:.3f} ms ({green_ms / alone_ms:.2f}x of full-card alone), "
                    f"B {green_rate:.1f} GEMMs/s ({green_rate / b_alone_rate:.2f}x)"
                )
            except RuntimeError as error:
                print(f"green {a_sms}/{b_sms}, captured on {capture_on}: {type(error).__name__}: {error}")


main()
