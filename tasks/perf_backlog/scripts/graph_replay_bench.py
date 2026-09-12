"""Device front end check: replay a CUDA graph of tiny dependent kernels.

Usage: python graph_replay_bench.py [--device cuda:0] [--nodes 1000] [--replays 500]

Captures a chain of small elementwise adds on a 256 element tensor as one CUDA graph
and replays it, then runs the same chain eagerly. Reports the time per replay, the time
per graph node and the time per eager launch. The graph replay time depends only on the
device's ability to dispatch tiny dependent kernels, not on the host, so it separates a
slowed device from a slowed host. Run it once on a session that is known good to record
the reference for the box; the Qwen3-TTS predictor graph (1062 nodes, real kernels)
replays in about 4 ms on an idle H100 at 16 rows, so a 1000 node chain of trivial adds
is expected well under that.
"""

import argparse
import statistics
import time

import torch


def chain(x: torch.Tensor, nodes: int) -> torch.Tensor:
    y = x
    for _ in range(nodes):
        y = y + 1
    return y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    print(f"device {torch.cuda.get_device_name(device)} torch {torch.__version__}")
    x = torch.zeros(256, device=device)
    stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(stream):
        for _ in range(3):
            chain(x, args.nodes)
    torch.cuda.synchronize(device)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        chain(x, args.nodes)
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize(device)

    replay_ms = []
    for _ in range(args.rounds):
        start = time.perf_counter()
        for _ in range(args.replays):
            graph.replay()
        torch.cuda.synchronize(device)
        replay_ms.append((time.perf_counter() - start) * 1e3 / args.replays)

    eager_us = []
    with torch.cuda.stream(stream):
        for _ in range(args.rounds):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.replays // 10):
                chain(x, args.nodes)
            torch.cuda.synchronize(device)
            eager_us.append(
                (time.perf_counter() - start) * 1e6 / (args.replays // 10 * args.nodes)
            )

    per_replay = statistics.median(replay_ms)
    print(
        f"graph of {args.nodes} nodes: {per_replay:.3f} ms per replay, "
        f"{per_replay * 1e3 / args.nodes:.2f} us per node "
        f"(rounds {', '.join(f'{v:.3f}' for v in replay_ms)})"
    )
    print(
        f"eager chain: {statistics.median(eager_us):.2f} us per launch, host and device together"
    )


if __name__ == "__main__":
    main()
