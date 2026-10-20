"""S0 of DESIGN_VOCODER_STANDARD_20261019.md, the device half of the two stream question: with no host waits in the
way, does a first hop's graph work on a stream one priority higher finish near its own device time while a follow-up
step's graphs run?

two_stream_probe_c20261019.py runs whole steps, which hold today's host waits (pageable uploads, the sine
multipliers, the final .cpu()); there the first hop finished with the follow-up step. This replays only the captured
graphs each step uses, so the device's scheduling is all that is left: the follow-up step's Flow graph (a 4 row
final over 2,400 frames, or a 16 row second hop) and its HiFT decode graph on vocoder A's stream at default priority,
and a first hop's prefix graph and HiFT decode graph on vocoder B (each vocoder its own pool, as the standard needs).
The first hop is enqueued 5 ms after the follow-up's graphs; its latency runs from an event recorded at that moment on
an idle stream to its last graph's end, measured on the device.

usage (in the container, one card): CUDA_VISIBLE_DEVICES=<card> python3 two_stream_graphs_c20261019.py [--rounds N]
"""

from __future__ import annotations

import argparse
import bisect
import statistics
import time

import torch

from sglang_omni.scheduling.streaming_vocoder import vocoder_decode_stream_priority

import two_stream_probe_c20261019 as steps

LAUNCH_DELAY_S = 0.005


def recorded_graphs(vocoder, step) -> list:
    """The graphs one run of step replays, in order, found by wrapping each runner's replay."""
    used: list = []
    runners = [
        vocoder.flow.prefix_cuda_graph_runner,
        vocoder.flow.whole_history_cuda_graph_runner,
        vocoder.hift_decode_cuda_graph_runner,
    ]
    originals = []
    for runner in runners:
        if runner is None:
            continue
        captured = runner.captured.values() if isinstance(runner.captured, dict) else runner.captured
        for entry in captured:
            graph = entry.graph
            originals.append((graph, graph.replay))

            def replay(graph=graph, original=graph.replay) -> None:
                used.append(graph)
                original()

            graph.replay = replay
    try:
        with torch.inference_mode(), vocoder.stream_context:
            step()
        torch.cuda.synchronize()
    finally:
        for graph, original in originals:
            graph.replay = original
    return used


def replay_all(graphs: list) -> None:
    for graph in graphs:
        graph.replay()


def summary(values: list[float]) -> str:
    ordered = sorted(values)
    return (
        f"p50 {ordered[len(ordered) // 2]:7.2f} p90 {ordered[int(0.9 * (len(ordered) - 1))]:7.2f}"
        f" mean {statistics.mean(values):7.2f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    follow_vocoder = steps.build(20.0)
    first_vocoder = steps.build(2.0)
    follow_stream = torch.cuda.Stream(priority=0)
    idle_stream = torch.cuda.Stream(priority=0)
    priority = vocoder_decode_stream_priority(torch.cuda)
    variants = {
        "same stream": follow_stream,
        "second stream, same priority": torch.cuda.Stream(priority=0),
        "second stream, one priority higher": torch.cuda.Stream(priority=priority),
    }
    first_graphs = recorded_graphs(first_vocoder, steps.FirstHop(first_vocoder))
    print(f"first hop replays {len(first_graphs)} graphs; first hop stream priority {priority}", flush=True)

    def device_time(stream, graphs) -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            start.record()
            replay_all(graphs)
            end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for name, make_step in (("final 4 rows", steps.FinalStep), ("hop 16 rows", steps.HopStep)):
        follow_graphs = recorded_graphs(follow_vocoder, make_step(follow_vocoder))
        for _ in range(3):
            device_time(follow_stream, follow_graphs)
            device_time(variants["second stream, one priority higher"], first_graphs)
        alone_follow = [device_time(follow_stream, follow_graphs) for _ in range(args.rounds)]
        alone_first = [device_time(variants["second stream, one priority higher"], first_graphs) for _ in range(args.rounds)]
        print(
            f"\n{name}: follow-up replays {len(follow_graphs)} graphs, alone {summary(alone_follow)};"
            f" first hop alone {summary(alone_first)}",
            flush=True,
        )
        for label, first_stream in variants.items():
            latencies, follows, spans = [], [], []
            for _ in range(args.rounds):
                torch.cuda.synchronize()
                follow_start, follow_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                ready, first_end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                with torch.cuda.stream(follow_stream):
                    follow_start.record()
                    replay_all(follow_graphs)
                    follow_end.record()
                time.sleep(LAUNCH_DELAY_S)
                with torch.cuda.stream(idle_stream):
                    ready.record()
                with torch.cuda.stream(first_stream):
                    replay_all(first_graphs)
                    first_end.record()
                torch.cuda.synchronize()
                latencies.append(ready.elapsed_time(first_end))
                follows.append(follow_start.elapsed_time(follow_end))
                spans.append(max(follow_start.elapsed_time(follow_end), follow_start.elapsed_time(first_end)))
            print(
                f"  first hop 5 ms in, {label}: latency {summary(latencies)}; follow-up {summary(follows)};"
                f" both {summary(spans)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
