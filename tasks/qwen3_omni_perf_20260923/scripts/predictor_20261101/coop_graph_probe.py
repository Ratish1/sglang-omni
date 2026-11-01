"""Whether a Triton kernel launched cooperatively, with a device-wide barrier between phases, captures in a CUDA graph
and replays correctly on this stack (the precondition of one persistent kernel per predictor pass, option O4).

The kernel runs PHASES phases over one buffer: in phase p every program adds 1 to its own slot, then waits at a
barrier (an atomic arrival counter per phase, each program spinning on a volatile load until every program arrived),
then reads its neighbour's slot, which must already hold p + 1 if the barrier held. One program per SM, launched with
launch_cooperative_grid=True. Eager first, then captured in torch.cuda.graph and replayed REPLAYS times with the
counters reset by the graph itself; any wrong neighbour read is counted. Also timed: one barrier's cost, from the
replay time at 64 phases against 1 phase.

usage: python3 coop_graph_probe.py
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

REPLAYS = 200


@triton.jit
def phases_kernel(slots, counters, errors, PHASES: tl.constexpr):
    program = tl.program_id(0)
    programs = tl.num_programs(0)
    neighbour = (program + 1) % programs
    for phase in range(PHASES):
        value = tl.load(slots + program)
        tl.store(slots + program, value + 1)
        tl.atomic_add(counters + phase, 1, sem="release")
        arrived = tl.atomic_add(counters + phase, 0, sem="acquire")
        while arrived < programs:
            arrived = tl.atomic_add(counters + phase, 0, sem="acquire")
        seen = tl.load(slots + neighbour, volatile=True)
        tl.atomic_add(errors, (seen != phase + 1).to(tl.int32))


def run(phases: int, programs: int, device: torch.device) -> tuple[int, float, int]:
    slots = torch.zeros(programs, dtype=torch.int32, device=device)
    counters = torch.zeros(phases, dtype=torch.int32, device=device)
    errors = torch.zeros(1, dtype=torch.int32, device=device)

    def launch():
        slots.zero_()
        counters.zero_()
        phases_kernel[(programs,)](slots, counters, errors, PHASES=phases, launch_cooperative_grid=True)

    launch()
    torch.cuda.synchronize()
    eager_errors = int(errors.item())
    errors.zero_()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    errors.zero_()
    with torch.cuda.graph(graph):
        launch()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPLAYS):
        graph.replay()
    end.record()
    end.synchronize()
    return eager_errors, start.elapsed_time(end) * 1000 / REPLAYS, int(errors.item())


def main() -> None:
    device = torch.device("cuda")
    programs = torch.cuda.get_device_properties(device).multi_processor_count
    print(f"{torch.cuda.get_device_name()}, {programs} SMs, triton {triton.__version__}, torch {torch.__version__}")
    times = {}
    for phases in (1, 64):
        eager_errors, replay_us, graph_errors = run(phases, programs, device)
        times[phases] = replay_us
        print(
            f"phases {phases:3d}: eager errors {eager_errors}, graph replay {replay_us:8.2f} us, "
            f"graph errors over {REPLAYS} replays {graph_errors}"
        )
    print(f"one barrier, from 64 against 1 phase: {(times[64] - times[1]) / 63:.2f} us")


if __name__ == "__main__":
    main()
