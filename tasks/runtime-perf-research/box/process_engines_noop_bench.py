"""Host cost part 2 adds to a process holding one SGLang engine: the turn calls an OmniScheduler loop pass makes.

usage: PYTHONPATH=<part 2 tree> taskset -c <cpu> python3 process_engines_noop_bench.py

One engine registered, the scheduler thread inside hold() as run() enters it. Per loop pass the sync path makes 5
yield_turn calls (omni_scheduler.py) and 2 more around the forward (base.py), one released() around the step-end wait,
and reads is_shared twice (lookahead choice, step-end sync). Prints ns per call and per pass, median of 7 repeats.
"""

import statistics
import time

from sglang.srt.runtime_context import get_context

from sglang_omni.model_runner.process_engines import process_engines

CALLS = 200_000
REPEATS = 7


def per_call_ns(body) -> float:
    runs = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        body()
        runs.append((time.perf_counter_ns() - start) / CALLS)
    return statistics.median(runs)


def yields() -> None:
    for _ in range(CALLS):
        process_engines().yield_turn()


def releases() -> None:
    for _ in range(CALLS):
        with process_engines().released():
            pass


def shared_reads() -> None:
    for _ in range(CALLS):
        process_engines().is_shared


def empty() -> None:
    for _ in range(CALLS):
        pass


engines = process_engines()
engines.engine_contexts.append(get_context())
with engines.hold(engines.engine_contexts[0], priority=0):
    loop_ns = per_call_ns(empty)
    yield_ns = per_call_ns(yields) - loop_ns
    release_ns = per_call_ns(releases) - loop_ns
    read_ns = per_call_ns(shared_reads) - loop_ns
per_pass_ns = 7 * yield_ns + release_ns + 2 * read_ns
print(f"one engine: yield_turn {yield_ns:.0f} ns, released() {release_ns:.0f} ns, is_shared {read_ns:.0f} ns")
print(f"per sync loop pass (7 yields, 1 release, 2 reads): {per_pass_ns / 1000:.2f} us")
