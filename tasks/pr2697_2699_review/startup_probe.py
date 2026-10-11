import asyncio
import json
import queue
from dataclasses import dataclass
from unittest.mock import patch

from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import StageGroup, StageWorkerProcessSpec


@dataclass
class Clock:
    seconds: float = 0.0

    def monotonic(self) -> float:
        return self.seconds


@dataclass
class Counter:
    value: int = 0


@dataclass
class Process:
    alive: bool
    exitcode: int | None
    checks: int = 0

    def is_alive(self) -> bool:
        self.checks += 1
        return self.alive


@dataclass
class ReadyEvent:
    clock: Clock
    counter: Counter
    ready_at: float | None
    progresses: bool

    def is_set(self) -> bool:
        return self.ready_at is not None and self.clock.seconds >= self.ready_at

    def wait(self, timeout: float) -> bool:
        self.clock.seconds += timeout
        if self.progresses:
            self.counter.value += 1
        else:
            pass
        return self.is_set()


async def probe(second_rank: str) -> dict[str, str | float | list[int]]:
    clock = Clock()
    counters = [Counter(), Counter()]
    events = [ReadyEvent(clock, counters[0], 4.0, True)]
    processes = [Process(True, None)]
    specifications = [StageWorkerProcessSpec("rank-0", [])]
    if second_rank != "none":
        events.append(ReadyEvent(clock, counters[1], None, False))
        processes.append(
            Process(second_rank != "dead", 1 if second_rank == "dead" else None)
        )
        specifications.append(StageWorkerProcessSpec("rank-1", []))
    else:
        pass
    group = StageGroup("thinker", specifications)
    group.ready_events = events
    group.startup_progress = counters[: len(events)]
    group.startup_error_channels = [queue.Queue() for _ in events]
    group._processes = processes
    with patch.object(stage_workers.time, "monotonic", clock.monotonic):
        try:
            await group.wait_ready(timeout=1.0)
        except (TimeoutError, RuntimeError) as error:
            outcome = type(error).__name__
            message = str(error)
        else:
            outcome = "ready"
            message = ""
    return {
        "second_rank": second_rank,
        "outcome": outcome,
        "virtual_seconds": clock.seconds,
        "liveness_checks": [process.checks for process in processes],
        "message": message,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            [asyncio.run(probe(case)) for case in ("none", "dead", "stalled")], indent=2
        )
    )
