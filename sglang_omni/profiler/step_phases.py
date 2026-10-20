# SPDX-License-Identifier: Apache-2.0
"""Wall and thread CPU time per scheduler loop phase, plus the device waits inside them.

Off unless OMNI_STEP_PHASES names an output file. Every two seconds the cumulative totals are appended to it as one
JSON line, keyed by iteration kind and batch size, so a reader takes the difference between two lines. Wall minus
thread CPU is time the scheduler thread did not run: interpreter lock waits, sleeps and blocking calls.
"""

from __future__ import annotations

import json
import os
import time

FLUSH_INTERVAL_S = 2.0


class StepPhases:
    def __init__(self, path: str) -> None:
        self.path = path
        # "kind|batch size" -> {"iterations": n, "phases": {phase: [count, wall, cpu]}, "waits": {wait: [count, s]}}
        self.totals: dict[str, dict] = {}
        self.phases: list[tuple[str, float, float]] = []
        self.waits: dict[str, list[float]] = {}
        self.wall = 0.0
        self.cpu = 0.0
        self.last_flush = time.perf_counter()

    def begin(self) -> None:
        self.phases.clear()
        self.waits.clear()
        self.wall = time.perf_counter()
        self.cpu = time.thread_time()

    def mark(self, phase: str) -> None:
        wall = time.perf_counter()
        cpu = time.thread_time()
        self.phases.append((phase, wall - self.wall, cpu - self.cpu))
        self.wall = wall
        self.cpu = cpu

    def wait(self, name: str, seconds: float) -> None:
        totals = self.waits.setdefault(name, [0, 0.0])
        totals[0] += 1
        totals[1] += seconds

    def end(self, kind: str, batch_size: int) -> None:
        self.mark("tail")
        entry = self.totals.setdefault(
            f"{kind}|{batch_size}", {"iterations": 0, "phases": {}, "waits": {}}
        )
        entry["iterations"] += 1
        for phase, wall, cpu in self.phases:
            totals = entry["phases"].setdefault(phase, [0, 0.0, 0.0])
            totals[0] += 1
            totals[1] += wall
            totals[2] += cpu
        for name, (count, seconds) in self.waits.items():
            totals = entry["waits"].setdefault(name, [0, 0.0])
            totals[0] += count
            totals[1] += seconds
        if self.wall - self.last_flush >= FLUSH_INTERVAL_S:
            self.last_flush = self.wall
            with open(self.path, "a", encoding="utf-8") as output:
                output.write(
                    json.dumps({"time": time.time(), "totals": self.totals}) + "\n"
                )
        else:
            pass


STEP_PHASES = (
    StepPhases(os.environ["OMNI_STEP_PHASES"])
    if os.environ.get("OMNI_STEP_PHASES")
    else None
)


def batch_kind(batch) -> str:
    if batch is None:
        return "idle"
    elif batch.forward_mode is None:
        return "unset"
    elif batch.forward_mode.is_mixed():
        return "mixed"
    elif batch.forward_mode.is_extend():
        return "extend"
    else:
        return "decode"
