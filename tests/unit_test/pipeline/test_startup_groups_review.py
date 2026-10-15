"""Independent probe of the runner's supervision across stage groups."""

import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from sglang_omni.pipeline import mp_runner, stage_workers
from sglang_omni.pipeline.stage_workers import StageGroup, StageWorkerProcessSpec
from tests.unit_test.pipeline.test_ipc import StubCoordinator, make_config


@dataclass
class ProbeProcess:
    alive: bool = True
    exitcode: int | None = None
    pid: int = 123

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout: float) -> None:
        pass


@pytest.mark.asyncio
async def test_ready_group_death_while_another_group_progresses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A completed wait must not leave its worker unmonitored during startup."""
    now_seconds = 0.0
    processes = [ProbeProcess(), ProbeProcess()]
    counters = [SimpleNamespace(value=0), SimpleNamespace(value=0)]

    class ReadyEvent:
        def __init__(self, index: int) -> None:
            self.index = index

        def is_set(self) -> bool:
            return self.index == 0 or now_seconds >= 30

        def wait(self, timeout: float) -> bool:
            nonlocal now_seconds
            now_seconds += timeout
            counters[1].value += 1
            if now_seconds >= 3:
                processes[0].alive = False
                processes[0].exitcode = 1
            else:
                pass
            return self.is_set()

    groups = [
        StageGroup(name, [StageWorkerProcessSpec(name, [])])
        for name in ("ready-encoder", "compiling-thinker")
    ]
    context = multiprocessing.get_context("spawn")
    for index, group in enumerate(groups):
        group._processes = [
            processes[index]
        ]  # noqa: leading-underscore  # production field
        group.ready_events = [ReadyEvent(index)]
        group.startup_progress = [counters[index]]
        group.startup_error_channels = [context.Queue()]

    monkeypatch.setattr(
        stage_workers, "time", SimpleNamespace(monotonic=lambda: now_seconds)
    )
    monkeypatch.setattr(StageGroup, "spawn", lambda self, context: None)
    monkeypatch.setattr(mp_runner, "Coordinator", StubCoordinator)
    monkeypatch.setattr(mp_runner, "build_stage_groups", lambda *args, **kwargs: groups)
    configuration = make_config(tmp_path)
    configuration.weight_share = "off"
    configuration.mps = "off"
    runner = mp_runner.MultiProcessPipelineRunner(configuration)
    with pytest.raises(RuntimeError, match="ready-encoder"):
        await runner.start(timeout=10)

    print({"death_at_seconds": 3, "detected_at_seconds": now_seconds})
    assert now_seconds == 30
    assert not runner.started
    assert all(not process.is_alive() for process in processes)
    assert list(tmp_path.iterdir()) == []
