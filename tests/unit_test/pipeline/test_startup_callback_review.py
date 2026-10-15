import logging
import multiprocessing
import threading

import pytest
import torch
from torch._dynamo.callback import callback_handler

from sglang_omni.pipeline import stage_workers
from sglang_omni.pipeline.stage_workers import StageLaunchConfig, StageWorkerProcessSpec


@pytest.mark.accelerator
def test_compile_progress_stops_at_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    progress = multiprocessing.get_context("spawn").Value("q", 0)
    ready = threading.Event()
    observations: dict[str, int] = {}
    callbacks_before = list(callback_handler.end_callbacks)

    def operation(inputs: torch.Tensor) -> torch.Tensor:
        return inputs.cos() * inputs.sin()

    compiled = torch.compile(operation, fullgraph=True, dynamic=False)

    class CompilingStage:
        name = "compile-probe"
        running = False

        async def start(self) -> None:
            before = progress.value
            compiled(torch.ones(128, device="cuda"))
            observations["start_compile_progress"] = progress.value - before
            self.running = True

        async def run(self) -> None:
            observations["ready_counter"] = progress.value
            assert ready.is_set()
            compiled(torch.ones(256, device="cuda"))
            torch.cuda.synchronize()
            observations["serving_counter"] = progress.value

        async def stop(self) -> None:
            self.running = False

    def construct_stage(
        stage_spec: StageLaunchConfig, logger: logging.Logger, *, local_dispatcher
    ) -> CompilingStage:
        before = progress.value
        compiled(torch.ones(64, device="cuda"))
        observations["constructor_compile_progress"] = progress.value - before
        return CompilingStage()

    monkeypatch.setattr(stage_workers, "construct_stage", construct_stage)
    specification = StageWorkerProcessSpec(
        process_name="compile-probe",
        stage_specs=[StageLaunchConfig(stage_name="compile-probe")],
        startup_progress=progress,
    )
    stage_workers.run_process(specification, ready, logging.getLogger(__name__))
    assert observations["constructor_compile_progress"] > 0
    assert observations["start_compile_progress"] > 0
    assert observations["serving_counter"] == observations["ready_counter"]
    assert callback_handler.end_callbacks == callbacks_before
    print(observations)
