# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from sglang.srt.runtime_context import ParallelContext, RuntimeContext

from sglang_omni.model_runner.process_engines import ProcessEngines
from sglang_omni.scheduling.sglang_backend import prompt_hidden as prompt_hidden_module
from sglang_omni.scheduling.sglang_backend.prompt_hidden import StagedPromptHidden
from tests.unit_test.fixtures.prompt_hidden import cuda_prompt_hidden_store


class TurnRecordingCopyEvent:
    def __init__(self, engines: ProcessEngines) -> None:
        self.engines = engines
        self.is_turn_held_while_waiting: bool | None = None

    def query(self) -> bool:
        return False

    def synchronize(self) -> None:
        self.is_turn_held_while_waiting = self.engines.is_turn_held


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host rows need CUDA")
def test_a_read_returns_the_rows_its_slots_were_last_computed_with() -> None:
    req_to_token = torch.zeros((2, 6), dtype=torch.int32, device="cuda")
    with cuda_prompt_hidden_store(slot_count=16, req_to_token=req_to_token) as store:
        first_rows = torch.arange(6.0, device="cuda").view(3, 2)
        store.stage(torch.tensor([3, 4, 5], device="cuda"), first_rows)
        second_rows = torch.full((1, 2), 7.0, device="cuda")
        store.stage(torch.tensor([9], device="cuda"), second_rows)
        reused_rows = torch.full((1, 2), 8.0, device="cuda")
        store.stage(torch.tensor([5], device="cuda"), reused_rows)
        req_to_token[1, 1:4] = torch.tensor([3, 4, 9], dtype=torch.int32)
        req_to_token[0, 2] = 5

        assert torch.equal(
            store.read(1, torch.tensor([1, 2, 3])),
            torch.cat([first_rows[:2], second_rows]).cpu(),
        )
        assert torch.equal(store.read(0, torch.tensor([2])), reused_rows.cpu())


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host rows need CUDA")
def test_a_read_in_a_shared_process_waits_for_staged_copies_outside_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read waited for copies staged behind long prefills while it held the turn, so the talker could not run."""
    engines = ProcessEngines()
    thinker_context, talker_context = (
        RuntimeContext(parallel=ParallelContext()) for _ in range(2)
    )
    engines.engine_contexts.extend([thinker_context, talker_context])
    monkeypatch.setattr(prompt_hidden_module, "process_engines", lambda: engines)
    req_to_token = torch.zeros((1, 6), dtype=torch.int32, device="cuda")
    req_to_token[0, 2] = 5
    copy_done = TurnRecordingCopyEvent(engines)
    with cuda_prompt_hidden_store(slot_count=16, req_to_token=req_to_token) as store:
        store.staged.append(
            StagedPromptHidden(
                slot_ids=torch.tensor([5]),
                hidden_states=torch.full((1, 2), 3.0),
                copy_done=copy_done,
            )
        )
        with engines.hold(thinker_context, priority=0):
            rows = store.read(0, torch.tensor([2]))

    assert copy_done.is_turn_held_while_waiting is False
    assert torch.equal(rows, torch.full((1, 2), 3.0))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned host rows need CUDA")
def test_a_destroyed_store_leaves_its_rows_free_to_register_again() -> None:
    req_to_token = torch.zeros((1, 6), dtype=torch.int32, device="cuda")
    with cuda_prompt_hidden_store(slot_count=16, req_to_token=req_to_token) as store:
        rows = store.hidden_states
    cudart = torch.cuda.cudart()

    assert int(cudart.cudaHostRegister(rows.data_ptr(), rows.nbytes, 0)) == 0
    assert int(cudart.cudaHostUnregister(rows.data_ptr())) == 0
