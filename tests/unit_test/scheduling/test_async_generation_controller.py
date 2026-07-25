# SPDX-License-Identifier: Apache-2.0
"""State-machine tests for the async-generation typed core (plan Phase 1).

These pin the transaction lifecycle against the design invariants:
I2 (pending depth), I6 (commit mask set once, before commit), I7 (KV validity
independent of output acceptance), I11 (which failures are reachable from
which lifecycle point).
"""

from __future__ import annotations

import pytest

from sglang_omni.scheduling.async_generation import (
    AsyncGenerationController,
    FailurePolicy,
    GenerationProtocol,
    InvariantViolation,
    LaggedGenerationContract,
    PendingGenerationStep,
    PendingRow,
    PendingStepState,
    RelayHandle,
    RetractionPolicy,
    StepKVClaim,
)

S = PendingStepState


def _row(i: int, serial: int = 1) -> PendingRow:
    return PendingRow(
        rid=f"r{i}",
        admission_serial=serial,
        req_pool_idx=i,
        input_key=100 + i,
        relay_handle=RelayHandle(pool_row=i),
        kv_claim=StepKVClaim(
            prior_committed_len=4, prior_allocated_len=4, num_new_slots=1
        ),
    )


def _step(step_id: int = 1, n_rows: int = 2) -> PendingGenerationStep:
    return PendingGenerationStep(
        step_id=step_id,
        batch_snapshot=object(),
        rows=[_row(i) for i in range(n_rows)],
    )


# ---------------------------------------------------------------------------
# Step lifecycle
# ---------------------------------------------------------------------------


def test_success_path_transitions():
    step = _step()
    for state in (S.LAUNCHED, S.DEVICE_READY, S.COMMITTING, S.COMMITTED):
        step.transition(state)
    assert step.state is S.COMMITTED


@pytest.mark.parametrize(
    ("path", "illegal"),
    [
        pytest.param((), S.DEVICE_READY, id="planned-skips-launch"),
        pytest.param((), S.COMMITTED, id="planned-skips-everything"),
        pytest.param((S.LAUNCHED,), S.COMMITTING, id="launched-skips-ready"),
        pytest.param((S.LAUNCHED,), S.COMMITTED, id="launched-skips-commit"),
        pytest.param(
            (S.LAUNCHED, S.DEVICE_READY, S.COMMITTING),
            S.ROLLED_BACK,
            id="mid-commit-rollback-is-not-exactly-once",
        ),
    ],
)
def test_illegal_transitions_raise(path, illegal):
    step = _step()
    for state in path:
        step.transition(state)
    with pytest.raises(InvariantViolation, match="illegal transition"):
        step.transition(illegal)


@pytest.mark.parametrize(
    "terminal", [S.COMMITTED, S.ROLLED_BACK, S.FATAL], ids=lambda s: s.value
)
def test_terminal_states_accept_nothing(terminal):
    step = _step()
    routes = {
        S.COMMITTED: (S.LAUNCHED, S.DEVICE_READY, S.COMMITTING, S.COMMITTED),
        S.ROLLED_BACK: (S.ROLLED_BACK,),
        S.FATAL: (S.LAUNCHED, S.FATAL),
    }
    for state in routes[terminal]:
        step.transition(state)
    for target in S:
        with pytest.raises(InvariantViolation):
            step.transition(target)


def test_failure_protocol_edges():
    # pre-enqueue failure: exact provisional rollback from PLANNED
    step = _step()
    step.transition(S.ROLLED_BACK)
    # post-enqueue failure without a proven rollback: FATAL from LAUNCHED
    step = _step()
    step.transition(S.LAUNCHED)
    step.transition(S.FATAL)
    # commit-phase failure after mutation started: FATAL, never retried
    step = _step()
    for state in (S.LAUNCHED, S.DEVICE_READY, S.COMMITTING):
        step.transition(state)
    step.transition(S.FATAL)


# ---------------------------------------------------------------------------
# Masks: commit acceptance and KV validity are independent (I6/I7)
# ---------------------------------------------------------------------------


def test_commit_mask_set_once_at_device_ready_only():
    step = _step(n_rows=2)
    with pytest.raises(InvariantViolation, match="commit_mask"):
        step.set_commit_mask([True, True])  # too early: PLANNED
    step.transition(S.LAUNCHED)
    with pytest.raises(InvariantViolation, match="commit_mask"):
        step.set_commit_mask([True, True])  # still too early: LAUNCHED
    step.transition(S.DEVICE_READY)
    step.set_commit_mask([True, False])
    assert step.commit_mask == (True, False)
    with pytest.raises(InvariantViolation, match="reset"):
        step.set_commit_mask([True, True])


def test_kv_cacheable_mask_finalized_after_launch_and_independent():
    # the terminal-overrun shape: output does not commit, KV is cacheable
    step = _step(n_rows=2)
    step.transition(S.LAUNCHED)
    step.set_kv_cacheable_mask([True, True])
    step.transition(S.DEVICE_READY)
    step.set_commit_mask([True, False])
    assert step.kv_cacheable_mask == (True, True)
    assert step.commit_mask == (True, False)


def test_mask_length_must_match_rows():
    step = _step(n_rows=2)
    step.transition(S.LAUNCHED)
    with pytest.raises(InvariantViolation, match="length"):
        step.set_kv_cacheable_mask([True])


# ---------------------------------------------------------------------------
# KV claim bookkeeping
# ---------------------------------------------------------------------------


def test_kv_claim_rejects_inconsistent_lengths():
    with pytest.raises(InvariantViolation):
        StepKVClaim(prior_committed_len=5, prior_allocated_len=4, num_new_slots=1)
    with pytest.raises(InvariantViolation):
        StepKVClaim(prior_committed_len=1, prior_allocated_len=2, num_new_slots=-1)


# ---------------------------------------------------------------------------
# Controller: pending depth and identities (I2)
# ---------------------------------------------------------------------------


def test_pending_depth_is_zero_or_one():
    c = AsyncGenerationController(GenerationProtocol.LAGGED)
    assert c.pending_depth == 0
    first = _step(step_id=c.next_step_id())
    first.transition(S.LAUNCHED)
    assert c.replace_pending(first) is None
    assert c.pending_depth == 1
    second = _step(step_id=c.next_step_id())
    second.transition(S.LAUNCHED)
    assert c.replace_pending(second) is first  # launch-first hands back prev
    assert c.pending_depth == 1
    assert c.take_pending() is second
    assert c.pending_depth == 0
    assert c.take_pending() is None


def test_only_launched_steps_may_become_pending():
    c = AsyncGenerationController(GenerationProtocol.LAGGED)
    with pytest.raises(InvariantViolation, match="must be LAUNCHED"):
        c.replace_pending(_step(step_id=c.next_step_id()))  # still PLANNED


def test_sync_controller_rejects_pending_steps():
    c = AsyncGenerationController(GenerationProtocol.SYNC)
    step = _step(step_id=1)
    step.transition(S.LAUNCHED)
    with pytest.raises(InvariantViolation, match="LAGGED"):
        c.replace_pending(step)


def test_barrier_requires_depth_zero():
    c = AsyncGenerationController(GenerationProtocol.LAGGED)
    c.assert_barrier_clear()
    step = _step(step_id=c.next_step_id())
    step.transition(S.LAUNCHED)
    c.replace_pending(step)
    with pytest.raises(InvariantViolation, match="barrier"):
        c.assert_barrier_clear()
    c.take_pending()
    c.assert_barrier_clear()


def test_identities_are_monotonic():
    c = AsyncGenerationController(GenerationProtocol.LAGGED)
    assert [c.next_step_id() for _ in range(3)] == [1, 2, 3]
    assert [c.next_admission_serial() for _ in range(3)] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def _contract(**overrides) -> LaggedGenerationContract:
    kwargs = dict(
        supported_forward_modes=frozenset({"DECODE", "MIXED", "EXTEND"}),
        publishes_next_input=True,
        commit_is_masked=True,
        retraction_policy=RetractionPolicy.REBUILD,
        failure_policy=FailurePolicy.FATAL,
    )
    kwargs.update(overrides)
    return LaggedGenerationContract(**kwargs)


def test_contract_valid_shape():
    assert _contract().is_valid


@pytest.mark.parametrize(
    ("overrides", "needle"),
    [
        ({"supported_forward_modes": frozenset()}, "empty"),
        ({"publishes_next_input": False}, "publish"),
        ({"commit_is_masked": False}, "mask"),
        ({"max_launch_batch_size": 0}, "max_launch_batch_size"),
    ],
)
def test_contract_invalid_shapes(overrides, needle):
    errors = _contract(**overrides).validation_errors()
    assert errors and any(needle in e for e in errors)


def test_contract_has_no_implicit_policies():
    with pytest.raises(TypeError):
        LaggedGenerationContract(  # missing retraction/failure policy
            supported_forward_modes=frozenset({"DECODE"}),
            publishes_next_input=True,
            commit_is_masked=True,
        )
