# SPDX-License-Identifier: Apache-2.0
"""Typed core for lagged (one-step-behind) generation scheduling.

Design: tasks/async_generation_scheduler_first_principles_plan_20260725.md.

A stage runs one immutable :class:`GenerationProtocol` for its whole
lifetime. Under ``LAGGED``, the scheduler plans and launches forward step
``F_(k+1)`` from a relay-published input before step ``F_k``'s result is
committed — upstream SGLang overlap semantics, carried by Omni-owned
machinery. This module owns the state that makes that lag explicit:

- :class:`OmniGenerationRelay` — the single owner of the launched-but-
  uncommitted next-input value (ends ``ScheduleBatch.output_ids``'s multiple
  duties);
- :class:`PendingGenerationStep` / :class:`PendingRow` /
  :class:`StepKVClaim` — the scheduler transaction: what was launched, which
  rows it owns, what it claimed, and what may commit;
- :class:`AsyncGenerationController` — pending-depth (0 or 1) ownership and
  legal lifecycle transitions.

The controller owns state transitions and assertions only. It must not own
request admission policy, sampling, model-specific journals, or the prefix
cache — those stay with pinned SGLang and the model runners.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch


class InvariantViolation(RuntimeError):
    """A lagged-scheduling invariant was broken.

    Production policy on raising this is fail-closed: the affected stage
    stops; nothing attempts a best-effort allocator repair and continues.
    """


class GenerationProtocol(enum.Enum):
    """The stage-lifetime scheduling protocol, fixed at construction.

    ``SYNC``: every generation result is committed before the next batch is
    planned (the existing normal event loop). ``LAGGED``: planning runs one
    result behind at every batch size. There is no per-epoch or per-batch-size
    selection and no runtime edge between the two — a stage whose measured
    ``bs=1`` lagged overhead is unacceptable ships ``SYNC``, statically.
    """

    SYNC = "sync"
    LAGGED = "lagged"


class PendingStepState(enum.Enum):
    """Lifecycle of one launched generation step (the scheduler transaction).

    ``PLANNED -> LAUNCHED -> DEVICE_READY -> COMMITTING -> COMMITTED`` is the
    success path; ``ROLLED_BACK`` and ``FATAL`` are terminal failure states.
    See ``_LEGAL_TRANSITIONS`` for which failures are reachable from where.
    """

    PLANNED = "planned"
    LAUNCHED = "launched"
    DEVICE_READY = "device_ready"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FATAL = "fatal"


#: Legal step transitions, each edge tied to the failure protocol:
#:  PLANNED   -> LAUNCHED     device enqueue succeeded
#:  PLANNED   -> ROLLED_BACK  pre-enqueue failure: exact provisional-claim
#:                            rollback (the only initially supported
#:                            request-scoped rollback)
#:  LAUNCHED  -> DEVICE_READY completion event passed
#:  LAUNCHED  -> ROLLED_BACK  post-enqueue failure AND the runner contract
#:                            declares a tested ROLLBACK (gated by the caller)
#:  LAUNCHED  -> FATAL        post-enqueue failure without a proven rollback
#:  DEVICE_READY -> COMMITTING / ROLLED_BACK / FATAL
#:  COMMITTING -> COMMITTED   the no-fail mutation phase finished
#:  COMMITTING -> FATAL       an exception after mutation started: commit may
#:                            be partial; retrying is not exactly-once
_LEGAL_TRANSITIONS: dict[PendingStepState, frozenset[PendingStepState]] = {
    PendingStepState.PLANNED: frozenset(
        {PendingStepState.LAUNCHED, PendingStepState.ROLLED_BACK}
    ),
    PendingStepState.LAUNCHED: frozenset(
        {
            PendingStepState.DEVICE_READY,
            PendingStepState.ROLLED_BACK,
            PendingStepState.FATAL,
        }
    ),
    PendingStepState.DEVICE_READY: frozenset(
        {
            PendingStepState.COMMITTING,
            PendingStepState.ROLLED_BACK,
            PendingStepState.FATAL,
        }
    ),
    PendingStepState.COMMITTING: frozenset(
        {PendingStepState.COMMITTED, PendingStepState.FATAL}
    ),
    PendingStepState.COMMITTED: frozenset(),
    PendingStepState.ROLLED_BACK: frozenset(),
    PendingStepState.FATAL: frozenset(),
}


class RetractionPolicy(enum.Enum):
    """What memory-pressure retraction may do to a request of this stage.

    ``REBUILD``: state is reconstructible from prompt plus committed output;
    the request may be reset and requeued. ``ABORT``: it is not (current
    Higgs/ZONOS2 feedback state); the request is terminally aborted rather
    than corruptly requeued. There is deliberately no implicit default.
    """

    REBUILD = "rebuild"
    ABORT = "abort"


class FailurePolicy(enum.Enum):
    """What a post-enqueue failure may do.

    ``ROLLBACK`` is only legal with a tested, idempotent rollback
    implementation; otherwise ``FATAL`` — the stage stops and is restarted,
    because continuing with unknown device state is forbidden.
    """

    ROLLBACK = "rollback"
    FATAL = "fatal"


@dataclass(frozen=True)
class LaggedGenerationContract:
    """A runner's immutable declaration that it satisfies the lagged protocol.

    Replaces per-batch ``lookahead_eligible``: eligibility is a stage-level
    capability, validated once at construction. A stage without a valid
    contract constructs as ``SYNC``.
    """

    #: every forward mode the stage can produce while lagged
    #: (e.g. {"DECODE", "MIXED", "EXTEND"}); must be non-empty
    supported_forward_modes: frozenset[str]
    #: launch returns the generic next token/sentinel plus all model-specific
    #: feedback the next forward needs; mandatory for LAGGED
    publishes_next_input: bool
    #: model commit accepts the scheduler's immutable commit mask; mandatory
    commit_is_masked: bool
    retraction_policy: RetractionPolicy
    failure_policy: FailurePolicy
    #: executable input-to-KV equivalence check (I5), or None while a runner
    #: proves it another way; documentation alone does not satisfy I5
    input_state_matches_relay: Callable[..., bool] | None = None
    #: execution/CUDA-graph capacity only — exceeding it means eager launch
    #: under the same lag protocol, never a switch to sync accounting
    max_launch_batch_size: int | None = None

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.supported_forward_modes:
            errors.append("supported_forward_modes is empty")
        if not self.publishes_next_input:
            errors.append("runner does not publish the next input at launch")
        if not self.commit_is_masked:
            errors.append("runner commit does not accept the commit mask")
        if self.max_launch_batch_size is not None and self.max_launch_batch_size < 1:
            errors.append("max_launch_batch_size must be None or >= 1")
        return errors

    @property
    def is_valid(self) -> bool:
        return not self.validation_errors()


@dataclass
class StepKVClaim:
    """One row's KV bookkeeping for one launched step.

    ``prior_*`` snapshot the request's lengths BEFORE preparation, which is
    what makes a pre-enqueue rollback exact: restore the lengths and free
    precisely the ``num_new_slots`` provisional positions
    ``req_to_token[req_pool_idx, prior_allocated_len : prior_allocated_len +
    num_new_slots]``. ``written``/``cacheable`` are the two independent
    outcomes the old compensating-free design conflated: a terminal overrun's
    KV is written AND cacheable while its output does not commit; a failed
    forward's is neither.
    """

    prior_committed_len: int
    prior_allocated_len: int
    num_new_slots: int
    written: bool = False
    cacheable: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.prior_committed_len <= self.prior_allocated_len:
            raise InvariantViolation(
                f"kv claim lengths invalid: committed={self.prior_committed_len} "
                f"allocated={self.prior_allocated_len}"
            )
        if self.num_new_slots < 0:
            raise InvariantViolation(f"negative slot claim: {self.num_new_slots}")


@dataclass(frozen=True)
class RelayHandle:
    """Opaque address of one request's unresolved next-input value."""

    pool_row: int


@dataclass
class PendingRow:
    """One launched row of a pending step, in launch/result order."""

    rid: str
    #: monotonic per-stage admission counter — prevents a reused RID or a
    #: recycled Req object from matching an older row (one mechanism, not
    #: "object identity or generation")
    admission_serial: int
    req_pool_idx: int
    #: scalar token or model-specific frame identity consumed by this forward
    input_key: Any
    relay_handle: RelayHandle | None
    kv_claim: StepKVClaim
    #: the forward completed sufficiently to classify the input KV and the
    #: model journal
    launch_valid: bool = False


class PendingGenerationStep:
    """The sole scheduler-owned unresolved transaction (replaces the untyped
    ``_async_pending`` tuple).

    ``batch_snapshot`` is the immutable request ordering plus only the fields
    ``process_batch_result`` needs — never a live mutable scheduling batch.
    ``commit_mask`` is set exactly once, immediately before commit, from the
    transaction snapshot and current request lifecycle; model hooks consume
    it instead of rediscovering policy from mutable ``Req.finished()``.
    ``kv_cacheable_mask`` is finalized after a successful launch and is
    independent of ``commit_mask`` (I7).
    """

    def __init__(
        self,
        *,
        step_id: int,
        batch_snapshot: Any,
        rows: Sequence[PendingRow],
        runner_step: Any = None,
        completion_event: Any = None,
        blocks_next_plan: bool = False,
    ) -> None:
        self.step_id = step_id
        self.batch_snapshot = batch_snapshot
        self.rows: tuple[PendingRow, ...] = tuple(rows)
        self.runner_step = runner_step
        self.completion_event = completion_event
        #: a semantic data dependency (e.g. a non-generative prefill chunk
        #: whose cache result the next chunk's plan needs) — never derived
        #: from queue length or batch size
        self.blocks_next_plan = bool(blocks_next_plan)
        self._state = PendingStepState.PLANNED
        self._commit_mask: tuple[bool, ...] | None = None
        self._kv_cacheable_mask: tuple[bool, ...] | None = None

    @property
    def state(self) -> PendingStepState:
        return self._state

    @property
    def commit_mask(self) -> tuple[bool, ...] | None:
        return self._commit_mask

    @property
    def kv_cacheable_mask(self) -> tuple[bool, ...] | None:
        return self._kv_cacheable_mask

    def transition(self, new_state: PendingStepState) -> None:
        legal = _LEGAL_TRANSITIONS[self._state]
        if new_state not in legal:
            raise InvariantViolation(
                f"step {self.step_id}: illegal transition "
                f"{self._state.value} -> {new_state.value} "
                f"(legal: {sorted(s.value for s in legal)})"
            )
        self._state = new_state

    def _check_mask(self, mask: Sequence[bool], name: str) -> tuple[bool, ...]:
        mask = tuple(bool(v) for v in mask)
        if len(mask) != len(self.rows):
            raise InvariantViolation(
                f"step {self.step_id}: {name} length {len(mask)} != "
                f"{len(self.rows)} rows"
            )
        return mask

    def set_kv_cacheable_mask(self, mask: Sequence[bool]) -> None:
        """Finalized after a successful launch (LAUNCHED or DEVICE_READY)."""
        if self._state not in (
            PendingStepState.LAUNCHED,
            PendingStepState.DEVICE_READY,
        ):
            raise InvariantViolation(
                f"step {self.step_id}: kv_cacheable_mask set in state "
                f"{self._state.value}"
            )
        if self._kv_cacheable_mask is not None:
            raise InvariantViolation(f"step {self.step_id}: kv_cacheable_mask reset")
        self._kv_cacheable_mask = self._check_mask(mask, "kv_cacheable_mask")

    def set_commit_mask(self, mask: Sequence[bool]) -> None:
        """Computed once, immediately before commit (DEVICE_READY only)."""
        if self._state is not PendingStepState.DEVICE_READY:
            raise InvariantViolation(
                f"step {self.step_id}: commit_mask set in state {self._state.value}"
            )
        if self._commit_mask is not None:
            raise InvariantViolation(f"step {self.step_id}: commit_mask reset")
        self._commit_mask = self._check_mask(mask, "commit_mask")


class OmniGenerationRelay:
    """Single owner of launched-but-uncommitted next-input values.

    Single-stream form: one ``[max_running_requests]`` device buffer, host-side
    publication bookkeeping. Publication-before-consumption is guaranteed by
    CUDA stream order plus host-ordered release-after-commit, so there are no
    generation or validity device buffers today.

    Stream-split contract (I4): any future forward/copy stream split MUST add
    generation tags, publication validity, poisoning on release, and
    ``record_stream()`` on transient relay tensors before it lands. That
    requirement is part of this class's contract, not an optimization note.

    Discipline (host-checked): each publication is consumed by exactly one
    materialization; publishing over an unconsumed value, materializing an
    unpublished row, or double-invalidating raises.
    """

    #: debug poison for invalidated rows — an accidental consume of a poisoned
    #: row produces an out-of-vocab id that fails loudly downstream
    POISON = -1

    def __init__(
        self,
        max_running_requests: int,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.int64,
    ) -> None:
        if max_running_requests < 1:
            raise InvariantViolation("relay needs max_running_requests >= 1")
        self.token_ids = torch.full(
            (max_running_requests,), self.POISON, dtype=dtype, device=device
        )
        self._published: set[int] = set()

    def handle(self, pool_row: int) -> RelayHandle:
        self._check_row(pool_row)
        return RelayHandle(pool_row=pool_row)

    def _check_row(self, row: int) -> None:
        if not 0 <= row < self.token_ids.numel():
            raise InvariantViolation(
                f"relay row {row} out of range ({self.token_ids.numel()})"
            )

    def publish(self, handles: Sequence[RelayHandle], values: torch.Tensor) -> None:
        """Write this step's sampled next inputs. Device-side this is one
        scatter enqueued after the sampler (stream-ordered before any consuming
        forward); host-side it marks the rows published."""
        if len(handles) != values.numel():
            raise InvariantViolation(
                f"relay publish: {len(handles)} handles != {values.numel()} values"
            )
        rows = []
        for h in handles:
            self._check_row(h.pool_row)
            if h.pool_row in self._published:
                raise InvariantViolation(
                    f"relay row {h.pool_row} republished before consumption"
                )
            rows.append(h.pool_row)
        idx = torch.tensor(rows, dtype=torch.long, device=self.token_ids.device)
        self.token_ids[idx] = values.to(self.token_ids.dtype).reshape(-1)
        self._published.update(rows)

    def materialize(self, handles: Sequence[RelayHandle]) -> torch.Tensor:
        """Gather the next forward's input rows, consuming the publications."""
        rows = []
        for h in handles:
            self._check_row(h.pool_row)
            if h.pool_row not in self._published:
                raise InvariantViolation(
                    f"relay row {h.pool_row} materialized without publication"
                )
            rows.append(h.pool_row)
        idx = torch.tensor(rows, dtype=torch.long, device=self.token_ids.device)
        out = self.token_ids[idx].clone()
        self._published.difference_update(rows)
        return out

    def invalidate(self, handles: Sequence[RelayHandle]) -> None:
        """Drop unconsumed publications on release/abort/retraction and poison
        the rows so a stale consume fails loudly."""
        rows = []
        for h in handles:
            self._check_row(h.pool_row)
            rows.append(h.pool_row)
        idx = torch.tensor(rows, dtype=torch.long, device=self.token_ids.device)
        self.token_ids[idx] = self.POISON
        self._published.difference_update(rows)

    def published_rows(self) -> frozenset[int]:
        return frozenset(self._published)


class AsyncGenerationController:
    """Owns the pending transaction (depth 0 or 1) and monotonic identities.

    Deliberately does NOT own: request admission, memory checks, cache
    eviction, retraction selection, mixed assembly (pinned SGLang), sampling,
    or model journals (runners).
    """

    def __init__(self, protocol: GenerationProtocol) -> None:
        self._protocol = protocol
        self._pending: PendingGenerationStep | None = None
        self._step_counter = 0
        self._admission_counter = 0

    @property
    def protocol(self) -> GenerationProtocol:
        return self._protocol

    def next_step_id(self) -> int:
        self._step_counter += 1
        return self._step_counter

    def next_admission_serial(self) -> int:
        self._admission_counter += 1
        return self._admission_counter

    @property
    def pending_depth(self) -> int:
        return 0 if self._pending is None else 1

    @property
    def pending(self) -> PendingGenerationStep | None:
        return self._pending

    def replace_pending(
        self, step: PendingGenerationStep
    ) -> PendingGenerationStep | None:
        """Install the just-launched step; return the previous one for commit.

        Launch-before-commit momentarily holds both inside one loop iteration;
        between iterations the controller owns at most one. Installation
        requires the newer step to be LAUNCHED — relay published, KV claims
        recorded — per I2.
        """
        if self._protocol is not GenerationProtocol.LAGGED:
            raise InvariantViolation(
                "pending steps exist only under the LAGGED protocol"
            )
        if step.state is not PendingStepState.LAUNCHED:
            raise InvariantViolation(
                f"step {step.step_id} installed in state {step.state.value}; "
                "a pending step must be LAUNCHED"
            )
        previous, self._pending = self._pending, step
        return previous

    def take_pending(self) -> PendingGenerationStep | None:
        """Remove and return the pending step for commit or failure handling."""
        step, self._pending = self._pending, None
        return step

    def assert_barrier_clear(self) -> None:
        """Administrative barriers complete only at pending depth 0."""
        if self._pending is not None:
            raise InvariantViolation(
                f"administrative barrier with step {self._pending.step_id} "
                f"pending in state {self._pending.state.value}"
            )
