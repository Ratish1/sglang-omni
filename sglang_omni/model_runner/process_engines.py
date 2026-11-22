# SPDX-License-Identifier: Apache-2.0
"""SGLang engines that share one process, and so one CUDA context.

SGLang reads one active runtime context per process, at init and at forward time, and an engine embedded beside
another must serialize its host work. The first engine a process builds keeps the process context. Each later one gets
a context of its own, active while its stage is built, and the process groups the first one built. While a process
holds more than one engine, SGLang host work runs only inside hold(engine_context): the process's one turn, with that
engine's context active. A thread waiting on its own device work or its inbox steps out with released(), and a
lower-priority engine hands the turn to a waiting higher-priority one at yield_turn(), so the talker's short steps
never queue behind the thinker's host work.
"""

from __future__ import annotations

import functools
import itertools
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass, field

from sglang.srt.runtime_context import (
    RuntimeContext,
    get_context,
    get_parallel,
    use_context,
)

PROCESS_GROUP_NAMES = (
    "world_group",
    "tp_group",
    "pp_group",
    "moe_ep_group",
    "moe_dp_group",
    "moe_tp_group",
    "attn_tp_group",
    "attn_cp_group",
    "shared_experts_tp_group",
    "dcp_group",
)


@dataclass(kw_only=True)
class HeldEngine:
    engine_context: RuntimeContext
    priority: int
    depth: int
    activation: AbstractContextManager[RuntimeContext] | None = None


@dataclass(kw_only=True)
class TurnWaiter:
    engine_context: RuntimeContext
    priority: int
    arrival: int
    granted: threading.Event = field(default_factory=threading.Event)


class ThreadHold(threading.local):
    def __init__(self) -> None:
        self.held_engine: HeldEngine | None = None


class ProcessEngines:
    """The SGLang engines this process holds and the one turn their host work shares."""

    def __init__(self) -> None:
        self.engine_contexts: list[RuntimeContext] = []
        self.thread_hold = ThreadHold()
        self.construction_stack: ExitStack | None = None
        self.turn_state_lock = threading.Lock()
        self.is_turn_held = False
        self.turn_waiters: list[TurnWaiter] = []
        self.arrivals = itertools.count()
        self.deferring_contexts: list[RuntimeContext] = []
        self.expedited_contexts: list[RuntimeContext] = []

    @property
    def is_shared(self) -> bool:
        return len(self.engine_contexts) > 1

    @property
    def has_engine(self) -> bool:
        return bool(self.engine_contexts)

    @contextmanager
    def stage_construction(self) -> Iterator[None]:
        """Scope one stage factory: a later engine's context, entered inside it, is left when the factory returns."""
        with ExitStack() as construction_stack:
            self.construction_stack = construction_stack
            try:
                yield
            finally:
                self.construction_stack = None

    def enter_later_engine_context(self, engine_context: RuntimeContext) -> None:
        """Activate a later engine's context for its construction and give it the process groups the first built."""
        assert (
            self.construction_stack is not None
        ), "a later engine is built inside stage_construction"
        self.construction_stack.enter_context(use_context(engine_context))
        first_parallel = self.engine_contexts[0].parallel
        built_groups = {}
        for group_name in PROCESS_GROUP_NAMES:
            try:
                built_groups[group_name] = getattr(first_parallel, group_name)
            except RuntimeError:
                # note (ratish): a group the first engine's topology left unbuilt raises on read, and stays unset here.
                continue
        get_parallel().override_permanently(**built_groups)

    def register_engine(self) -> RuntimeContext:
        engine_context = get_context()
        self.engine_contexts.append(engine_context)
        return engine_context

    @contextmanager
    def hold(self, engine_context: RuntimeContext, *, priority: int) -> Iterator[None]:
        """Run this thread's SGLang host work with the process's turn and the engine's context."""
        if not self.is_shared:
            yield
            return
        else:
            pass
        held_engine = self.thread_hold.held_engine
        if held_engine is not None:
            assert (
                held_engine.engine_context is engine_context
            ), "a thread holds one engine's turn at a time"
            held_engine.depth += 1
            try:
                yield
            finally:
                held_engine.depth -= 1
            return
        else:
            pass
        held_engine = HeldEngine(
            engine_context=engine_context, priority=priority, depth=1
        )
        self.take_turn(held_engine)
        try:
            yield
        finally:
            self.leave_turn(held_engine)

    @contextmanager
    def released(self) -> Iterator[None]:
        """Give the turn up while this thread waits on its own device work or its inbox."""
        held_engine = self.thread_hold.held_engine
        if held_engine is None:
            yield
            return
        else:
            pass
        self.leave_turn(held_engine)
        try:
            yield
        finally:
            self.take_turn(held_engine)

    def defer(self, engine_context: RuntimeContext, *, is_deferring: bool) -> None:
        """Rank an engine below every other while it waits on their output, so they go first."""
        with self.turn_state_lock:
            self.deferring_contexts = listed_if(
                self.deferring_contexts, engine_context, is_listed=is_deferring
            )

    def expedite(self, engine_context: RuntimeContext, *, is_expedited: bool) -> None:
        """Rank an engine above every other while it holds work a request's first output waits on."""
        with self.turn_state_lock:
            self.expedited_contexts = listed_if(
                self.expedited_contexts, engine_context, is_listed=is_expedited
            )

    def rank(
        self, engine_context: RuntimeContext, priority: int
    ) -> tuple[bool, bool, int]:
        """An engine's place in the turn order, read under turn_state_lock: expedited first, deferring last, then by priority."""
        is_expedited = any(
            listed is engine_context for listed in self.expedited_contexts
        )
        is_deferring = any(
            listed is engine_context for listed in self.deferring_contexts
        )
        return (is_expedited, not is_deferring, priority)

    def yield_turn(self) -> None:
        """Hand the turn to a waiting engine that outranks this one and take it back after it."""
        held_engine = self.thread_hold.held_engine
        if held_engine is None:
            return
        else:
            pass
        with self.turn_state_lock:
            held_rank = self.rank(held_engine.engine_context, held_engine.priority)
            is_outranked = any(
                self.rank(waiter.engine_context, waiter.priority) > held_rank
                for waiter in self.turn_waiters
            )
        if is_outranked:
            self.leave_turn(held_engine)
            self.take_turn(held_engine)
        else:
            pass

    def take_turn(self, held_engine: HeldEngine) -> None:
        with self.turn_state_lock:
            if not self.is_turn_held and not self.turn_waiters:
                self.is_turn_held = True
                waiter = None
            else:
                waiter = TurnWaiter(
                    engine_context=held_engine.engine_context,
                    priority=held_engine.priority,
                    arrival=next(self.arrivals),
                )
                self.turn_waiters.append(waiter)
        if waiter is not None:
            waiter.granted.wait()
        else:
            pass
        held_engine.activation = use_context(held_engine.engine_context)
        held_engine.activation.__enter__()
        self.thread_hold.held_engine = held_engine

    def leave_turn(self, held_engine: HeldEngine) -> None:
        """Hand the turn straight to the highest-ranked, earliest waiter, so the leaver cannot take it back first."""
        assert (
            self.thread_hold.held_engine is held_engine
        ), "the turn is left by the thread holding it"
        self.thread_hold.held_engine = None
        held_engine.activation.__exit__(None, None, None)
        held_engine.activation = None
        with self.turn_state_lock:
            if self.turn_waiters:
                next_waiter = max(
                    self.turn_waiters,
                    key=lambda waiter: (
                        *self.rank(waiter.engine_context, waiter.priority),
                        -waiter.arrival,
                    ),
                )
                self.turn_waiters.remove(next_waiter)
                next_waiter.granted.set()
            else:
                self.is_turn_held = False


def listed_if(
    contexts: list[RuntimeContext], engine_context: RuntimeContext, *, is_listed: bool
) -> list[RuntimeContext]:
    """The contexts with engine_context in them exactly when is_listed, compared by identity."""
    others = [listed for listed in contexts if listed is not engine_context]
    return [*others, engine_context] if is_listed else others


@functools.cache
def process_engines() -> ProcessEngines:
    return ProcessEngines()
