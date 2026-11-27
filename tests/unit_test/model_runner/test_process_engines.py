# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import time

from sglang.srt.runtime_context import ParallelContext, RuntimeContext, get_context

from sglang_omni.model_runner.process_engines import ProcessEngines

THINKER_PRIORITY = 0
TALKER_PRIORITY = 1


def make_engines(engine_count: int) -> tuple[ProcessEngines, list[RuntimeContext]]:
    engines = ProcessEngines()
    contexts = [RuntimeContext(parallel=ParallelContext()) for _ in range(engine_count)]
    engines.engine_contexts.extend(contexts)
    return engines, contexts


def wait_until_waiting(engines: ProcessEngines, waiter_count: int) -> None:
    while True:
        with engines.turn_state_lock:
            if len(engines.turn_waiters) == waiter_count:
                return
            else:
                pass
        time.sleep(0)


def test_a_process_with_one_engine_takes_no_turn_and_keeps_its_context() -> None:
    engines, (only_context,) = make_engines(1)
    process_context = get_context()
    with engines.hold(only_context, priority=THINKER_PRIORITY):
        assert not engines.is_turn_held
        assert get_context() is process_context
        with engines.released():
            assert not engines.is_turn_held
    assert get_context() is process_context


def test_a_released_turn_runs_the_other_engine_in_its_own_context() -> None:
    engines, (thinker_context, talker_context) = make_engines(2)
    process_context = get_context()
    talker_may_hold = threading.Event()
    talker_held = threading.Event()
    seen: dict[str, RuntimeContext] = {}

    def talker() -> None:
        talker_may_hold.wait()
        with engines.hold(talker_context, priority=TALKER_PRIORITY):
            seen["talker"] = get_context()
        talker_held.set()

    talker_thread = threading.Thread(target=talker)
    talker_thread.start()
    with engines.hold(thinker_context, priority=THINKER_PRIORITY):
        seen["thinker before"] = get_context()
        with engines.released():
            talker_may_hold.set()
            talker_held.wait()
        seen["thinker after"] = get_context()
    talker_thread.join()
    assert seen == {
        "thinker before": thinker_context,
        "talker": talker_context,
        "thinker after": thinker_context,
    }
    assert get_context() is process_context
    assert not engines.is_turn_held


def test_a_yielded_turn_goes_to_the_waiting_talker_and_comes_back() -> None:
    engines, (thinker_context, talker_context) = make_engines(2)
    order: list[str] = []

    def talker() -> None:
        with engines.hold(talker_context, priority=TALKER_PRIORITY):
            order.append("talker")

    with engines.hold(thinker_context, priority=THINKER_PRIORITY):
        order.append("thinker before")
        talker_thread = threading.Thread(target=talker)
        talker_thread.start()
        wait_until_waiting(engines, 1)
        engines.yield_turn()
        order.append("thinker after")
        assert get_context() is thinker_context
    talker_thread.join()
    assert order == ["thinker before", "talker", "thinker after"]


def test_a_left_turn_goes_to_the_talker_before_an_earlier_thinker() -> None:
    engines, (holder_context, thinker_context, talker_context) = make_engines(3)
    order: list[str] = []

    def hold_and_record(context: RuntimeContext, priority: int, name: str) -> None:
        with engines.hold(context, priority=priority):
            order.append(name)

    with engines.hold(holder_context, priority=THINKER_PRIORITY):
        thinker_thread = threading.Thread(
            target=hold_and_record, args=(thinker_context, THINKER_PRIORITY, "thinker")
        )
        thinker_thread.start()
        wait_until_waiting(engines, 1)
        talker_thread = threading.Thread(
            target=hold_and_record, args=(talker_context, TALKER_PRIORITY, "talker")
        )
        talker_thread.start()
        wait_until_waiting(engines, 2)
    thinker_thread.join()
    talker_thread.join()
    assert order == ["talker", "thinker"]


def test_a_deferring_talker_lets_the_thinker_keep_the_turn_until_it_stops_deferring() -> (
    None
):
    engines, (thinker_context, talker_context) = make_engines(2)
    order: list[str] = []

    def talker() -> None:
        with engines.hold(talker_context, priority=TALKER_PRIORITY):
            order.append("talker")

    engines.defer(talker_context, is_deferring=True)
    with engines.hold(thinker_context, priority=THINKER_PRIORITY):
        talker_thread = threading.Thread(target=talker)
        talker_thread.start()
        wait_until_waiting(engines, 1)
        engines.yield_turn()
        order.append("thinker kept the turn")
        engines.defer(talker_context, is_deferring=False)
        engines.yield_turn()
        order.append("thinker after")
    talker_thread.join()
    assert order == ["thinker kept the turn", "talker", "thinker after"]


def test_a_left_turn_goes_to_a_waiting_thinker_before_a_deferring_talker() -> None:
    engines, (holder_context, thinker_context, talker_context) = make_engines(3)
    order: list[str] = []

    def hold_and_record(context: RuntimeContext, priority: int, name: str) -> None:
        with engines.hold(context, priority=priority):
            order.append(name)

    engines.defer(talker_context, is_deferring=True)
    with engines.hold(holder_context, priority=THINKER_PRIORITY):
        talker_thread = threading.Thread(
            target=hold_and_record, args=(talker_context, TALKER_PRIORITY, "talker")
        )
        talker_thread.start()
        wait_until_waiting(engines, 1)
        thinker_thread = threading.Thread(
            target=hold_and_record, args=(thinker_context, THINKER_PRIORITY, "thinker")
        )
        thinker_thread.start()
        wait_until_waiting(engines, 2)
    talker_thread.join()
    thinker_thread.join()
    assert order == ["thinker", "talker"]


def test_an_expedited_lower_priority_engine_is_handed_the_turn_before_the_talker() -> (
    None
):
    engines, (holder_context, thinker_context, talker_context) = make_engines(3)
    order: list[str] = []

    def hold_and_record(context: RuntimeContext, priority: int, name: str) -> None:
        with engines.hold(context, priority=priority):
            order.append(name)

    engines.expedite(thinker_context, is_expedited=True)
    with engines.hold(holder_context, priority=THINKER_PRIORITY):
        talker_thread = threading.Thread(
            target=hold_and_record, args=(talker_context, TALKER_PRIORITY, "talker")
        )
        talker_thread.start()
        wait_until_waiting(engines, 1)
        thinker_thread = threading.Thread(
            target=hold_and_record, args=(thinker_context, THINKER_PRIORITY, "thinker")
        )
        thinker_thread.start()
        wait_until_waiting(engines, 2)
    talker_thread.join()
    thinker_thread.join()
    assert order == ["thinker", "talker"]


def test_an_engine_that_raises_inside_its_turn_hands_it_to_the_waiting_engine() -> None:
    engines, (thinker_context, talker_context) = make_engines(2)
    process_context = get_context()
    order: list[str] = []

    def talker() -> None:
        with engines.hold(talker_context, priority=TALKER_PRIORITY):
            order.append("talker")
            assert get_context() is talker_context

    talker_thread = threading.Thread(target=talker)
    try:
        with engines.hold(thinker_context, priority=THINKER_PRIORITY):
            talker_thread.start()
            wait_until_waiting(engines, 1)
            raise RuntimeError("thinker step failed")
    except RuntimeError:
        order.append("thinker raised")
    talker_thread.join()
    assert sorted(order) == ["talker", "thinker raised"]
    assert get_context() is process_context
    assert not engines.is_turn_held


def test_a_nested_hold_releases_and_leaves_the_turn_once() -> None:
    engines, (thinker_context, talker_context) = make_engines(2)
    talker_held = threading.Event()

    def talker() -> None:
        with engines.hold(talker_context, priority=TALKER_PRIORITY):
            assert get_context() is talker_context
        talker_held.set()

    with engines.hold(thinker_context, priority=THINKER_PRIORITY):
        with engines.hold(thinker_context, priority=THINKER_PRIORITY):
            with engines.released():
                talker_thread = threading.Thread(target=talker)
                talker_thread.start()
                talker_held.wait()
            assert get_context() is thinker_context
            assert engines.is_turn_held
        assert engines.is_turn_held
    talker_thread.join()
    assert not engines.is_turn_held
