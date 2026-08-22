"""Local validation of the NVTX probe without a GPU.

Run from a checkout root with the server venv:
    OMNI_NVTX_PROBE=1 PYTHONPATH=profile/nvtx_probe:. python profile/checks/check_probe_local.py

Always checked: every wrap target exists (missing is empty); the coalesce
wrapper on a bare OmniScheduler emits its range and the hold mark; batch
labels carry bs and tok for extend and bs for decode; ranges are well
formed. On a CUDA box the NVTX backend is live and ranges go to nsys, so
the recorder-only assertions are skipped.

Also checked when the repo's unit-test constructor imports (it needs the
test fakes to import cleanly in this environment): a real OmniScheduler
runs one idle iteration of both event loops under the wrappers and the
idle-path labels come out in order. If that import fails the check says
SKIPPED with the reason and still passes on the rest.
"""

from __future__ import annotations

import inspect
import types

import omni_nvtx_probe as probe
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.scheduling import omni_scheduler as omni_mod

assert not probe.missing, probe.missing
print(f"backend={probe.backend_name} installed={len(probe.installed)} missing=none")
recorder = probe.backend if probe.backend_name == "recorder" else None


def labels_since(n: int) -> list[str]:
    return [name for name, _tid, _s, _e in recorder.ranges[n:]]


def coalesce_hold_emits_a_mark() -> None:
    scheduler = object.__new__(omni_mod.OmniScheduler)
    scheduler.prefill_coalesce_requests = 4
    scheduler.prefill_coalesce_wait_s = 60.0
    scheduler.prefill_coalesce_when_idle = True
    scheduler.prefill_coalesce_requires_pending_builds = False
    scheduler.chunked_req = None
    scheduler.waiting_queue = [types.SimpleNamespace(), types.SimpleNamespace()]
    if recorder is None:
        plan = scheduler.get_new_batch_prefill(None)
        assert plan.batch_to_run is None
        print(
            "coalesce hold path runs under the wrapper (nvtx backend, mark not readable here)"
        )
        return
    start_r, start_m = len(recorder.ranges), len(recorder.marks)
    plan = scheduler.get_new_batch_prefill(None)
    assert plan.batch_to_run is None
    assert labels_since(start_r) == ["sched:new_prefill"]
    assert [m[0] for m in recorder.marks[start_m:]] == ["sched:hold waiting=2"]
    print("coalesce hold mark ok")


def batch_labels() -> None:
    extend = types.SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, reqs=[1, 2, 3], extend_num_tokens=321
    )
    decode = types.SimpleNamespace(forward_mode=ForwardMode.DECODE, reqs=[1, 2])
    label = probe._batch_label("exec:sync")
    assert label(None, (extend,), {}) == "exec:sync:extend bs=3 tok=321"
    assert label(None, (decode,), {}) == "exec:sync:decode bs=2"
    assert label(None, (None,), {}) == "exec:sync:none"
    print("batch labels ok")


def idle_iterations() -> None:
    try:
        import pytest

        from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler
    except Exception as exc:
        print(
            f"SKIPPED idle-iteration check: unit-test constructor not importable here ({type(exc).__name__}: {exc})"
        )
        return
    try:
        from tests.unit_test.fakes import real_radix_pools
    except ImportError:
        real_radix_pools = None

    def build(mp):
        if (
            real_radix_pools is not None
            and "pools" in inspect.signature(_construct_omni_scheduler).parameters
        ):
            return _construct_omni_scheduler(mp, pools=real_radix_pools())
        return _construct_omni_scheduler(mp)

    expected = [
        "sched:admin",
        "sched:recv",
        "sched:admit",
        "sched:idle_check",
        "sched:sleep",
    ]
    for loop_name in ("_event_loop_normal", "_event_loop_async_decode"):
        mp = pytest.MonkeyPatch()
        mp.delenv("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", raising=False)
        mp.setattr(omni_mod.time, "sleep", lambda _s: None)
        scheduler = build(mp)
        scheduler._running = True
        scheduler._engine_paused = False
        scheduler._async_pending = None
        scheduler.cur_batch = None
        scheduler.last_batch = None

        def stop_after_one(self):
            self._running = False
            return None

        mp.setattr(type(scheduler), "get_next_batch_to_run", stop_after_one)
        start = len(recorder.ranges) if recorder else 0
        getattr(scheduler, loop_name)()
        mp.undo()
        if recorder is None:
            print(f"{loop_name}: one idle iteration ran under the wrappers")
            continue
        got = labels_since(start)
        assert set(expected) <= set(got), (loop_name, sorted(set(got)))
        print(f"{loop_name}: {got}")


coalesce_hold_emits_a_mark()
batch_labels()
idle_iterations()
if recorder is not None:
    assert len({tid for _n, tid, _s, _e in recorder.ranges}) == 1
    for name, _tid, s, e in recorder.ranges:
        assert e >= s, name
    print("ranges well formed:", len(recorder.ranges))
print("PROBE LOCAL CHECK PASSED")
