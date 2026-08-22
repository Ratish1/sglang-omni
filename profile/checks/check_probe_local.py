"""Local validation of the NVTX probe without a GPU (recorder backend).

Run from a checkout root with the sglang venv:
    OMNI_NVTX_PROBE=1 PYTHONPATH=profile/nvtx_probe:. python profile/checks/check_probe_local.py

Checks: every wrap target exists (missing is empty); a real OmniScheduler
built by the unit-test constructor still runs one idle iteration of both
event loops under the wrappers; the recorded labels match the plan's set
for that path; batch labels carry bs and tok for extend and bs for decode.
"""

from __future__ import annotations

import types

import omni_nvtx_probe as probe
import pytest
from sglang.srt.model_executor.forward_batch_info import ForwardMode

from sglang_omni.scheduling import omni_scheduler as omni_mod
from tests.unit_test.fakes import real_radix_pools
from tests.unit_test.pipeline.test_scheduler import _construct_omni_scheduler

assert probe.backend_name == "recorder", probe.backend_name
assert not probe.missing, probe.missing
rec = probe.backend


def labels_since(n: int) -> list[str]:
    return [name for name, _tid, _s, _e in rec.ranges[n:]]


def one_idle_iteration(loop_name: str) -> list[str]:
    """Run one iteration of the real loop on an empty scheduler.

    The unit-test fakes do not carry the server_args the upstream batch
    builder reads, so get_next_batch_to_run is stubbed to None exactly as
    the pipeline tests do; the coalesce wrapper is exercised separately.
    """
    mp = pytest.MonkeyPatch()
    mp.delenv("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY", raising=False)
    mp.setattr(omni_mod.time, "sleep", lambda _s: None)
    scheduler = _construct_omni_scheduler(mp, pools=real_radix_pools())
    scheduler._running = True
    scheduler._engine_paused = False
    scheduler._async_pending = None
    scheduler.cur_batch = None
    scheduler.last_batch = None

    def stop_after_one(self):
        self._running = False
        return None

    mp.setattr(type(scheduler), "get_next_batch_to_run", stop_after_one)
    start = len(rec.ranges)
    getattr(scheduler, loop_name)()
    mp.undo()
    return labels_since(start)


expected_idle = {
    "sched:admin",
    "sched:recv",
    "sched:admit",
    "sched:idle_check",
    "sched:sleep",
}
for loop in ("_event_loop_normal", "_event_loop_async_decode"):
    got = one_idle_iteration(loop)
    assert expected_idle <= set(got), (loop, sorted(set(got)))
    print(f"{loop}: {got}")


def coalesce_hold_emits_a_mark() -> None:
    mp = pytest.MonkeyPatch()
    scheduler = _construct_omni_scheduler(mp, pools=real_radix_pools())
    scheduler.prefill_coalesce_requests = 4
    scheduler.prefill_coalesce_wait_s = 60.0
    scheduler.prefill_coalesce_when_idle = True
    scheduler.prefill_coalesce_requires_pending_builds = False
    scheduler.chunked_req = None
    scheduler.waiting_queue = [types.SimpleNamespace(), types.SimpleNamespace()]
    start_r, start_m = len(rec.ranges), len(rec.marks)
    plan = scheduler.get_new_batch_prefill(None)
    mp.undo()
    assert plan.batch_to_run is None
    assert labels_since(start_r) == ["sched:new_prefill"]
    assert [m[0] for m in rec.marks[start_m:]] == ["sched:hold waiting=2"]
    print("coalesce hold mark ok")


coalesce_hold_emits_a_mark()

extend = types.SimpleNamespace(
    forward_mode=ForwardMode.EXTEND, reqs=[1, 2, 3], extend_num_tokens=321
)
decode = types.SimpleNamespace(forward_mode=ForwardMode.DECODE, reqs=[1, 2])
label = probe._batch_label("exec:sync")
assert label(None, (extend,), {}) == "exec:sync:extend bs=3 tok=321"
assert label(None, (decode,), {}) == "exec:sync:decode bs=2"
assert label(None, (None,), {}) == "exec:sync:none"
print("batch labels ok")

threads = {tid for _n, tid, _s, _e in rec.ranges}
assert len(threads) == 1
for name, _tid, s, e in rec.ranges:
    assert e >= s, name
print("ranges well formed:", len(rec.ranges))
print("PROBE LOCAL CHECK PASSED")
