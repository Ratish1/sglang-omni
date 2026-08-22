"""NVTX ranges around every host phase of one OmniScheduler iteration.

Experiment-only. Installed by profile/nvtx_probe/sitecustomize.py when
OMNI_NVTX_PROBE=1 is set; applies at class level after the omni modules
import, so the same file instruments every arm without a source edit. Each
wrapper is push, try call, finally pop: no control flow changes.

Labels are documented in profile/README.md and tasks/loop_profile_plan_20260822.md. A
target missing on an arm is logged and skipped; the installed set is logged
once per process so the capture record shows which labels were live.

Without CUDA (local validation) the ranges go to an in-process recorder
instead of NVTX; profile/checks/check_probe_local.py reads it back.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import threading
import time

logger = logging.getLogger("omni_nvtx_probe")

_ENV = "OMNI_NVTX_PROBE"


class _Recorder:
    """Fallback backend when torch.cuda.nvtx is unavailable: keeps ranges per thread."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ranges: list[tuple[str, int, float, float]] = []
        self.marks: list[tuple[str, int, float]] = []
        self._stacks: dict[int, list[tuple[str, float]]] = {}

    def push(self, name: str) -> None:
        tid = threading.get_ident()
        with self.lock:
            self._stacks.setdefault(tid, []).append((name, time.perf_counter()))

    def pop(self) -> None:
        tid = threading.get_ident()
        end = time.perf_counter()
        with self.lock:
            name, start = self._stacks[tid].pop()
            self.ranges.append((name, tid, start, end))

    def mark(self, name: str) -> None:
        with self.lock:
            self.marks.append((name, threading.get_ident(), time.perf_counter()))


class _Nvtx:
    def __init__(self, nvtx) -> None:
        self.push = nvtx.range_push
        self.pop = nvtx.range_pop
        self.mark = nvtx.mark


def _backend():
    try:
        import torch

        if torch.cuda.is_available():
            return _Nvtx(torch.cuda.nvtx), "nvtx"
    except Exception:
        pass
    return _Recorder(), "recorder"


backend, backend_name = _backend()
installed: list[str] = []
missing: list[str] = []


def _wrap(cls, attr: str, label) -> None:
    """Replace cls.attr with a pushed and popped version.

    label is a string or a callable (self, args, kwargs) -> str evaluated
    before the call; a callable that raises falls back to the attribute
    name so the probe can never fail the wrapped call.
    """
    target = f"{cls.__name__}.{attr}"
    fn = cls.__dict__.get(attr)
    if fn is None:
        missing.append(target)
        return
    if isinstance(fn, staticmethod):
        missing.append(target + " (staticmethod)")
        return

    fallback = label if isinstance(label, str) else attr

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        if isinstance(label, str):
            name = label
        else:
            # A label is decoration; it must never raise into the loop.
            try:
                name = label(self, args, kwargs)
            except Exception:
                name = fallback
        backend.push(name)
        try:
            return fn(self, *args, **kwargs)
        finally:
            backend.pop()

    setattr(cls, attr, wrapper)
    installed.append(f"{target}")


def _batch_kind(batch) -> str:
    mode = batch.forward_mode
    if mode is None:
        return "none"
    if mode.is_extend():
        return "extend"
    if mode.is_decode():
        return "decode"
    return "other"


def _batch_label(prefix: str):
    def label(self, args, kwargs):
        batch = args[0] if args else kwargs.get("batch")
        if batch is None:
            return f"{prefix}:none"
        kind = _batch_kind(batch)
        bs = len(batch.reqs)
        if kind == "extend":
            tok = batch.extend_num_tokens
            return f"{prefix}:{kind} bs={bs} tok={tok}"
        return f"{prefix}:{kind} bs={bs}"

    return label


def _install_scheduler(omni_mod, upstream_mod, schedule_batch_mod) -> None:
    S = omni_mod.OmniScheduler
    _wrap(S, "_process_admin_requests", "sched:admin")
    _wrap(S, "recv_requests", "sched:recv")
    _wrap(S, "process_input_requests", "sched:admit")
    _wrap(S, "get_next_batch_to_run", "sched:next_batch")
    _wrap(S, "_run_batch_launch", _batch_label("exec:launch"))
    _wrap(S, "run_batch", _batch_label("exec:sync"))
    _wrap(S, "_run_batch_resolve", _batch_label("exec:resolve"))
    _wrap(S, "_resolve_pending_async", "sched:drain")
    _wrap(S, "_drop_stale_overrun", "sched:drop_stale")
    _wrap(S, "self_check_during_idle", "sched:idle_check")
    _wrap(S, "_sleep_during_idle", "sched:sleep")
    _wrap(S, "_run_request_builder", "build:req")

    # The coalesce wrapper: range plus a mark when it holds a non-empty queue.
    fn = S.__dict__.get("get_new_batch_prefill")
    if fn is None:
        missing.append("OmniScheduler.get_new_batch_prefill")
    else:

        @functools.wraps(fn)
        def new_prefill(self, *args, **kwargs):
            backend.push("sched:new_prefill")
            try:
                plan = fn(self, *args, **kwargs)
            finally:
                backend.pop()
            if plan is not None and plan.batch_to_run is None and self.waiting_queue:
                backend.mark(f"sched:hold waiting={len(self.waiting_queue)}")
            return plan

        S.get_new_batch_prefill = new_prefill
        installed.append("OmniScheduler.get_new_batch_prefill")

    # Upstream methods reached through OmniScheduler.__getattr__ bind from the
    # upstream class, so wrapping there is what the omni loop sees.
    U = upstream_mod.Scheduler
    _wrap(U, "process_batch_result", _batch_label("exec:process"))
    SB = schedule_batch_mod.ScheduleBatch
    _wrap(SB, "prepare_for_extend", "sched:prepare_extend")
    _wrap(SB, "prepare_for_decode", "sched:prepare_decode")


def _install_runner(base_mod) -> None:
    R = base_mod.ModelRunner
    _wrap(R, "_build_forward_batch", "run:build")
    _wrap(R, "_prepare_and_forward", "run:forward")
    _wrap(R, "_finalize", "run:finalize")


def _install_encoder(pre_lm_mod) -> None:
    E = pre_lm_mod.PreLMEncoderService

    def batch_label(self, args, kwargs):
        items = args[0] if args else kwargs.get("items")
        return f"enc:batch n={len(items)}"

    _wrap(E, "_execute_batch", batch_label)
    _wrap(E, "synchronize_batch", "enc:sync")
    _wrap(E, "cache_embedding", "enc:cache")


_done = False


def install() -> None:
    """Wrap every target that is importable. Safe to call more than once."""
    global _done
    if _done:
        return
    _done = True
    from sglang.srt.managers import schedule_batch as schedule_batch_mod
    from sglang.srt.managers import scheduler as upstream_mod

    from sglang_omni.model_runner import base as base_mod
    from sglang_omni.scheduling import omni_scheduler as omni_mod
    from sglang_omni.scheduling import pre_lm_encoder as pre_lm_mod

    _install_scheduler(omni_mod, upstream_mod, schedule_batch_mod)
    _install_runner(base_mod)
    _install_encoder(pre_lm_mod)
    logger.warning(
        "[nvtx-probe] backend=%s pid=%d installed=%d missing=%s",
        backend_name,
        os.getpid(),
        len(installed),
        missing or "none",
    )


class _PostImportHook:
    """sys.meta_path entry that runs install() right after omni_scheduler imports."""

    TRIGGER = "sglang_omni.scheduling.omni_scheduler"

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.TRIGGER:
            return None
        import importlib.machinery
        import importlib.util

        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            install()

        loader.exec_module = exec_module
        return spec


def arm() -> None:
    """Register the post-import hook when OMNI_NVTX_PROBE=1."""
    if os.environ.get(_ENV) != "1":
        return
    if any(isinstance(f, _PostImportHook) for f in sys.meta_path):
        return
    if _PostImportHook.TRIGGER in sys.modules:
        install()
        return
    sys.meta_path.insert(0, _PostImportHook())
