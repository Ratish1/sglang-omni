"""AR engine loop probe: NVTX ranges around every phase of OmniScheduler's step and of its model
runner, so an nsys trace shows what the scheduler thread does between the engine's GPU steps.
Put this directory on PYTHONPATH and set OMNI_LOOP_PROBE=1. No runtime code changes.

Ranges (one per call, on the calling thread):
  loop.admin, loop.recv, loop.admit, loop.next_batch, loop.idle   scheduler intake and choice
  loop.step <MODE> bs=<rows>                                     run_batch, one per engine step
  loop.launch <MODE> bs=<rows>, loop.resolve                     the async decode loop's halves
  loop.emit, loop.result                                         stream output, result processing
  mr.build, mr.prepare_forward, mr.post_decode, mr.post_prefill,  the runner's step
  mr.ensure, mr.publish, mr.finalize, mr.host_tokens,
  mr.post_process
Methods are wrapped on the scheduler's own class when it is built and on the runner's own class
when it is bound (bind_model_runner), so model overrides are covered.
"""

import functools
import importlib.abc
import importlib.util
import os
import sys

SCHEDULER = "sglang_omni.scheduling.omni_scheduler"

LOOP_METHODS = {
    "process_admin_requests": "loop.admin",
    "recv_requests": "loop.recv",
    "process_input_requests": "loop.admit",
    "get_next_batch_to_run": "loop.next_batch",
    "sleep_during_idle": "loop.idle",
    "emit_stream_output": "loop.emit",
    "process_batch_result": "loop.result",
    "resolve_and_process": "loop.resolve",
}
BATCH_METHODS = {"run_batch": "loop.step", "run_batch_launch": "loop.launch"}
RUNNER_METHODS = {
    "build_forward_batch": "mr.build",
    "prepare_and_forward": "mr.prepare_forward",
    "post_decode": "mr.post_decode",
    "post_prefill": "mr.post_prefill",
    "ensure_next_token_ids": "mr.ensure",
    "publish_next_tokens": "mr.publish",
    "finalize": "mr.finalize",
    "resolve_host_token_ids": "mr.host_tokens",
    "post_process_outputs": "mr.post_process",
}


def batch_label(label, batch):
    if batch is None:
        return label
    mode = getattr(getattr(batch, "forward_mode", None), "name", "NONE")
    return f"{label} {mode} bs={len(getattr(batch, 'reqs', ()) or ())}"


def wrap(cls, name, label, push, pop, with_batch=False):
    fn = cls.__dict__.get(name)
    if fn is None:
        for base in cls.__mro__[1:]:
            if name in base.__dict__:
                fn = base.__dict__[name]
                break
    if fn is None or getattr(fn, "loop_probe", False):
        return

    @functools.wraps(fn)
    def ranged(self, *args, **kwargs):
        push(batch_label(label, args[0] if args else None) if with_batch else label)
        try:
            return fn(self, *args, **kwargs)
        finally:
            pop()

    ranged.loop_probe = True
    setattr(cls, name, ranged)


def mark_requests(cls, mark):
    """10-19: sched.admit rid=R when a built request enters the AR's waiting queue, and
    sched.prefill rid=R when an extend batch holding it runs, so a request's wait for a
    running slot is read from the trace."""
    if getattr(cls, "request_marks", False):
        return
    cls.request_marks = True
    enqueue = cls.enqueue_built_request
    run_batch = cls.run_batch

    @functools.wraps(enqueue)
    def enqueue_marked(self, payload, *args, **kwargs):
        mark(f"sched.admit rid={payload.request_id}")
        return enqueue(self, payload, *args, **kwargs)

    @functools.wraps(run_batch)
    def run_batch_marked(self, batch, *args, **kwargs):
        if batch is not None and batch.forward_mode.is_extend():
            for req in batch.reqs:
                mark(f"sched.prefill rid={req.rid}")
        return run_batch(self, batch, *args, **kwargs)

    cls.enqueue_built_request = enqueue_marked
    cls.run_batch = run_batch_marked


def patch_scheduler(module):
    from torch.cuda import nvtx

    cls = module.OmniScheduler
    init = cls.__init__
    bind = cls.bind_model_runner

    @functools.wraps(init)
    def instrumented_init(self, *args, **kwargs):
        init(self, *args, **kwargs)
        for name, label in LOOP_METHODS.items():
            wrap(type(self), name, label, nvtx.range_push, nvtx.range_pop)
        for name, label in BATCH_METHODS.items():
            wrap(
                type(self), name, label, nvtx.range_push, nvtx.range_pop, with_batch=True
            )
        mark_requests(type(self), nvtx.mark)
        print(f"loop_probe scheduler {type(self).__name__}", file=sys.stderr, flush=True)

    @functools.wraps(bind)
    def instrumented_bind(self, model_runner):
        bind(self, model_runner)
        for name, label in RUNNER_METHODS.items():
            wrap(type(model_runner), name, label, nvtx.range_push, nvtx.range_pop)
        print(
            f"loop_probe runner {type(model_runner).__name__} on {type(self).__name__}",
            file=sys.stderr,
            flush=True,
        )

    cls.__init__ = instrumented_init
    cls.bind_model_runner = instrumented_bind


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name != SCHEDULER or name in self.seen:
            return None
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch_scheduler(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("OMNI_LOOP_PROBE") == "1":
    sys.meta_path.insert(0, PatchOnImport())
