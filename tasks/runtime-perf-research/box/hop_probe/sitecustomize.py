"""Hop, loop lag and garbage collection records for every server process, without runtime code changes.

Put this directory ahead of the tree on PYTHONPATH and set OMNI_HOP_PROBE to an output prefix. Each process appends
JSON lines to <prefix>.<pid>.jsonl from a flush thread (once a second and at exit). Timestamps are time.time_ns(), the
clock of the request event recorder, so records join its events per request.

  queue       a JSON list, one per put and per successful get on every stage scheduler's inbox and outbox, and one
              per message of put_messages (a whole step's outbox put under one lock): [ns, queue, op, thread,
              request, message type, target, chunk id of a stream item, its source stage]
  kind "lag"  per second and event loop: sampler wakeups, total and max lateness of a 5 ms sleep; and each wakeup
              later than 2 ms on its own
  kind "gc"   each collection of generation 1 or 2, and each of any generation over 1 ms: generation, duration,
              collected objects, thread
  kind "stall" once per stall of an event loop (its sampler silent for more than 200 ms): every thread's innermost
              14 frames, taken by a watchdog thread while the stall lasts
  kind "heap" once per process, when its first new request reaches a stage inbox: tracked objects, the 25 most
              common types and owning packages, and the duration of one full collection over that heap
"""

import atexit
import functools
import gc
import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time

PREFIX = os.environ.get("OMNI_HOP_PROBE", "")
# OMNI_HOP_PROBE_QUEUES=0 drops the queue records, whose tuples live up to a second and would add their own
# promotions to a garbage collection frequency measurement
QUEUE_RECORDS = os.environ.get("OMNI_HOP_PROBE_QUEUES", "1") == "1"
# what-if, not a probe: OMNI_BIND_THREAD_DEVICE=1 makes every new thread bind the CUDA device its creator had current,
# as the stage scheduler thread does (pipeline/stage/runtime.py:278), to check that unbound workers cause the context
# a stage process placed off device 0 opens on device 0
BIND_THREAD_DEVICE = os.environ.get("OMNI_BIND_THREAD_DEVICE") == "1"
# what-if, not a probe: OMNI_FREEZE_STARTUP_GC=1 moves each process's heap into the permanent generation when its
# first new request arrives, as the launcher does for the API process at startup (serve/launcher.py:173-176)
FREEZE_STARTUP_GC = os.environ.get("OMNI_FREEZE_STARTUP_GC") == "1"
LAG_PERIOD_S = 0.005
LAG_REPORT_S = 0.002
STALL_S = 0.2

records = []
records_lock = threading.Lock()
gc_started = {}
sampled_loops = set()
heap_taken = []


def record(item):
    item["ns"] = item.get("ns") or time.time_ns()
    with records_lock:
        records.append(item)


def record_queue(ns, label, op, message):
    """A queue record as a tuple, the cheapest form on the hot path: ns, queue, op, thread, request, type, target,
    chunk id, source stage."""
    if not QUEUE_RECORDS:
        return
    else:
        pass
    data = getattr(message, "data", None)
    item = (
        ns, label, op, threading.current_thread().name, getattr(message, "request_id", None),
        getattr(message, "type", None), getattr(message, "target", None), getattr(data, "chunk_id", None),
        getattr(data, "from_stage", None),
    )
    with records_lock:
        records.append(item)


def flush():
    with records_lock:
        pending = records[:]
        records.clear()
    if pending:
        with open(f"{PREFIX}.{os.getpid()}.jsonl", "a", encoding="utf-8") as sink:
            sink.write("".join(json.dumps(item) + "\n" for item in pending))
    else:
        pass


def flush_forever():
    while True:
        time.sleep(1.0)
        flush()


def on_gc(phase, info):
    thread = threading.get_ident()
    if phase == "start":
        gc_started[thread] = time.perf_counter_ns()
        return
    else:
        pass
    started = gc_started.pop(thread, None)
    if started is None:
        return
    else:
        pass
    duration_ns = time.perf_counter_ns() - started
    if info["generation"] >= 1 or duration_ns > 1_000_000:
        record({"k": "gc", "gen": info["generation"], "dur_ns": duration_ns, "collected": info["collected"],
                "th": threading.current_thread().name})
    else:
        pass


def take_heap():
    """Census of the tracked heap and one timed full collection over it, then the freeze of the what-if."""
    objects = gc.get_objects()
    by_type, by_package = {}, {}
    for obj in objects:
        kind = type(obj)
        name = f"{kind.__module__}.{kind.__qualname__}"
        by_type[name] = by_type.get(name, 0) + 1
        try:
            owner = getattr(obj, "__module__", None) if callable(obj) else kind.__module__
        except Exception:
            owner = None
        package = (owner if isinstance(owner, str) else "?").split(".")[0]
        by_package[package] = by_package.get(package, 0) + 1
    tracked = len(objects)
    del objects
    started = time.perf_counter_ns()
    gc.collect()
    collect_ns = time.perf_counter_ns() - started
    if FREEZE_STARTUP_GC:
        gc.freeze()
    else:
        pass
    record({"k": "heap", "tracked": tracked, "full_collect_ns": collect_ns, "frozen": gc.get_freeze_count(),
            "top_types": sorted(by_type.items(), key=lambda kv: -kv[1])[:25],
            "top_packages": sorted(by_package.items(), key=lambda kv: -kv[1])[:25]})


def instrument_queue(q, label):
    if getattr(q, "hop_probe_label", None):
        return
    else:
        pass
    q.hop_probe_label = label
    put, get = q.put, q.get

    def put_recorded(item, *args, **kwargs):
        record_queue(time.time_ns(), label, "put", item)
        if not heap_taken and label.endswith(".in") and getattr(item, "type", None) == "new_request":
            heap_taken.append(label)
            take_heap()
        else:
            pass
        return put(item, *args, **kwargs)

    def get_recorded(*args, **kwargs):
        item = get(*args, **kwargs)
        record_queue(time.time_ns(), label, "get", item)
        return item

    # note: the stdlib put_nowait and get_nowait call self.put and self.get, so the instance wrappers see them too
    q.put, q.get = put_recorded, get_recorded


def watch_stalls(name, ticks, loop_thread_id):
    """Records every thread's stack once per stall when the loop's sampler has not ticked for STALL_S."""
    import traceback

    stalled = False
    while True:
        time.sleep(STALL_S / 4)
        quiet = time.monotonic() - ticks[0]
        if quiet > STALL_S and not stalled:
            stalled = True
            frames = sys._current_frames()
            names = {thread.ident: thread.name for thread in threading.enumerate()}
            stacks = {
                names.get(ident, str(ident)): [f"{f.filename}:{f.lineno} {f.name}" for f in traceback.extract_stack(frame)][-14:]
                for ident, frame in frames.items()
            }
            record({"k": "stall", "loop": name, "quiet_ms": quiet * 1e3,
                    "loop_thread": names.get(loop_thread_id, str(loop_thread_id)), "stacks": stacks})
        elif quiet <= STALL_S:
            stalled = False
        else:
            pass


async def sample_lag(name):
    import asyncio

    loop = asyncio.get_running_loop()
    ticks = [time.monotonic()]
    threading.Thread(target=watch_stalls, args=(name, ticks, threading.get_ident()),
                     name=f"hop-probe-stall-{name}", daemon=True).start()
    window_end = loop.time() + 1.0
    count, total, worst = 0, 0.0, 0.0
    while True:
        ticks[0] = time.monotonic()
        target = loop.time() + LAG_PERIOD_S
        await asyncio.sleep(LAG_PERIOD_S)
        late = loop.time() - target
        count, total, worst = count + 1, total + late, max(worst, late)
        if late > LAG_REPORT_S:
            record({"k": "lag1", "loop": name, "late_ms": late * 1e3})
        else:
            pass
        if loop.time() >= window_end:
            record({"k": "lag", "loop": name, "n": count, "sum_ms": total * 1e3, "max_ms": worst * 1e3})
            window_end = loop.time() + 1.0
            count, total, worst = 0, 0.0, 0.0
        else:
            pass


def start_lag_sampler(name):
    import asyncio

    loop = asyncio.get_running_loop()
    if id(loop) in sampled_loops:
        return
    else:
        pass
    sampled_loops.add(id(loop))
    loop.create_task(sample_lag(name))


def patch_runtime(module):
    stage = module.Stage
    start, run = stage.start, stage.run

    @functools.wraps(start)
    async def start_recorded(self):
        scheduler = self.scheduler
        if scheduler is not None:
            for attr, short in (("inbox", "in"), ("outbox", "out")):
                q = getattr(scheduler, attr, None)
                if q is not None and hasattr(q, "get_nowait"):
                    instrument_queue(q, f"{self.name}.{short}")
                else:
                    pass
        else:
            pass
        return await start(self)

    @functools.wraps(run)
    async def run_sampled(self, *args, **kwargs):
        start_lag_sampler(f"stage:{self.name}")
        return await run(self, *args, **kwargs)

    stage.start, stage.run = start_recorded, run_sampled


def patch_put_messages(module):
    put_messages = module.put_messages

    def put_messages_recorded(outbox, messages):
        label = getattr(outbox, "hop_probe_label", None)
        if label is not None:
            now = time.time_ns()
            for message in messages:
                record_queue(now, label, "put", message)
        else:
            pass
        return put_messages(outbox, messages)

    module.put_messages = put_messages_recorded


def patch_coordinator(module):
    coordinator = module.Coordinator
    run_completion_loop = coordinator.run_completion_loop

    @functools.wraps(run_completion_loop)
    async def run_sampled(self, *args, **kwargs):
        start_lag_sampler("launcher")
        return await run_completion_loop(self, *args, **kwargs)

    coordinator.run_completion_loop = run_sampled


def bind_thread_device():
    """New threads set the CUDA device that was current in the creating thread before their target runs."""
    start, run = threading.Thread.start, threading.Thread.run

    def start_capturing(self, *args, **kwargs):
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_initialized():
            self.bound_device = torch.cuda.current_device()
        else:
            pass
        return start(self, *args, **kwargs)

    def run_bound(self, *args, **kwargs):
        device = getattr(self, "bound_device", None)
        if device is not None:
            sys.modules["torch"].cuda.set_device(device)
        else:
            pass
        return run(self, *args, **kwargs)

    threading.Thread.start, threading.Thread.run = start_capturing, run_bound


PATCHES = {
    "sglang_omni.pipeline.stage.runtime": patch_runtime,
    "sglang_omni.scheduling.omni_scheduler": patch_put_messages,
    "sglang_omni.models.qwen3_omni.talker_model_runner": patch_put_messages,
    "sglang_omni.pipeline.coordinator": patch_coordinator,
}


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES or name in getattr(self, "seen", ()):
            return None
        else:
            pass
        self.seen = (*getattr(self, "seen", ()), name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        else:
            pass
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                PATCHES[name](module)
            except Exception as exc:
                print(f"hop_probe patch {name} failed: {exc!r}", file=sys.stderr, flush=True)

        spec.loader.exec_module = exec_and_patch
        return spec


if PREFIX:
    sys.meta_path.insert(0, PatchOnImport())
    gc.callbacks.append(on_gc)
    threading.Thread(target=flush_forever, name="hop-probe-flush", daemon=True).start()
    atexit.register(flush)
    print(f"hop_probe on, prefix {PREFIX}", file=sys.stderr, flush=True)
else:
    pass
if BIND_THREAD_DEVICE:
    bind_thread_device()
    print("hop_probe what-if: new threads bind their creator's CUDA device", file=sys.stderr, flush=True)
else:
    pass
if FREEZE_STARTUP_GC:
    print("hop_probe what-if: the heap is frozen at the first new request", file=sys.stderr, flush=True)
else:
    pass
