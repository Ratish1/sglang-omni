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
LAG_PERIOD_S = 0.005
LAG_REPORT_S = 0.002

records = []
records_lock = threading.Lock()
gc_started = {}
sampled_loops = set()


def record(item):
    item["ns"] = item.get("ns") or time.time_ns()
    with records_lock:
        records.append(item)


def record_queue(ns, label, op, message):
    """A queue record as a tuple, the cheapest form on the hot path: ns, queue, op, thread, request, type, target,
    chunk id, source stage."""
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


def instrument_queue(q, label):
    if getattr(q, "hop_probe_label", None):
        return
    else:
        pass
    q.hop_probe_label = label
    put, get = q.put, q.get

    def put_recorded(item, *args, **kwargs):
        record_queue(time.time_ns(), label, "put", item)
        return put(item, *args, **kwargs)

    def get_recorded(*args, **kwargs):
        item = get(*args, **kwargs)
        record_queue(time.time_ns(), label, "get", item)
        return item

    # note: the stdlib put_nowait and get_nowait call self.put and self.get, so the instance wrappers see them too
    q.put, q.get = put_recorded, get_recorded


async def sample_lag(name):
    import asyncio

    loop = asyncio.get_running_loop()
    window_end = loop.time() + 1.0
    count, total, worst = 0, 0.0, 0.0
    while True:
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
