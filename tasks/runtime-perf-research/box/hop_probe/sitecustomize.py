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
  kind "frozen" every OMNI_HOP_PROBE_FREEZE_COUNT_S seconds when set: objects in the permanent generation
  kind "tb"   with OMNI_HOP_PROBE_TALKER_BATCHES=1, each Qwen3-Omni talker decode step: its rows and their request
              ids (Qwen3OmniTalker.prepare_decode_buffers)
  kind "req"  with OMNI_HOP_PROBE_TALKER_BATCHES=1, each Qwen3-Omni request at preprocessing: its id and the text of
              its last message, which joins a benchmark sample to its request id
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
# as the stage scheduler thread does (pipeline/stage/runtime.py:278); OMNI_BIND_THREAD_DEVICE=cuda:<n> makes every new
# thread of a process that has imported torch bind device n before its target runs, so a server placed wholly on
# device n shows whether unbound threads are the only source of the context it opens on device 0
BIND_THREAD_DEVICE = os.environ.get("OMNI_BIND_THREAD_DEVICE", "")
# what-if, not a probe: OMNI_FREEZE_STARTUP_GC=1 moves each process's heap into the permanent generation when its
# first new request arrives, as the launcher does for the API process at startup (serve/launcher.py:173-176)
FREEZE_STARTUP_GC = os.environ.get("OMNI_FREEZE_STARTUP_GC") == "1"
# OMNI_HOP_PROBE_FREEZE_COUNT_S=<s> records gc.get_freeze_count() every s seconds, which walks the permanent
# generation under the GIL, so it is for a check boot and not for a measured one
FREEZE_COUNT_PERIOD_S = float(os.environ.get("OMNI_HOP_PROBE_FREEZE_COUNT_S", "0"))
TALKER_BATCHES = os.environ.get("OMNI_HOP_PROBE_TALKER_BATCHES") == "1"
# OMNI_HOP_PROBE_SYNC_SITES=1 sets torch's sync debug mode to warn in every process and counts each synchronizing
# CUDA call by thread and by its innermost sglang frames (kind "sync", the running counts once a second); a check
# boot, not a measured one, since every count walks the Python stack
SYNC_SITES = os.environ.get("OMNI_HOP_PROBE_SYNC_SITES") == "1"
# OMNI_HOP_PROBE_TALKER_BUILD_REFERENCE=<tree> runs that tree's Qwen3-Omni talker prompt build beside the running one on
# every admission, on the same builder state, and records whether every output is bit identical (kind "build")
TALKER_BUILD_REFERENCE = os.environ.get("OMNI_HOP_PROBE_TALKER_BUILD_REFERENCE", "")
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
        ns,
        label,
        op,
        threading.current_thread().name,
        getattr(message, "request_id", None),
        getattr(message, "type", None),
        getattr(message, "target", None),
        getattr(data, "chunk_id", None),
        getattr(data, "from_stage", None),
    )
    with records_lock:
        records.append(item)


sync_sites = {}
sync_sites_flushed = [0]
show_warning = None


def count_sync_site(message, category, filename, lineno, file=None, line=None):
    """warnings.showwarning for the sync debug warnings: counted by thread and the innermost sglang frames."""
    if "synchronizing CUDA operation" not in str(message):
        return show_warning(message, category, filename, lineno, file, line)
    else:
        pass
    import traceback

    frames = [
        frame
        for frame in traceback.extract_stack()[:-1]
        if "sglang" in frame.filename and "hop_probe" not in frame.filename
    ][-4:]
    key = " <- ".join(
        f"{frame.filename.rsplit('/sglang_omni/', 1)[-1].rsplit('/sglang/', 1)[-1]}:{frame.lineno} {frame.name}"
        for frame in reversed(frames)
    )
    site = f"{threading.current_thread().name} | {key}"
    with records_lock:
        sync_sites[site] = sync_sites.get(site, 0) + 1


def patch_torch(module):
    import warnings

    global show_warning
    show_warning = warnings.showwarning
    warnings.showwarning = count_sync_site
    warnings.filterwarnings("always", message=".*synchronizing CUDA operation.*")
    module.cuda.set_sync_debug_mode("warn")


def load_reference_builder(tree):
    """The reference tree's TalkerPrefillBuilder class, its prompt build bound to the reference talker_input."""
    name = "sglang_omni.models.qwen3_omni.components.talker_input"
    components = os.path.join(tree, "sglang_omni/models/qwen3_omni/components")

    def load(module_name, path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    reference_input = load("reference_talker_input", os.path.join(components, "talker_input.py"))
    running_input = sys.modules[name]
    sys.modules[name] = reference_input
    try:
        reference_prefill = load(
            "reference_talker_prefill", os.path.join(components, "talker_prefill.py")
        )
    finally:
        sys.modules[name] = running_input
    return reference_prefill.TalkerPrefillBuilder


def patch_talker_prefill(module):
    torch = sys.modules["torch"]
    reference_class = load_reference_builder(TALKER_BUILD_REFERENCE)
    build = module.TalkerPrefillBuilder.build_prompt_prefill

    @functools.wraps(build)
    def build_compared(self, payload, thinker_chunks, *, thinker_done):
        built = build(self, payload, thinker_chunks, thinker_done=thinker_done)
        shadow = object.__new__(reference_class)
        shadow.__dict__ = self.__dict__
        reference = reference_class.build_prompt_prefill(
            shadow, payload, thinker_chunks, thinker_done=thinker_done
        )
        built_rows, reference_rows = list(built["pending_text_queue"]), list(
            reference["pending_text_queue"]
        )
        same = {
            "input_embeds": torch.equal(built["input_embeds"], reference["input_embeds"]),
            "input_ids": torch.equal(built["input_ids"].cpu(), reference["input_ids"].cpu()),
            "pending_text_rows": len(built_rows) == len(reference_rows)
            and all(torch.equal(a, b) for a, b in zip(built_rows, reference_rows)),
            "tts_pad_embed": torch.equal(built["tts_pad_embed"], reference["tts_pad_embed"]),
            "tts_eos_embed": torch.equal(built["tts_eos_embed"], reference["tts_eos_embed"]),
        }
        record(
            {
                "k": "build",
                "rid": payload.request_id,
                "same": same,
                "devices": [str(built["input_embeds"].device), str(built["input_ids"].device)],
            }
        )
        return built

    module.TalkerPrefillBuilder.build_prompt_prefill = build_compared


def flush():
    with records_lock:
        total = sum(sync_sites.values())
        if total != sync_sites_flushed[0]:
            sync_sites_flushed[0] = total
            records.append({"k": "sync", "ns": time.time_ns(), "counts": dict(sync_sites)})
        else:
            pass
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


def record_freeze_count_forever(period_s):
    while True:
        time.sleep(period_s)
        record({"k": "frozen", "count": gc.get_freeze_count()})


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
        record(
            {
                "k": "gc",
                "gen": info["generation"],
                "dur_ns": duration_ns,
                "collected": info["collected"],
                "th": threading.current_thread().name,
            }
        )
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
            owner = (
                getattr(obj, "__module__", None) if callable(obj) else kind.__module__
            )
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
    record(
        {
            "k": "heap",
            "tracked": tracked,
            "full_collect_ns": collect_ns,
            "frozen": gc.get_freeze_count(),
            "top_types": sorted(by_type.items(), key=lambda kv: -kv[1])[:25],
            "top_packages": sorted(by_package.items(), key=lambda kv: -kv[1])[:25],
        }
    )


def instrument_queue(q, label):
    if getattr(q, "hop_probe_label", None):
        return
    else:
        pass
    q.hop_probe_label = label
    put, get = q.put, q.get

    def put_recorded(item, *args, **kwargs):
        record_queue(time.time_ns(), label, "put", item)
        if (
            not heap_taken
            and label.endswith(".in")
            and getattr(item, "type", None) == "new_request"
        ):
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
                names.get(ident, str(ident)): [
                    f"{f.filename}:{f.lineno} {f.name}"
                    for f in traceback.extract_stack(frame)
                ][-14:]
                for ident, frame in frames.items()
            }
            record(
                {
                    "k": "stall",
                    "loop": name,
                    "quiet_ms": quiet * 1e3,
                    "loop_thread": names.get(loop_thread_id, str(loop_thread_id)),
                    "stacks": stacks,
                }
            )
        elif quiet <= STALL_S:
            stalled = False
        else:
            pass


async def sample_lag(name):
    import asyncio

    loop = asyncio.get_running_loop()
    ticks = [time.monotonic()]
    threading.Thread(
        target=watch_stalls,
        args=(name, ticks, threading.get_ident()),
        name=f"hop-probe-stall-{name}",
        daemon=True,
    ).start()
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
            record(
                {
                    "k": "lag",
                    "loop": name,
                    "n": count,
                    "sum_ms": total * 1e3,
                    "max_ms": worst * 1e3,
                }
            )
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


def patch_talker(module):
    talker = module.Qwen3OmniTalker
    prepare = talker.prepare_decode_buffers

    @functools.wraps(prepare)
    def prepare_recorded(self, requests):
        record(
            {
                "k": "tb",
                "rows": len(requests),
                "rids": [request.data.req.rid for request in requests],
            }
        )
        return prepare(self, requests)

    talker.prepare_decode_buffers = prepare_recorded


def patch_preprocessor(module):
    preprocessor = module.Qwen3OmniPreprocessor
    call_impl = preprocessor.call_impl

    @functools.wraps(call_impl)
    async def call_recorded(self, payload):
        # note: a probe must not fail a request, so an input of another shape records no text
        try:
            content = payload.request.inputs["messages"][-1]["content"]
        except (AttributeError, IndexError, KeyError, TypeError):
            content = None
        record(
            {
                "k": "req",
                "rid": payload.request_id,
                "text": content[:300] if isinstance(content, str) else None,
            }
        )
        return await call_impl(self, payload)

    preprocessor.call_impl = call_recorded


def bind_thread_device(fixed_device):
    """New threads set a CUDA device before their target runs: fixed_device when given, else the device that was
    current in the creating thread."""
    start, run = threading.Thread.start, threading.Thread.run

    def start_capturing(self, *args, **kwargs):
        torch = sys.modules.get("torch")
        if torch is not None and fixed_device is not None:
            self.bound_device = fixed_device
        elif torch is not None and torch.cuda.is_initialized():
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
if SYNC_SITES:
    PATCHES["torch"] = patch_torch
else:
    pass
if TALKER_BUILD_REFERENCE:
    PATCHES["sglang_omni.models.qwen3_omni.components.talker_prefill"] = (
        patch_talker_prefill
    )
else:
    pass
if TALKER_BATCHES:
    PATCHES["sglang_omni.models.qwen3_omni.components.talker"] = patch_talker
    PATCHES["sglang_omni.models.qwen3_omni.components.preprocessor"] = (
        patch_preprocessor
    )
else:
    pass


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
                print(
                    f"hop_probe patch {name} failed: {exc!r}",
                    file=sys.stderr,
                    flush=True,
                )

        spec.loader.exec_module = exec_and_patch
        return spec


if PREFIX:
    sys.meta_path.insert(0, PatchOnImport())
    gc.callbacks.append(on_gc)
    threading.Thread(target=flush_forever, name="hop-probe-flush", daemon=True).start()
    if FREEZE_COUNT_PERIOD_S > 0:
        threading.Thread(
            target=record_freeze_count_forever,
            args=(FREEZE_COUNT_PERIOD_S,),
            name="hop-probe-freeze-count",
            daemon=True,
        ).start()
    else:
        pass
    atexit.register(flush)
    print(f"hop_probe on, prefix {PREFIX}", file=sys.stderr, flush=True)
else:
    pass
if BIND_THREAD_DEVICE == "1":
    bind_thread_device(None)
    print(
        "hop_probe what-if: new threads bind their creator's CUDA device",
        file=sys.stderr,
        flush=True,
    )
elif BIND_THREAD_DEVICE.startswith("cuda:"):
    bind_thread_device(int(BIND_THREAD_DEVICE.split(":")[1]))
    print(
        f"hop_probe what-if: new threads bind {BIND_THREAD_DEVICE}",
        file=sys.stderr,
        flush=True,
    )
else:
    pass
if FREEZE_STARTUP_GC:
    print(
        "hop_probe what-if: the heap is frozen at the first new request",
        file=sys.stderr,
        flush=True,
    )
else:
    pass
