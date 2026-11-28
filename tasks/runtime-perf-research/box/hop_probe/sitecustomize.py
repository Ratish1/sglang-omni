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
              ids, at the decode prep's start (QwenTalkerModelRunner.before_decode)
  kind "tr"   with OMNI_HOP_PROBE_TALKER_BATCHES=1, each talker step's result processing (decode and prefill): its
              forward mode and rows as it starts (ns0) and its end (ns), after the step's outputs reached the host
  kind "te"   with OMNI_HOP_PROBE_TALKER_BATCHES=1, each talker decode batch update with a row not ready: its rows and,
              per row not ready, its request id and what it waits for (its feedback row, its next text, the pad that
              follows the thinker's last text)
  kind "req"  with OMNI_HOP_PROBE_TALKER_BATCHES=1, each Qwen3-Omni request at preprocessing: its id and the text of
              its last message, which joins a benchmark sample to its request id
  kind "ts"   with OMNI_HOP_PROBE_THINKER_STEPS=1, each thinker step run on the synchronous path (prefill, mixed, and
              decode below the lookahead size): its forward mode, row request ids, extend tokens and the rows whose
              extend is longer than one token as [row, tokens], from its start (ns0) to the end of its stream output (ns)
  kind "tl"   with OMNI_HOP_PROBE_THINKER_STEPS=1, each thinker step launched on the lookahead path, at its launch: its
              forward mode and row request ids
  kind "tz"   with OMNI_HOP_PROBE_THINKER_STEPS=1, each lookahead step's resolve, from its start (ns0, the wait for the
              device) to the end of its stream output (ns); resolves run in launch order
  kind "tq"   with OMNI_HOP_PROBE_THINKER_STEPS=1, each thinker loop pass that received inputs: their count (n) and the
              time from the start of their processing (ns0, request builds run inline there) to its end (ns)
  kind "tg"   with OMNI_HOP_PROBE_THINKER_STEPS=1, each thinker batch selection that returned an extend or mixed batch,
              or took over 2 ms: the batch's forward mode, from its start (ns0) to its end (ns)
  kind "td"   with OMNI_HOP_PROBE_THINKER_STEPS=1, each drain of a pending lookahead step (resolve_pending_async), from
              its start (ns0) to its end (ns)
  kind "tp"   with OMNI_HOP_PROBE_THINKER_STEPS=1, each thinker step's result processing (process_batch_result): the
              step's forward mode, from its start (ns0) to its end (ns)
  kind "tx"   with OMNI_HOP_PROBE_THINKER_PARTS=1, each synchronous Qwen3-Omni thinker execute: forward mode, rows,
              extend tokens, host ms in build_forward_batch, prepare_and_forward and finalize (its host token wait
              included), the whole execute from ns0 to ns (the rest is the post_prefill or post_decode hooks and token
              publication), the forward's device span in ms (gpu, CUDA events around prepare_and_forward, time the
              card served other contexts included), whether the forward replayed a graph (graph), the longest row
              context in tokens (kv) and the rows over 4,096 tokens (long)
  kind "rp"   with OMNI_HOP_PROBE_PAYLOAD_READS=1, each stage payload read (comm.stage_io.read_payload) in every
              process: request id, transfer bytes, the tensors of 1 MiB or more as [path, MiB], the reading thread, from
              its start (ns0) to its end (ns)
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
THINKER_STEPS = os.environ.get("OMNI_HOP_PROBE_THINKER_STEPS") == "1"
# OMNI_HOP_PROBE_THINKER_PARTS=1 times the parts of each synchronous Qwen3-Omni thinker step and its forward's device span
# from two CUDA events; reading the span adds one event wait after the step's own finalize wait
THINKER_PARTS = os.environ.get("OMNI_HOP_PROBE_THINKER_PARTS") == "1"
# OMNI_HOP_PROBE_TURNS=1 records each wait for a shared process's engine turn over 0.05 ms (kind "tw") and each leave
# that hands the turn to a waiter (kind "th": the leaver's priority, its hold span and the sglang_omni frames it left from)
TURNS = os.environ.get("OMNI_HOP_PROBE_TURNS") == "1"
# OMNI_HOP_PROBE_FA_NUM_SPLITS=<n> sets every FlashAttention backend's KV split count to n, a measurement arm only: it
# removes the split count as a rounding channel so two trees' quality can be compared without it
FA_NUM_SPLITS = os.environ.get("OMNI_HOP_PROBE_FA_NUM_SPLITS", "")
PAYLOAD_READS = os.environ.get("OMNI_HOP_PROBE_PAYLOAD_READS") == "1"
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

    reference_input = load(
        "reference_talker_input", os.path.join(components, "talker_input.py")
    )
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
            "input_embeds": torch.equal(
                built["input_embeds"], reference["input_embeds"]
            ),
            "input_ids": torch.equal(
                built["input_ids"].cpu(), reference["input_ids"].cpu()
            ),
            "pending_text_rows": len(built_rows) == len(reference_rows)
            and all(torch.equal(a, b) for a, b in zip(built_rows, reference_rows)),
            "tts_pad_embed": torch.equal(
                built["tts_pad_embed"], reference["tts_pad_embed"]
            ),
            "tts_eos_embed": torch.equal(
                built["tts_eos_embed"], reference["tts_eos_embed"]
            ),
        }
        record(
            {
                "k": "build",
                "rid": payload.request_id,
                "same": same,
                "devices": [
                    str(built["input_embeds"].device),
                    str(built["input_ids"].device),
                ],
            }
        )
        return built

    module.TalkerPrefillBuilder.build_prompt_prefill = build_compared


def flush():
    with records_lock:
        total = sum(sync_sites.values())
        if total != sync_sites_flushed[0]:
            sync_sites_flushed[0] = total
            records.append(
                {"k": "sync", "ns": time.time_ns(), "counts": dict(sync_sites)}
            )
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


def patch_talker_runner(module):
    patch_put_messages(module)
    runner = module.QwenTalkerModelRunner
    before_decode = runner.before_decode

    @functools.wraps(before_decode)
    def before_decode_recorded(self, forward_batch, schedule_batch, requests, **kwargs):
        record(
            {
                "k": "tb",
                "rows": len(requests),
                "rids": [request.data.req.rid for request in requests],
            }
        )
        return before_decode(self, forward_batch, schedule_batch, requests, **kwargs)

    runner.before_decode = before_decode_recorded


def talker_row_wait(data):
    if data is None:
        return "data"
    elif not data.pending_feedback_queue:
        return "feedback"
    elif data.pending_text_queue:
        return None
    elif not data.thinker_chunks_done:
        return "text"
    elif data.tts_pad_embed is None:
        return "pad"
    else:
        return None


def patch_talker_scheduler(module):
    scheduler = module.QwenTalkerScheduler
    process = scheduler.process_batch_result
    update_running_batch = scheduler.update_running_batch

    @functools.wraps(update_running_batch)
    def update_running_batch_recorded(self, batch):
        if getattr(self.model_runner, "feedback_enabled", False):
            waits = [
                [request.rid, wait]
                for request in batch.reqs
                if (wait := talker_row_wait(getattr(request, "omni_data", None)))
            ]
            if waits:
                record({"k": "te", "rows": len(batch.reqs), "waits": waits})
            else:
                pass
        else:
            pass
        return update_running_batch(self, batch)

    scheduler.update_running_batch = update_running_batch_recorded

    @functools.wraps(process)
    def process_recorded(self, batch, result):
        started_ns = time.time_ns()
        mode = batch.forward_mode.name
        rows = len(batch.reqs)
        try:
            return process(self, batch, result)
        finally:
            record({"k": "tr", "ns0": started_ns, "mode": mode, "rows": rows})

    scheduler.process_batch_result = process_recorded


def thinker_step_shape(batch):
    mode = batch.forward_mode
    shape = {
        "mode": mode.name,
        "rids": [request.rid for request in batch.reqs],
    }
    if mode.is_extend():
        extend_lens = list(batch.extend_lens or ())
        shape["ext"] = batch.extend_num_tokens
        shape["pre"] = [
            [row, tokens] for row, tokens in enumerate(extend_lens) if tokens > 1
        ]
    else:
        pass
    return shape


def patch_thinker_scheduler(module):
    patch_put_messages(module)
    scheduler = module.OmniScheduler
    run_batch, launch, resolve = (
        scheduler._run_batch,
        scheduler.run_batch_launch,
        scheduler.run_batch_resolve,
    )

    @functools.wraps(run_batch)
    def run_batch_recorded(self, batch, pp_proxy_tensors=None):
        if type(self) is not scheduler:
            return run_batch(self, batch, pp_proxy_tensors)
        else:
            pass
        started_ns = time.time_ns()
        shape = thinker_step_shape(batch)
        try:
            return run_batch(self, batch, pp_proxy_tensors)
        finally:
            record({"k": "ts", "ns0": started_ns, **shape})

    @functools.wraps(launch)
    def launch_recorded(self, batch):
        if type(self) is scheduler:
            record({"k": "tl", **thinker_step_shape(batch)})
        else:
            pass
        return launch(self, batch)

    @functools.wraps(resolve)
    def resolve_recorded(self, batch, sched_output, pending_step, skip_rids=()):
        if type(self) is not scheduler:
            return resolve(self, batch, sched_output, pending_step, skip_rids)
        else:
            pass
        started_ns = time.time_ns()
        try:
            return resolve(self, batch, sched_output, pending_step, skip_rids)
        finally:
            record({"k": "tz", "ns0": started_ns, "rows": len(batch.reqs)})

    process_input = scheduler.process_input_requests

    @functools.wraps(process_input)
    def process_input_recorded(self, recv_reqs):
        if type(self) is not scheduler or not recv_reqs:
            return process_input(self, recv_reqs)
        else:
            pass
        started_ns = time.time_ns()
        count = len(recv_reqs)
        try:
            return process_input(self, recv_reqs)
        finally:
            record({"k": "tq", "ns0": started_ns, "n": count})

    next_batch, drain = scheduler.get_next_batch_to_run, scheduler.resolve_pending_async
    process_result = scheduler.process_batch_result

    @functools.wraps(process_result)
    def process_result_recorded(self, batch, result):
        if type(self) is not scheduler:
            return process_result(self, batch, result)
        else:
            pass
        started_ns = time.time_ns()
        mode = batch.forward_mode.name
        try:
            return process_result(self, batch, result)
        finally:
            record({"k": "tp", "ns0": started_ns, "mode": mode})

    @functools.wraps(next_batch)
    def next_batch_recorded(self):
        if type(self) is not scheduler:
            return next_batch(self)
        else:
            pass
        started_ns = time.time_ns()
        batch = next_batch(self)
        mode = None if batch is None else batch.forward_mode.name
        if mode not in (None, "DECODE") or time.time_ns() - started_ns > 2_000_000:
            record({"k": "tg", "ns0": started_ns, "mode": mode})
        else:
            pass
        return batch

    @functools.wraps(drain)
    def drain_recorded(self):
        if type(self) is not scheduler or self.async_pending is None:
            return drain(self)
        else:
            pass
        started_ns = time.time_ns()
        try:
            return drain(self)
        finally:
            record({"k": "td", "ns0": started_ns})

    scheduler._run_batch, scheduler.run_batch_launch, scheduler.run_batch_resolve = (
        run_batch_recorded,
        launch_recorded,
        resolve_recorded,
    )
    scheduler.process_input_requests = process_input_recorded
    scheduler.get_next_batch_to_run = next_batch_recorded
    scheduler.resolve_pending_async = drain_recorded
    scheduler.process_batch_result = process_result_recorded


def patch_stage_io(module):
    read_payload = module.read_payload

    @functools.wraps(read_payload)
    async def read_payload_recorded(relay, request_id, data_ref, *args, **kwargs):
        started_ns = time.time_ns()
        try:
            return await read_payload(relay, request_id, data_ref, *args, **kwargs)
        finally:
            record(
                {
                    "k": "rp",
                    "ns0": started_ns,
                    "rid": request_id,
                    "bytes": (
                        int(data_ref.buffer.length)
                        if data_ref.buffer is not None
                        else 0
                    ),
                    "tensors": [
                        [entry.path, round(entry.size / 2**20, 1)]
                        for entry in data_ref.tensors
                        if entry.size >= 2**20
                    ],
                    "thread": threading.current_thread().name,
                }
            )

    module.read_payload = read_payload_recorded


THINKER_RUNNERS = ("Qwen3OmniThinkerModelRunner",)


def patch_thinker_runner(module):
    runner = module.ModelRunner
    execute = runner.execute
    timed = {
        name: getattr(runner, name)
        for name in ("build_forward_batch", "prepare_and_forward", "finalize")
    }

    def timed_part(name, method):
        @functools.wraps(method)
        def part_recorded(self, *args, **kwargs):
            parts = getattr(self, "hop_probe_parts", None)
            if parts is None:
                return method(self, *args, **kwargs)
            else:
                pass
            torch = sys.modules["torch"]
            started_ns = time.time_ns()
            if name == "prepare_and_forward":
                parts["gpu"] = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                parts["gpu"][0].record()
            else:
                pass
            result = None
            try:
                result = method(self, *args, **kwargs)
                return result
            finally:
                if name == "prepare_and_forward":
                    parts["gpu"][1].record()
                    parts["graph"] = bool(getattr(result, "can_run_cuda_graph", False))
                else:
                    pass
                parts[name] = (time.time_ns() - started_ns) / 1e6

        return part_recorded

    @functools.wraps(execute)
    def execute_recorded(self, scheduler_output):
        batch = scheduler_output.batch_data
        if type(self).__name__ not in THINKER_RUNNERS or batch is None:
            return execute(self, scheduler_output)
        else:
            pass
        mode = batch.forward_mode.name
        rows = len(batch.reqs)
        ext = batch.extend_num_tokens if batch.forward_mode.is_extend() else rows
        contexts = [
            len(req.origin_input_ids) + len(req.output_ids) for req in batch.reqs
        ]
        self.hop_probe_parts = {}
        started_ns = time.time_ns()
        try:
            return execute(self, scheduler_output)
        finally:
            parts, self.hop_probe_parts = self.hop_probe_parts, None
            item = {
                "k": "tx",
                "ns0": started_ns,
                "mode": mode,
                "rows": rows,
                "ext": ext,
                "kv": max(contexts, default=0),
                "long": sum(context > 4096 for context in contexts),
            }
            item.update({name: value for name, value in parts.items() if name != "gpu"})
            events = parts.get("gpu")
            if events is not None:
                events[1].synchronize()
                item["gpu"] = events[0].elapsed_time(events[1])
            else:
                pass
            record(item)

    for name, method in timed.items():
        setattr(runner, name, timed_part(name, method))
    runner.execute = execute_recorded


def patch_preprocessor(module):
    preprocessor = module.Qwen3OmniPreprocessor
    call_impl = preprocessor.call_impl

    @functools.wraps(call_impl)
    async def call_recorded(self, payload, *read_inputs):
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
        return await call_impl(self, payload, *read_inputs)

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


def leave_site(frame):
    """The first three sglang_omni frames above the registry, innermost first, as name:line."""
    site = []
    while frame is not None and len(site) < 3:
        path = frame.f_code.co_filename
        if (
            "sglang_omni" in path
            and not path.endswith("process_engines.py")
            and "hop_probe" not in path
        ):
            site.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
        else:
            pass
        frame = frame.f_back
    return " < ".join(site)


def patch_flash_attention(module):
    backend = module.FlashAttentionBackend
    init = backend.__init__

    @functools.wraps(init)
    def init_with_splits(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.num_splits = int(FA_NUM_SPLITS)

    backend.__init__ = init_with_splits


def patch_process_engines(module):
    engines = module.ProcessEngines
    take_turn, leave_turn = engines.take_turn, engines.leave_turn

    @functools.wraps(take_turn)
    def take_recorded(self, held_engine):
        started_ns = time.time_ns()
        take_turn(self, held_engine)
        taken_ns = time.time_ns()
        held_engine.hop_probe_taken_ns = taken_ns
        if taken_ns - started_ns > 50_000:
            record(
                {
                    "k": "tw",
                    "ns0": started_ns,
                    "ns": taken_ns,
                    "p": held_engine.priority,
                }
            )
        else:
            pass

    @functools.wraps(leave_turn)
    def leave_recorded(self, held_engine):
        with self.turn_state_lock:
            handing = bool(self.turn_waiters)
        if handing:
            left_ns = time.time_ns()
            record(
                {
                    "k": "th",
                    "ns0": getattr(held_engine, "hop_probe_taken_ns", left_ns),
                    "ns": left_ns,
                    "p": held_engine.priority,
                    "site": leave_site(sys._getframe(1)),
                }
            )
        else:
            pass
        leave_turn(self, held_engine)

    engines.take_turn, engines.leave_turn = take_recorded, leave_recorded


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
    PATCHES["sglang_omni.models.qwen3_omni.talker_model_runner"] = patch_talker_runner
    PATCHES["sglang_omni.models.qwen3_omni.talker_scheduler"] = patch_talker_scheduler
    PATCHES["sglang_omni.models.qwen3_omni.components.preprocessor"] = (
        patch_preprocessor
    )
else:
    pass
if THINKER_STEPS:
    PATCHES["sglang_omni.scheduling.omni_scheduler"] = patch_thinker_scheduler
else:
    pass
if THINKER_PARTS:
    PATCHES["sglang_omni.model_runner.base"] = patch_thinker_runner
else:
    pass
if PAYLOAD_READS:
    PATCHES["sglang_omni.comm.stage_io"] = patch_stage_io
else:
    pass
if TURNS:
    PATCHES["sglang_omni.model_runner.process_engines"] = patch_process_engines
else:
    pass
if FA_NUM_SPLITS:
    PATCHES["sglang.srt.layers.attention.flashattention_backend"] = (
        patch_flash_attention
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
