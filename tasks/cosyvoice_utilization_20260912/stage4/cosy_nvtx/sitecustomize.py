"""NVTX ranges and marks over every Fun-CosyVoice3 component, hop and graph capture, for nsys.

Put this directory on PYTHONPATH and set OMNI_PIPE_NVTX=1. No runtime code changes: the
functions are wrapped at import. Ranges are thread local and nested, marks are instants.
pipeline_census.py joins them with the kernels, copies, syncs, OS runtime waits and GIL
ranges of the same nsys report. The generic part (threads, queues, captures, the
coordinator, the OmniScheduler and the model runner) is the Qwen3-TTS probe's
(feat/omni-step-profiler, pipeline_nvtx), unchanged.

OMNI_PIPE_LINES=capture adds a range per dispatched op inside every CUDA graph capture,
named by the op and its call sites, so every replayed node kernel resolves to the line
that captured it. OMNI_PIPE_LINES=all also labels eager ops on every probed thread (for
attribution only, never timing). Compiled callables run with the label mode popped,
because Dynamo runs a compiled function's eager ops when a dispatch mode is active.

Names (rid = request id, the first word is the kind the census groups by):

  thread name=N native=T                   mark, once per thread
  q <stage>.<in|out>.<put|get> rid=R t=T   mark on every stage inbox and outbox hop
  coord.recv rid=R n=S                     mark, coordinator receives S audio samples
  pre.payload rid=R, pre.prepare           preprocessing of one request
  pre.reference, pre.load, pre.speaker, pre.tokenize, pre.mel, pre.embeds
  sched.<recv|input|pick|idle|emit|build>  AR scheduler loop phases
  sched.batch extend|decode bs=B           one AR forward, sched.result its result
  mark sched.admit rid=R, sched.prefill rid=R
  mr.*                                     AR model runner (forward, prefill_embeds,
                                           sample, collect, emit, finalize, d2h)
  voc.ingest rid=R                         speech tokens of one chunk ingested
  voc.collect chunks|buffered              gather window of the vocoder loop
  voc.chunks n=N, voc.buffered n=N         one chunk batch, one buffered batch
  voc.select, voc.step <plan> rows=N       one streaming step and its participant pick
  mark voc.steprid rid=R plan=P            one participant of that step
  mark voc.ready rid=R plan=P tokens=N at=ingest|step  a stream turns runnable
  mark voc.rank rid=R plan=P slack=S chosen=C sel=K  every runnable stream at selection
                                           K, its playback slack in ms (u = no audio
                                           yet) and whether the step took it
  voc.done rid=R, voc.finish rid=R         stream end and its flush
  flow.hop rows=N tok=T                    causal packed Flow over N rows
  flow.hop.prefix rows=N tok=T             the same over the frames past each row's
                                           cached prefix (#2406)
  flow.leftover rows=N tok=T               full context packed Flow (stream finals)
  flow.buffered rows=N                     padded Flow of a buffered group
  flow.graph rows=B frames=F               Flow CUDA graph replay
  flow.eager streaming=S                   padded eager Euler solve
  flow.packed causal|full, flow.euler_packed rows=N
  flow.cond, flow.pack, flow.split         conditioning, input packing, mel split
  hift.delta frames=F final=X              streaming HiFT over one request's history
  hift.batch n=N                           buffered HiFT group
  hift.step rows=N final=X, hift.group rows=N  batched streaming HiFT step and its decodes
  cap <site>                               one CUDA graph capture
    op <aten op> @<library site> <omni site>
  compiled                                 a compiled callable run with labels popped
"""

import functools
import importlib.abc
import importlib.util
import os
import sys
import threading

ENABLED = os.environ.get("OMNI_PIPE_NVTX") == "1"
LINES = os.environ.get("OMNI_PIPE_LINES", "")
ROOTS = (
    "sglang_omni/",
    "sglang/",
    "cosyvoice/",
    "matcha/",
    "x_transformers/",
    "transformers/",
    "flashinfer/",
    "sgl_kernel/",
    "torchaudio/",
)
PROBE_FILE = os.path.abspath(__file__)
local = threading.local()
nvtx = None


def torch_nvtx():
    global nvtx
    if nvtx is None:
        import torch

        nvtx = torch.cuda.nvtx
    return nvtx


def note_thread():
    if getattr(local, "named", False):
        return
    local.named = True
    thread = threading.current_thread()
    torch_nvtx().mark(f"thread name={thread.name} native={threading.get_native_id()}")
    if LINES == "all":
        enter_labels()


def mark(text):
    note_thread()
    torch_nvtx().mark(text)


def label_of(name, fn, args, kwargs):
    """The label, or the function's own name when the label cannot be built."""
    try:
        return name(*args, **kwargs)
    except Exception:
        return getattr(fn, "__qualname__", "unnamed")


def ranged(fn, name):
    """Wrap fn in an NVTX range named by name(*args, **kwargs)."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        note_thread()
        torch_nvtx().range_push(label_of(name, fn, args, kwargs))
        try:
            return fn(*args, **kwargs)
        finally:
            torch_nvtx().range_pop()

    return wrapped


def ranged_async(fn, name):
    """A coroutine's range; the vocoder loop runs one coroutine at a time on its thread."""

    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        note_thread()
        torch_nvtx().range_push(label_of(name, fn, args, kwargs))
        try:
            return await fn(*args, **kwargs)
        finally:
            torch_nvtx().range_pop()

    return wrapped


def marked_async(fn, name):
    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        mark(label_of(name, fn, args, kwargs))
        return await fn(*args, **kwargs)

    return wrapped


def fixed(label):
    return lambda *args, **kwargs: label


def rid_of(item):
    rid = getattr(item, "request_id", None)
    if rid is None and isinstance(item, tuple):
        for part in item:
            if isinstance(part, str):
                return part
    return rid


def instrument_queue(q, label):
    """Mark every put before it happens and every get after it returns."""
    if getattr(q, "pipe_nvtx_label", None):
        return
    q.pipe_nvtx_label = label
    put, get = q.put, q.get

    def put_marked(item, *args, **kwargs):
        mark(f"q {label}.put rid={rid_of(item)} t={getattr(item, 'type', '-')}")
        return put(item, *args, **kwargs)

    def get_marked(*args, **kwargs):
        item = get(*args, **kwargs)
        mark(f"q {label}.get rid={rid_of(item)} t={getattr(item, 'type', '-')}")
        return item

    # note: the stdlib put_nowait and get_nowait call self.put and self.get, so the
    # instance wrappers see them too; wrapping them as well would mark twice
    q.put, q.get = put_marked, get_marked


def call_sites():
    """Innermost library or omni frame, and innermost omni frame, outside torch."""
    frame = sys._getframe(2)
    first, omni = None, None
    while frame is not None and omni is None:
        filename = frame.f_code.co_filename
        if filename != PROBE_FILE and "/torch/" not in filename:
            for root in ROOTS:
                index = filename.rfind(root)
                if index >= 0:
                    site = f"{filename[index:]}:{frame.f_lineno}"
                    if first is None:
                        first = site
                    if root == "sglang_omni/":
                        omni = site
                    break
        frame = frame.f_back
    return f"@{first or '?'} <{omni or '?'}>"


def label_mode_class():
    from torch.utils._python_dispatch import TorchDispatchMode

    class LineLabels(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            push = torch_nvtx().range_push
            push(f"op {func.__name__} {call_sites()}")
            try:
                return func(*args, **(kwargs or {}))
            finally:
                torch_nvtx().range_pop()

    return LineLabels


def enter_labels():
    if getattr(local, "labels", None) is not None:
        return
    local.labels = label_mode_class()()
    local.labels.__enter__()


def exit_labels():
    labels = getattr(local, "labels", None)
    if labels is None:
        return
    local.labels = None
    labels.__exit__(None, None, None)


def patch_torch(module):
    if not LINES:
        return
    compile_ = module.compile

    def popping_labels(compiled):
        if not callable(compiled) or isinstance(compiled, module.nn.Module):
            return compiled

        @functools.wraps(compiled)
        def run(*call_args, **call_kwargs):
            if getattr(local, "labels", None) is None:
                return compiled(*call_args, **call_kwargs)
            from torch.utils._python_dispatch import _disable_current_modes

            torch_nvtx().range_push("compiled")
            try:
                with _disable_current_modes():
                    return compiled(*call_args, **call_kwargs)
            finally:
                torch_nvtx().range_pop()

        return run

    def compile_popping_labels(*args, **kwargs):
        if args and callable(args[0]):
            return popping_labels(compile_(*args, **kwargs))
        decorator = compile_(*args, **kwargs)
        return lambda fn: popping_labels(decorator(fn))

    module.compile = compile_popping_labels


def patch_graphs(module):
    graph = module.CUDAGraph
    begin, end = graph.capture_begin, graph.capture_end

    def begin_marked(self, *args, **kwargs):
        note_thread()
        torch_nvtx().range_push(f"cap {call_sites()}")
        result = begin(self, *args, **kwargs)
        if LINES and getattr(local, "labels", None) is None:
            enter_labels()
            self.pipe_nvtx_labels = True
        return result

    def end_marked(self, *args, **kwargs):
        if getattr(self, "pipe_nvtx_labels", False):
            exit_labels()
            self.pipe_nvtx_labels = False
        try:
            return end(self, *args, **kwargs)
        finally:
            torch_nvtx().range_pop()

    graph.capture_begin, graph.capture_end = begin_marked, end_marked


def patch_runtime(module):
    stage = module.Stage
    start = stage.start

    @functools.wraps(start)
    async def start_marked(self):
        scheduler = self.scheduler
        if scheduler is not None:
            for attr, short in (("inbox", "in"), ("outbox", "out")):
                q = getattr(scheduler, attr, None)
                if q is not None and hasattr(q, "get_nowait"):
                    instrument_queue(q, f"{self.name}.{short}")
        return await start(self)

    stage.start = start_marked


def coordinator_label(self, msg):
    """The request and the chunk's sample count, from the audio payload's shape."""
    chunk = getattr(msg, "chunk", None)
    shape = chunk.get("audio_waveform_shape") if isinstance(chunk, dict) else None
    samples = shape[0] if shape else "-"
    return f"coord.recv rid={getattr(msg, 'request_id', None)} n={samples}"


def patch_coordinator(module):
    coordinator = module.Coordinator
    coordinator.handle_stream = marked_async(
        coordinator.handle_stream, coordinator_label
    )


def batch_label(prefix, batch):
    mode = "extend" if batch.forward_mode.is_extend() else "decode"
    label = f"{prefix} {mode} bs={len(batch.reqs)}"
    if mode == "extend":
        label += f" toks={batch.extend_num_tokens}"
    return label


def patch_scheduler(module):
    scheduler = module.OmniScheduler
    run_batch = scheduler._run_batch

    def run_batch_marked(self, batch, pp_proxy_tensors=None):
        if batch.forward_mode.is_extend():
            for req in batch.reqs:
                mark(f"sched.prefill rid={req.rid}")
        return run_batch(self, batch, pp_proxy_tensors)

    scheduler._run_batch = ranged(
        run_batch_marked,
        lambda self, batch, pp_proxy_tensors=None: batch_label("sched.batch", batch),
    )
    scheduler.process_batch_result = ranged(
        scheduler.process_batch_result,
        lambda self, batch, result: batch_label("sched.result", batch),
    )
    scheduler.run_request_builder = ranged(
        scheduler.run_request_builder,
        lambda self, payload, active_stage: f"sched.build rid={payload.request_id}",
    )
    scheduler.recv_requests = ranged(scheduler.recv_requests, fixed("sched.recv"))
    scheduler.process_input_requests = ranged(
        scheduler.process_input_requests,
        lambda self, recv_reqs: f"sched.input n={len(recv_reqs or ())}",
    )
    scheduler.get_next_batch_to_run = ranged(
        scheduler.get_next_batch_to_run, fixed("sched.pick")
    )
    scheduler.sleep_during_idle = ranged(
        scheduler.sleep_during_idle, fixed("sched.idle")
    )
    scheduler.emit_stream_output = ranged(
        scheduler.emit_stream_output, fixed("sched.emit")
    )
    enqueue = scheduler.enqueue_built_request

    def enqueue_marked(self, payload, pending_stream_done, req_data, **kwargs):
        mark(f"sched.admit rid={payload.request_id}")
        return enqueue(self, payload, pending_stream_done, req_data, **kwargs)

    scheduler.enqueue_built_request = enqueue_marked


def patch_base_runner(module):
    runner = module.ModelRunner
    for name, label in (
        ("execute", "mr.execute"),
        ("build_forward_batch", "mr.build_fb"),
        ("prepare_and_forward", "mr.prepare_forward"),
        ("finalize", "mr.finalize"),
        ("resolve_host_token_ids", "mr.wait_d2h"),
        ("stage_token_ids", "mr.stage_d2h"),
        ("publish_next_tokens", "mr.publish"),
    ):
        setattr(runner, name, ranged(getattr(runner, name), fixed(label)))


def patch_worker(module):
    worker = module.ModelWorker
    worker.forward_batch_generation = ranged(
        worker.forward_batch_generation, fixed("mr.forward")
    )


def patch_cosy_utils(module):
    module.SpeechTokenizerV3.extract_speech_token = ranged(
        module.SpeechTokenizerV3.extract_speech_token, fixed("pre.tokenize")
    )
    module.SpeakerEncoder.extract_embedding = ranged(
        module.SpeakerEncoder.extract_embedding, fixed("pre.speaker")
    )


def patch_cosy_builders(module):
    module.preprocess_cosyvoice3_payload = ranged(
        module.preprocess_cosyvoice3_payload,
        lambda payload: f"pre.payload rid={payload.request_id}",
    )
    module.prepare_cosyvoice3_request = ranged(
        module.prepare_cosyvoice3_request, fixed("pre.prepare")
    )
    for name, label in (
        ("load_prompt_audio", "pre.load"),
        ("load_prompt_audio_24k", "pre.load"),
        ("extract_prompt_speech_feat", "pre.mel"),
        ("build_llm_prompt_embeddings", "pre.embeds"),
    ):
        setattr(module, name, ranged(getattr(module, name), fixed(label)))
    hook = module.CosyVoice3ReferenceEncodeHook
    hook.encode_one = ranged(hook.encode_one, fixed("pre.reference"))


def patch_cosy_runner(module):
    runner = module.FunCosyVoice3ModelRunner
    for name, label in (
        ("custom_prefill_forward", "mr.prefill_embeds"),
        ("sample_next_token_ids", "mr.sample"),
        ("collect_tokens", "mr.collect"),
    ):
        setattr(runner, name, ranged(getattr(runner, name), fixed(label)))
    runner.emit_code_chunk = ranged(
        runner.emit_code_chunk,
        lambda self, request_id, data, codes: f"mr.emit rid={request_id}",
    )


def token_count(items):
    return sum(int(item.token.shape[1]) for item in items)


def patch_cosy_stages(module):
    for name, label in (
        ("prepare_flow_conditioning", fixed("flow.cond")),
        ("pack_flow_inputs", fixed("flow.pack")),
        ("split_generated_mels", fixed("flow.split")),
        (
            "solve_flow_euler",
            lambda *args, streaming=False, **kwargs: f"flow.eager streaming={streaming}",
        ),
        (
            "generate_flow_packed",
            lambda flow, packed, *, streaming, finalize: (
                f"flow.packed {'causal' if streaming else 'full'}"
            ),
        ),
        (
            "solve_flow_euler_packed",
            lambda estimator, noise, *args, **kwargs: (
                f"flow.euler_packed frames={int(noise.shape[1])}"
            ),
        ),
    ):
        setattr(module, name, ranged(getattr(module, name), label))
    # the prefix cached solve of #2406, looked up through the stages module
    if hasattr(module, "solve_flow_euler_prefix"):
        module.solve_flow_euler_prefix = ranged(
            module.solve_flow_euler_prefix,
            lambda estimator, pool, noise, *args, **kwargs: (
                f"flow.euler_prefix frames={int(noise.shape[1])}"
            ),
        )
    runner = module.FlowCudaGraphRunner
    runner.run = ranged(
        runner.run,
        lambda self, noisy_mel, *args, **kwargs: (
            f"flow.graph rows={int(noisy_mel.shape[0])} frames={int(noisy_mel.shape[2])}"
        ),
    )
    flow = module.FunCosyVoice3Flow
    flow.inference = ranged(
        flow.inference, lambda self, inputs: f"flow.buffered rows={len(inputs)}"
    )
    vocoder = module.CosyVoice3Vocoder
    vocoder.hop_batch = ranged(
        vocoder.hop_batch,
        lambda self, items: f"flow.hop rows={len(items)} tok={token_count(items)}",
    )
    if hasattr(vocoder, "hop_batch_prefix"):
        vocoder.hop_batch_prefix = ranged(
            vocoder.hop_batch_prefix,
            lambda self, items, caches: (
                f"flow.hop.prefix rows={len(items)} tok={token_count(items)}"
            ),
        )
    vocoder.leftover_batch = ranged(
        vocoder.leftover_batch,
        lambda self, items: f"flow.leftover rows={len(items)} tok={token_count(items)}",
    )
    vocoder.hift_delta = ranged(
        vocoder.hift_delta,
        lambda self, tts_mel, *, hift_mel, speech_offset, finalize: (
            f"hift.delta frames="
            f"{int(tts_mel.shape[2]) + (0 if hift_mel is None else int(hift_mel.shape[2]))}"
            f" final={int(finalize)}"
        ),
    )
    vocoder.mel2wav_batch = ranged(
        vocoder.mel2wav_batch, lambda self, mels: f"hift.batch n={len(mels)}"
    )
    # the batched streaming HiFT step of #2392, rows as tuples on its head and as
    # HiftStepRow after it; a group is (plans, members) there and (windows) here
    if hasattr(vocoder, "hift_step"):
        vocoder.hift_step = ranged(
            vocoder.hift_step,
            lambda self, rows: (
                f"hift.step rows={len(rows)} final="
                f"{int(any(row.is_final if hasattr(row, 'is_final') else row[2] for row in rows))}"
            ),
        )
        vocoder.hift_group = ranged(
            vocoder.hift_group,
            lambda self, *args: f"hift.group rows={len(args[-1])}",
        )
    vocoder.decode_batch = ranged_async(
        vocoder.decode_batch, lambda self, items: f"voc.decode_batch n={len(items)}"
    )


def patch_simple_scheduler(module):
    scheduler = module.StreamingSimpleScheduler
    scheduler.collect_new_request_batch = ranged(
        scheduler.collect_new_request_batch, fixed("voc.collect buffered")
    )
    scheduler.collect_stream_chunk_batch = ranged(
        scheduler.collect_stream_chunk_batch, fixed("voc.collect chunks")
    )
    scheduler.run_non_streaming_batch = ranged(
        scheduler.run_non_streaming_batch,
        lambda self, batch, loop: f"voc.buffered n={len(batch)}",
    )
    scheduler.handle_stream_chunk_batch = ranged(
        scheduler.handle_stream_chunk_batch,
        lambda self, batch: f"voc.chunks n={len(batch)}",
    )
    scheduler.handle_stream_done = ranged(
        scheduler.handle_stream_done,
        lambda self, request_id: f"voc.done rid={request_id}",
    )


def patch_vocoder_base(module):
    base = module.StreamingVocoderBase
    base.finish_stream = ranged(
        base.finish_stream, lambda self, request_id: f"voc.finish rid={request_id}"
    )


def patch_cosy_vocoder(module):
    scheduler = module.FunCosyVoice3StreamingVocoderScheduler
    ingest = scheduler.ingest

    def ingest_marked(self, request_id, state, codes):
        waiting = state.next_decode() == "wait"
        ingest(self, request_id, state, codes)
        if waiting and state.next_decode() != "wait":
            mark(
                f"voc.ready rid={request_id} plan={state.next_decode()} "
                f"tokens={len(state.tokens)} at=ingest"
            )

    scheduler.ingest = ranged(
        ingest_marked,
        lambda self, request_id, state, codes: f"voc.ingest rid={request_id}",
    )
    select = scheduler.select_step_participants
    selections = iter(range(1 << 62))

    def select_marked(self):
        chosen = select(self)
        taken = {request_id for request_id, _ in chosen}
        now = self.clock()
        sel = next(selections)
        for request_id, state in self.stream_state_items():
            plan = state.next_decode()
            if plan == "wait" or self.is_aborted(request_id):
                continue
            if state.first_emit_at is None:
                slack = "u"
            else:
                slack = f"{(state.speech_offset / self.sample_rate - (now - state.first_emit_at)) * 1e3:.0f}"
            mark(
                f"voc.rank rid={request_id} plan={plan} slack={slack} "
                f"chosen={int(request_id in taken)} sel={sel}"
            )
        return chosen

    scheduler.select_step_participants = ranged(select_marked, fixed("voc.select"))
    run_step = scheduler.run_step

    def run_step_marked(self, participants, plan):
        for request_id, _ in participants:
            mark(f"voc.steprid rid={request_id} plan={plan}")
        decoded = run_step(self, participants, plan)
        # a stream whose next hop's tokens arrived during the step is ready at its end
        for request_id, state in participants:
            if state.next_decode() == "causal_window":
                mark(
                    f"voc.ready rid={request_id} plan=causal_window "
                    f"tokens={len(state.tokens)} at=step"
                )
        return decoded

    scheduler.run_step = ranged(
        run_step_marked,
        lambda self, participants, plan: f"voc.step {plan} rows={len(participants)}",
    )
    # the scheduler side of #2406 (cached rows plus the whole history fallback) and
    # of #2392 (one HiFT call per step)
    if hasattr(scheduler, "hop_batch_with_prefix"):
        scheduler.hop_batch_with_prefix = ranged(
            scheduler.hop_batch_with_prefix,
            lambda self, participants, items: (
                f"voc.hop_with_prefix rows={len(participants)}"
            ),
        )
    if hasattr(scheduler, "hift_step"):
        scheduler.hift_step = ranged(scheduler.hift_step, fixed("voc.hift_step"))


PATCHES = {
    "torch": patch_torch,
    "torch.cuda.graphs": patch_graphs,
    "sglang_omni.pipeline.stage.runtime": patch_runtime,
    "sglang_omni.pipeline.coordinator": patch_coordinator,
    "sglang_omni.scheduling.omni_scheduler": patch_scheduler,
    "sglang_omni.model_runner.base": patch_base_runner,
    "sglang_omni.model_runner.model_worker": patch_worker,
    "sglang_omni.scheduling.streaming_simple_scheduler": patch_simple_scheduler,
    "sglang_omni.scheduling.streaming_vocoder": patch_vocoder_base,
    "sglang_omni.models.fun_cosyvoice3.utils": patch_cosy_utils,
    "sglang_omni.models.fun_cosyvoice3.request_builders": patch_cosy_builders,
    "sglang_omni.models.fun_cosyvoice3.model_runner": patch_cosy_runner,
    "sglang_omni.models.fun_cosyvoice3.stages": patch_cosy_stages,
    "sglang_omni.models.fun_cosyvoice3.streaming_vocoder": patch_cosy_vocoder,
}


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES or name in getattr(self, "seen", ()):
            return None
        self.seen = (*getattr(self, "seen", ()), name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                PATCHES[name](module)
            except Exception as exc:
                print(f"cosy_nvtx patch {name} failed: {exc!r}", file=sys.stderr)

        spec.loader.exec_module = exec_and_patch
        return spec


if ENABLED:
    sys.meta_path.insert(0, PatchOnImport())
    print(f"cosy_nvtx on, lines={LINES or 'off'}", file=sys.stderr, flush=True)
