"""NVTX ranges and marks over every Qwen3-TTS component, hop and graph capture, for nsys.

Put this directory on PYTHONPATH and set OMNI_PIPE_NVTX=1. No runtime code changes: the
functions are wrapped at import. Ranges are thread local and nested; marks are instants.
pipeline_census.py joins them with the kernels, copies, syncs, OS runtime waits and GIL
ranges of the same nsys report.

OMNI_PIPE_LINES=capture adds a range per dispatched op inside every CUDA graph capture,
named by the op and its call sites, so every replayed node kernel resolves to the line
that captured it (nsys --cuda-graph-trace=node projects capture time ranges through the
node creation events). OMNI_PIPE_LINES=all also labels eager ops on every probed thread;
that pass is for attribution only, its timings are not used. Compiled callables run with
the label mode popped, because Dynamo runs a compiled function's eager ops when a
dispatch mode is active, which would profile a different program.

Names (rid = request id, the first word is the kind the census groups by):

  thread name=N native=T                  mark, once per thread, names the OS thread
  q <stage>.<in|out>.<put|get> rid=R t=T  mark on every stage inbox and outbox hop
  coord.recv rid=R                        mark, coordinator receives a stream message
  pre.*                                   preprocessing worker ranges (payload, prepare,
                                          reference, normalize, resample, speaker, mel,
                                          spk_embed, build_inputs, key_ids)
  ref.encode, ref.sync, ref.drain         reference code batcher thread
  sched.<recv|input|pick|idle|emit>       scheduler loop phases
  sched.batch extend|decode bs=B          one forward, sched.result its result
  sched.build rid=R, sched.adopt          request build on the build pool
  mark sched.admit rid=R, sched.prefill rid=R
  mr.*                                    model runner inside a forward (prep, talker,
                                          sample, collect, predictor, stage_d2h, wait_d2h,
                                          post, feedback, decode_buffers, finalize)
  voc.ingest rid=R                        one codec frame message ingested
  mark voc.put <initial|followup> rid=R   decode work queued
  voc.collect <initial|followup> n=N      batch gather window, marks voc.take rid=R
  voc.initial, voc.followup, voc.group, voc.launch, voc.finish, voc.drain,
  voc.cohort, voc.windows, voc.replay, voc.resolve, voc.commit rid=R,
  voc.commit_followup rid=R
  cap <site>                              one CUDA graph capture
    op <aten op> @<library site> <omni site>   one dispatched op inside a capture
  compiled                                a compiled callable run with labels popped
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
    "qwen_tts/",
    "transformers/",
    "flashinfer/",
    "sgl_kernel/",
    "torchaudio/",
)
PROBE_FILE = os.path.abspath(__file__)
install_lock = threading.Lock()
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
    put, get, get_nowait = q.put, q.get, q.get_nowait

    def put_marked(item, *args, **kwargs):
        mark(f"q {label}.put rid={rid_of(item)} t={getattr(item, 'type', '-')}")
        return put(item, *args, **kwargs)

    def get_marked(*args, **kwargs):
        item = get(*args, **kwargs)
        mark(f"q {label}.get rid={rid_of(item)} t={getattr(item, 'type', '-')}")
        return item

    def get_nowait_marked():
        item = get_nowait()
        mark(f"q {label}.get rid={rid_of(item)} t={getattr(item, 'type', '-')}")
        return item

    q.put, q.get, q.get_nowait = put_marked, get_marked, get_nowait_marked


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


def patch_coordinator(module):
    coordinator = module.Coordinator
    coordinator.handle_stream = marked_async(
        coordinator.handle_stream,
        lambda self, msg: f"coord.recv rid={getattr(msg, 'request_id', None)}",
    )


def patch_builders(module):
    hook = module.Qwen3TTSAdhocReferenceHook
    batcher = module.Qwen3TTSRefCodeBatcher
    module.preprocess_qwen3_tts_payload = ranged(
        module.preprocess_qwen3_tts_payload,
        lambda payload, *args, **kwargs: f"pre.payload rid={payload.request_id}",
    )
    module.prepare_qwen3_tts_request = ranged(
        module.prepare_qwen3_tts_request,
        lambda payload, **kwargs: f"pre.prepare rid={payload.request_id}",
    )
    module.build_embedding_cache_key_ids = ranged(
        module.build_embedding_cache_key_ids,
        lambda embeds: f"pre.key_ids n={int(embeds.shape[0])}",
    )
    module.adopt_prepared_tensors = ranged(
        module.adopt_prepared_tensors, fixed("sched.adopt")
    )
    module.stream_output_builder = ranged(
        module.stream_output_builder, fixed("sched.stream_builder")
    )
    encode_one = hook.encode_one

    def encode_one_ranged(self, item):
        with install_lock:
            if not getattr(self, "pipe_nvtx_installed", False):
                import librosa

                self.wrapper._normalize_audio_inputs = ranged(
                    self.wrapper._normalize_audio_inputs, fixed("pre.normalize")
                )
                self.model.extract_speaker_embedding = ranged(
                    self.model.extract_speaker_embedding, fixed("pre.speaker")
                )
                librosa.resample = ranged(librosa.resample, fixed("pre.resample"))
                self.pipe_nvtx_installed = True
        return encode_one(self, item)

    hook.encode_one = ranged(encode_one_ranged, fixed("pre.reference"))
    batcher.encode_waveform = ranged(
        batcher.encode_waveform,
        lambda self, waveform, sample_rate: f"ref.encode n={len(waveform)}",
    )
    batcher.synchronize_outcomes = ranged(
        batcher.synchronize_outcomes,
        lambda self, outcomes: f"ref.sync b={len(outcomes)}",
    )
    batcher.drain = ranged(batcher.drain, fixed("ref.drain"))


def patch_model(module):
    talker = module.Qwen3TTSTalker
    talker.code_predictor_forward = ranged(
        talker.code_predictor_forward, fixed("mr.predictor")
    )
    talker.prepare_decode_buffers = ranged(
        talker.prepare_decode_buffers, fixed("mr.decode_buffers")
    )
    mixin = module.Qwen3TTSPromptBuilderMixin
    mixin.build_voice_clone_inputs = ranged(
        mixin.build_voice_clone_inputs, fixed("pre.build_inputs")
    )


def patch_speaker(module):
    runner = module.Qwen3TTSSpeakerEncoderCudaGraphRunner
    runner.mel = ranged(runner.mel, fixed("pre.mel"))
    runner.embed = ranged(runner.embed, fixed("pre.spk_embed"))


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
        worker.forward_batch_generation, fixed("mr.talker")
    )


def patch_qwen_runner(module):
    runner = module.Qwen3TTSModelRunner
    for name, label in (
        ("before_prefill", "mr.before_prefill"),
        ("before_decode", "mr.before_decode"),
        ("sample_next_token_ids", "mr.sample"),
        ("apply_codec_suppress_tokens", "mr.suppress"),
        ("collect_codes", "mr.collect"),
        ("post_process_outputs", "mr.post"),
        ("write_feedback_buffers", "mr.feedback"),
    ):
        setattr(runner, name, ranged(getattr(runner, name), fixed(label)))


def patch_runner(module):
    runner = module.Qwen3TTSIncrementalCodecCudaGraphRunner
    runner.decode_slots = ranged(
        runner.decode_slots,
        lambda self, codes, slots: (
            f"voc.replay {self.mode} w={int(codes.shape[2])} rows={int(codes.shape[0])}"
        ),
    )


def patch_vocoder(module):
    scheduler = module.Qwen3TTSStreamingVocoderScheduler
    scheduler.ingest_stream_item = ranged(
        scheduler.ingest_stream_item,
        lambda self, request_id, item: f"voc.ingest rid={request_id}",
    )
    schedule_initial = scheduler.schedule_initial

    def schedule_initial_marked(self, request_id, state):
        mark(f"voc.put initial rid={request_id}")
        return schedule_initial(self, request_id, state)

    scheduler.schedule_initial = schedule_initial_marked
    enqueue_followup = scheduler.enqueue_followup

    def enqueue_followup_marked(self, request_id, state):
        mark(f"voc.put followup rid={request_id}")
        return enqueue_followup(self, request_id, state)

    scheduler.enqueue_followup = enqueue_followup_marked

    def taking(fn, which):
        @functools.wraps(fn)
        def wrapped(self, *args, **kwargs):
            note_thread()
            torch_nvtx().range_push(f"voc.collect {which}")
            try:
                batch = fn(self, *args, **kwargs)
            finally:
                torch_nvtx().range_pop()
            for entry in batch or ():
                mark(f"voc.take {which} rid={entry[0]}")
            return batch

        return wrapped

    scheduler.collect_async_batch = taking(scheduler.collect_async_batch, "initial")
    scheduler.collect_followup_batch = taking(
        scheduler.collect_followup_batch, "followup"
    )
    scheduler.run_initial_batch = ranged(
        scheduler.run_initial_batch,
        lambda self, batch: f"voc.initial rows={len(batch)}",
    )
    scheduler.run_followup_batch = ranged(
        scheduler.run_followup_batch,
        lambda self, batch: f"voc.followup rows={len(batch)}",
    )
    scheduler.decode_incremental_group = ranged(
        scheduler.decode_incremental_group,
        lambda self, group, **kwargs: f"voc.group rows={len(group)}",
    )
    scheduler.launch_incremental_group = ranged(
        scheduler.launch_incremental_group,
        lambda self, group, **kwargs: f"voc.launch rows={len(group)}",
    )
    scheduler.finish_incremental_group = ranged(
        scheduler.finish_incremental_group, fixed("voc.finish")
    )
    scheduler.drain_pending_incremental = ranged(
        scheduler.drain_pending_incremental,
        lambda self, *, keep: f"voc.drain keep={keep}",
    )
    scheduler.decode_incremental_cohort = ranged(
        scheduler.decode_incremental_cohort,
        lambda self, gpu_input, plans, incremental, stream: (
            f"voc.cohort w={int(plans[0].fresh_frames)} rows={len(plans)} "
            f"init={int(stream is self.decode_stream)}"
        ),
    )
    scheduler.decode_incremental_windows = ranged(
        scheduler.decode_incremental_windows,
        lambda self, gpu_input, plans, incremental, runner, split: (
            f"voc.windows n={len(split)}"
        ),
    )
    scheduler.commit_initial = ranged(
        scheduler.commit_initial,
        lambda self, request_id, state, plan, delta: f"voc.commit rid={request_id}",
    )
    scheduler.commit_followup = ranged(
        scheduler.commit_followup,
        lambda self, request_id, *args, **kwargs: (
            f"voc.commit_followup rid={request_id}"
        ),
    )
    handle = module.Qwen3TTSDecodeHandle
    handle.resolve_partial = ranged(handle.resolve_partial, fixed("voc.resolve"))


PATCHES = {
    "torch": patch_torch,
    "torch.cuda.graphs": patch_graphs,
    "sglang_omni.pipeline.stage.runtime": patch_runtime,
    "sglang_omni.pipeline.coordinator": patch_coordinator,
    "sglang_omni.models.qwen3_tts.request_builders": patch_builders,
    "sglang_omni.models.qwen3_tts.sglang_model": patch_model,
    "sglang_omni.models.qwen3_tts.speaker_encoder_cuda_graph": patch_speaker,
    "sglang_omni.scheduling.omni_scheduler": patch_scheduler,
    "sglang_omni.model_runner.base": patch_base_runner,
    "sglang_omni.model_runner.model_worker": patch_worker,
    "sglang_omni.models.qwen3_tts.model_runner": patch_qwen_runner,
    "sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph": patch_runner,
    "sglang_omni.models.qwen3_tts.streaming_vocoder": patch_vocoder,
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
                print(f"pipeline_nvtx patch {name} failed: {exc!r}", file=sys.stderr)

        spec.loader.exec_module = exec_and_patch
        return spec


if ENABLED:
    sys.meta_path.insert(0, PatchOnImport())
    print(f"pipeline_nvtx on, lines={LINES or 'off'}", file=sys.stderr, flush=True)
