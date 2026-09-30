"""NVTX ranges and marks over every Qwen3-Omni stage, handoff and graph capture, for nsys.

Put this directory on PYTHONPATH and set OMNI_PIPE_NVTX=1. No runtime code changes: the
functions are wrapped at import in every stage process, so main, an arm or a contributor
branch is profiled as it is. Ranges are thread local and nested, pushed and popped on the
calling thread, and never span an await; async functions get an entry and an exit mark
instead. omni_census.py joins them with the kernels, copies, syncs, OS runtime waits,
GIL ranges and GPU context switches of the same nsys report.

OMNI_PIPE_LINES=capture adds a range per dispatched op inside every CUDA graph capture,
named by the op and its call sites, so every replayed node kernel resolves to the line
that captured it (nsys --cuda-graph-trace=node projects capture time ranges through the
node creation events). Compiled callables run with the label mode popped, because Dynamo
runs a compiled function's eager ops when a dispatch mode is active, which would profile
a different program.

Names (rid = request id; the first word is the kind the census groups by):

  thread name=N native=T                     mark, once per thread
  proc stage=S                               mark, once per stage, on its IO thread
  q <stage>.<in|out>.<put|get> rid=R t=T     mark on every scheduler inbox and outbox hop
  io.<recv|chunk|send|stream|coord> rid=R A->B, io.result rid=R S, each with a .end
                                             marks at entry and exit of the stage IO
                                             coroutines (receive a payload, receive a
                                             stream chunk, send a payload, send a stream
                                             item, send to the coordinator, route a result)
  io.ser <fn>, io.deser <fn>                 relay serialization on the calling thread
  coord.submit rid=R                         mark, a request enters the coordinator
  coord.recv rid=R from=S n=N                mark, a stream chunk reaches the coordinator
                                             (n = audio samples, - for text)
  coord.done rid=R from=S                    mark, a stage reports completion
  pre.run rid=R                              one preprocessing job on its pool thread
  enc.single rid=R, enc.batch n=N            encoder compute (image, audio) and other
                                             simple stages, on their scheduler thread
  sched.batch <extend|mixed|decode> bs=B toks=T, sched.launch ..., sched.resolve
                                             one engine forward (sync, or async launch and
                                             resolve), thinker and talker alike
  sched.result, sched.recv, sched.input n=N, sched.pick, sched.idle, sched.emit
  sched.build rid=R                          request build
  mark sched.admit rid=R, sched.prefill rid=R
  mr.*                                       model runner phases inside a forward
  tk.build rid=R chunks=C done=D, tk.text_chunk
                                             talker prompt build from the thinker stream
  c2w.ingest rid=R, c2w.decode rid=R final=F, c2w.emit rid=R, c2w.step rows=N,
  c2w.sub rows=N, c2w.replay rows=N frames=F code2wav
  cap <site>                                 one CUDA graph capture
    op <aten op> @<library site> <omni site> one dispatched op inside a capture
  compiled                                   a compiled callable run with labels popped
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
    """Wrap a synchronous fn in an NVTX range named by name(*args, **kwargs)."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        note_thread()
        torch_nvtx().range_push(label_of(name, fn, args, kwargs))
        try:
            return fn(*args, **kwargs)
        finally:
            torch_nvtx().range_pop()

    return wrapped


def bracketed_async(fn, name):
    """Wrap a coroutine function in an entry mark and an exit mark (label + '.end')."""

    @functools.wraps(fn)
    async def wrapped(*args, **kwargs):
        label = label_of(name, fn, args, kwargs)
        mark(label)
        try:
            return await fn(*args, **kwargs)
        finally:
            words = label.split(" ", 1)
            mark(f"{words[0]}.end {words[1] if len(words) > 1 else ''}".rstrip())

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
            torch_nvtx().range_push(f"op {func.__name__} {call_sites()}")
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
    compile_function = module.compile

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
            return popping_labels(compile_function(*args, **kwargs))
        decorator = compile_function(*args, **kwargs)
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


def patch_stage(module):
    stage = module.Stage
    start = stage.start

    @functools.wraps(start)
    async def start_marked(self):
        mark(f"proc stage={self.name}")
        scheduler = self.scheduler
        if scheduler is not None:
            for attr, short in (("inbox", "in"), ("outbox", "out")):
                q = getattr(scheduler, attr, None)
                if q is not None and hasattr(q, "get_nowait"):
                    instrument_queue(q, f"{self.name}.{short}")
        return await start(self)

    stage.start = start_marked
    stage.on_data_ready = bracketed_async(
        stage.on_data_ready,
        lambda self, msg, *args, **kwargs: (
            f"io.recv rid={msg.request_id} {msg.from_stage}->{self.name}"
        ),
    )
    stage.on_stream_chunk = bracketed_async(
        stage.on_stream_chunk,
        lambda self, msg, *args, **kwargs: (
            f"io.chunk rid={msg.request_id} {msg.from_stage}->{self.name}"
        ),
    )
    stage.send_to_stage = bracketed_async(
        stage.send_to_stage,
        lambda self, request_id, target, *args, **kwargs: (
            f"io.send rid={request_id} {self.name}->{target}"
        ),
    )
    stage.send_stream_to_target = bracketed_async(
        stage.send_stream_to_target,
        lambda self, request_id, data, target, *args, **kwargs: (
            f"io.stream rid={request_id} {self.name}->{target}"
        ),
    )
    stage.send_stream_to_coordinator = bracketed_async(
        stage.send_stream_to_coordinator,
        lambda self, request_id, *args, **kwargs: (
            f"io.coord rid={request_id} {self.name}->coordinator"
        ),
    )
    stage.route_result = bracketed_async(
        stage.route_result,
        lambda self, request_id, *args, **kwargs: (
            f"io.result rid={request_id} {self.name}"
        ),
    )


def patch_stage_io(module):
    for name in (
        "serialize_direct_cuda_ipc_payload",
        "serialize_direct_cuda_ipc_stream_chunk",
        "serialize_inline_stream_chunk",
        "pack_tensors",
        "ipc_pickle",
    ):
        setattr(module, name, ranged(getattr(module, name), fixed(f"io.ser {name}")))
    for name in (
        "deserialize_direct_cuda_ipc_payload",
        "deserialize_direct_cuda_ipc_stream_chunk",
        "deserialize_inline_stream_chunk",
    ):
        setattr(module, name, ranged(getattr(module, name), fixed(f"io.deser {name}")))


def coordinator_recv_label(self, msg):
    """The request, the sending stage and the chunk's audio sample count."""
    chunk = getattr(msg, "chunk", None)
    shape = chunk.get("audio_waveform_shape") if isinstance(chunk, dict) else None
    samples = shape[-1] if shape else "-"
    return f"coord.recv rid={msg.request_id} from={msg.from_stage} n={samples}"


def patch_coordinator(module):
    coordinator = module.Coordinator
    start = coordinator.start

    @functools.wraps(start)
    async def start_marked(self):
        mark("proc stage=coordinator")
        return await start(self)

    coordinator.start = start_marked
    coordinator.handle_stream = marked_async(
        coordinator.handle_stream, coordinator_recv_label
    )
    coordinator.handle_completion = marked_async(
        coordinator.handle_completion,
        lambda self, msg: f"coord.done rid={msg.request_id} from={msg.from_stage}",
    )
    coordinator.submit_request = marked_async(
        coordinator.submit_request,
        lambda self, request_id, *args, **kwargs: f"coord.submit rid={request_id}",
    )


def patch_threaded_simple(module):
    scheduler = module.ThreadedSimpleScheduler
    scheduler.run_one = ranged(
        scheduler.run_one,
        lambda self, payload: f"pre.run rid={getattr(payload, 'request_id', None)}",
    )


def patch_simple(module):
    scheduler = module.SimpleScheduler
    scheduler.run_single = ranged(
        scheduler.run_single,
        lambda self, msg, loop: f"enc.single rid={msg.request_id}",
    )
    scheduler.run_batch = ranged(
        scheduler.run_batch,
        lambda self, batch, loop: f"enc.batch n={len(batch)}",
    )


def mode_of(batch):
    forward_mode = batch.forward_mode
    if forward_mode.is_mixed():
        return "mixed"
    elif forward_mode.is_extend():
        return "extend"
    else:
        return "decode"


def batch_label(prefix, batch):
    mode = mode_of(batch)
    label = f"{prefix} {mode} bs={len(batch.reqs)}"
    if mode != "decode":
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
    run_launch = scheduler.run_batch_launch

    def run_launch_marked(self, batch):
        if batch.forward_mode.is_extend():
            for req in batch.reqs:
                mark(f"sched.prefill rid={req.rid}")
        return run_launch(self, batch)

    scheduler.run_batch_launch = ranged(
        run_launch_marked, lambda self, batch: batch_label("sched.launch", batch)
    )
    scheduler.run_batch_resolve = ranged(
        scheduler.run_batch_resolve,
        lambda self, batch, *args, **kwargs: batch_label("sched.resolve", batch),
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

    @functools.wraps(enqueue)
    def enqueue_marked(self, payload, *args, **kwargs):
        mark(f"sched.admit rid={payload.request_id}")
        return enqueue(self, payload, *args, **kwargs)

    scheduler.enqueue_built_request = enqueue_marked


def patch_base_runner(module):
    runner = module.ModelRunner
    for name, label in (
        ("execute", "mr.execute"),
        ("execute_launch", "mr.launch"),
        ("execute_resolve", "mr.resolve"),
        ("build_forward_batch", "mr.build_fb"),
        ("prepare_and_forward", "mr.prepare_forward"),
        ("finalize", "mr.finalize"),
        ("resolve_host_token_ids", "mr.wait_d2h"),
        ("stage_token_ids", "mr.stage_d2h"),
        ("publish_next_tokens", "mr.publish"),
        ("sample_next_token_ids", "mr.sample"),
    ):
        setattr(runner, name, ranged(getattr(runner, name), fixed(label)))


def patch_worker(module):
    worker = module.ModelWorker
    worker.forward_batch_generation = ranged(
        worker.forward_batch_generation, fixed("mr.forward")
    )


def patch_thinker_runner(module):
    runner = module.Qwen3OmniThinkerModelRunner
    runner.before_prefill = ranged(runner.before_prefill, fixed("mr.before_prefill"))
    runner.custom_prefill_forward = ranged(
        runner.custom_prefill_forward, fixed("mr.prefill_forward")
    )


def patch_talker_runner(module):
    runner = module.QwenTalkerModelRunner
    for name, label in (
        ("before_prefill", "mr.before_prefill"),
        ("before_decode", "mr.before_decode"),
        ("post_prefill", "mr.post_prefill"),
        ("post_decode", "mr.post_decode"),
        ("emit_code_chunks_and_feedback", "mr.emit_codes"),
        ("write_feedback_buffers", "mr.feedback"),
        ("compose_prefill_embeds", "mr.prefill_embeds"),
    ):
        setattr(runner, name, ranged(getattr(runner, name), fixed(label)))


def patch_talker_model(module):
    talker = module.Qwen3OmniTalker
    talker.code_predictor_forward = ranged(
        talker.code_predictor_forward, fixed("mr.predictor")
    )


def patch_talker_prefill(module):
    builder = module.TalkerPrefillBuilder
    builder.build_prompt_prefill = ranged(
        builder.build_prompt_prefill,
        lambda self, payload, thinker_chunks, *, thinker_done: (
            f"tk.build rid={payload.request_id} chunks={len(thinker_chunks)} "
            f"done={int(thinker_done)}"
        ),
    )
    builder.append_text_chunk = ranged(
        builder.append_text_chunk, fixed("tk.text_chunk")
    )


def patch_code2wav(module):
    scheduler = module.Code2WavScheduler
    scheduler.ingest = ranged(
        scheduler.ingest,
        lambda self, request_id, state, codes: f"c2w.ingest rid={request_id}",
    )
    scheduler.decode_delta = ranged(
        scheduler.decode_delta,
        lambda self, request_id, state, *, is_final: (
            f"c2w.decode rid={request_id} final={int(is_final)}"
        ),
    )
    scheduler.decode_and_emit = ranged(
        scheduler.decode_and_emit,
        lambda self, request_id, state: f"c2w.emit rid={request_id}",
    )
    scheduler.run_step = ranged(
        scheduler.run_step,
        lambda self, participants, plan: f"c2w.step rows={len(participants)}",
    )
    scheduler.run_sub_batch = ranged(
        scheduler.run_sub_batch,
        lambda self, group, decoded: f"c2w.sub rows={len(group)}",
    )


def patch_code2wav_graph(module):
    runner = module.Code2WavCudaGraphRunner
    runner.run = ranged(
        runner.run,
        lambda self, codes, **kwargs: (
            f"c2w.replay rows={int(codes.shape[0])} frames={int(codes.shape[-1])}"
        ),
    )


PATCHES = {
    "torch": patch_torch,
    "torch.cuda.graphs": patch_graphs,
    "sglang_omni.pipeline.stage.runtime": patch_stage,
    "sglang_omni.comm.stage_io": patch_stage_io,
    "sglang_omni.pipeline.coordinator": patch_coordinator,
    "sglang_omni.scheduling.threaded_simple_scheduler": patch_threaded_simple,
    "sglang_omni.scheduling.simple_scheduler": patch_simple,
    "sglang_omni.scheduling.omni_scheduler": patch_scheduler,
    "sglang_omni.model_runner.base": patch_base_runner,
    "sglang_omni.model_runner.model_worker": patch_worker,
    "sglang_omni.models.qwen3_omni.thinker_model_runner": patch_thinker_runner,
    "sglang_omni.models.qwen3_omni.talker_model_runner": patch_talker_runner,
    "sglang_omni.models.qwen3_omni.components.talker": patch_talker_model,
    "sglang_omni.models.qwen3_omni.components.talker_prefill": patch_talker_prefill,
    "sglang_omni.models.qwen3_omni.components.code2wav_scheduler": patch_code2wav,
    "sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph": patch_code2wav_graph,
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
                print(
                    f"omni_pipeline_nvtx patch {name} failed: {exc!r}", file=sys.stderr
                )

        spec.loader.exec_module = exec_and_patch
        return spec


if ENABLED:
    sys.meta_path.insert(0, PatchOnImport())
    print(
        f"omni_pipeline_nvtx on, lines={LINES or 'off'}, pid={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
