"""Memory diagnostics for colocated SGLang AR stages, loaded through PYTHONPATH.

Wraps OmniKVCacheConfigurator._profile_available_bytes and SGLModelRunner.init_cuda_graphs
once sglang_omni.model_runner.sglang_model_runner is imported, and prints MEMDIAG lines:
NVML process bytes, torch allocated and reserved, before KV sizing and around graph capture.
MEMDIAG_GC=1 runs gc.collect() and torch.cuda.empty_cache() before the KV sizing reads memory
(the order sglang's own KVCacheConfigurator._profile_available_bytes uses); otherwise the
collection runs after sizing, so the pool is sized exactly as on main and the log shows how
much was collectable. MEMDIAG_FIND=1 first lists the CUDA tensors that only reference cycles
keep alive at sizing, with the frames and objects holding them (that collection frees them).
MEMDIAG_RATIO=1 counts the talker's new-token-ratio resets and decays and logs the current
ratio every 5 s; MEMDIAG_NO_RATIO_RESET=1 also turns the talker's resets into no-ops
(diagnostic arm only). MEMDIAG_ADMISSION=1 logs, in the talker process only, each request's
answer tokens and prompt rows at build and its frames at finish, every PrefillAdder
admission attempt with its budget terms, and every decode retraction.
"""

import gc
import importlib.abc
import importlib.util
import multiprocessing
import os
import sys

TARGET = "sglang_omni.model_runner.sglang_model_runner"
GIB = 1 << 30


def log(message):
    print(f"MEMDIAG {message}", file=sys.stderr, flush=True)


def snapshot(gpu_id):
    import torch

    from sglang_omni.utils.gpu_memory import get_process_gpu_memory_bytes

    nvml = get_process_gpu_memory_bytes(gpu_id) or 0
    return (
        f"nvml={nvml / GIB:.3f} alloc={torch.cuda.memory_allocated() / GIB:.3f} "
        f"reserved={torch.cuda.memory_reserved() / GIB:.3f}"
    )


def collect():
    import torch

    collected = gc.collect()
    torch.cuda.empty_cache()
    return collected


def report_cycle_garbage(stage):
    """Collect with DEBUG_SAVEALL and print the CUDA tensors that only cycles kept alive."""
    import collections
    import types

    import torch

    gc.set_debug(gc.DEBUG_SAVEALL)
    gc.collect()
    garbage = list(gc.garbage)
    gc.garbage.clear()
    gc.set_debug(0)
    garbage_ids = {id(obj) for obj in garbage}
    storages = {}
    for obj in garbage:
        if isinstance(obj, torch.Tensor) and obj.is_cuda:
            storage = obj.untyped_storage()
            storages.setdefault(storage.data_ptr(), (storage.nbytes(), obj))
    total = sum(nbytes for nbytes, _ in storages.values())
    log(
        f"{stage} cycle_garbage objects={len(garbage)} cuda_storages={len(storages)} "
        f"bytes_gib={total / GIB:.3f}"
    )
    types_count = collections.Counter(type(obj).__qualname__ for obj in garbage)
    log(f"{stage} cycle_types {types_count.most_common(15)}")
    frames = collections.Counter(
        f"{obj.f_code.co_filename}:{obj.f_code.co_firstlineno}:{obj.f_code.co_name}"
        for obj in garbage
        if isinstance(obj, types.FrameType)
    )
    for frame, count in frames.most_common(20):
        log(f"{stage} cycle_frame x{count} {frame}")
    biggest = sorted(storages.values(), key=lambda item: -item[0])[:10]
    for nbytes, tensor in biggest:
        holders = []
        for referrer in gc.get_referrers(tensor):
            if id(referrer) not in garbage_ids:
                continue
            if isinstance(referrer, types.FrameType):
                code = referrer.f_code
                holders.append(f"frame {code.co_filename}:{code.co_name}")
            elif isinstance(referrer, dict):
                owners = [
                    type(owner).__qualname__
                    for owner in gc.get_referrers(referrer)
                    if id(owner) in garbage_ids
                    and getattr(owner, "__dict__", None) is referrer
                ]
                holders.append(f"dict keys={list(referrer)[:6]} owners={owners}")
            else:
                holders.append(type(referrer).__qualname__)
        log(
            f"{stage} cycle_tensor {tuple(tensor.shape)} {tensor.dtype} "
            f"mib={nbytes / (1 << 20):.1f} held_by={holders[:4]}"
        )
    del garbage, storages, biggest
    gc.collect()


def patch(module):
    configurator = module.OmniKVCacheConfigurator
    profile_original = configurator._profile_available_bytes

    def profile(self, pre_model_load_memory):
        stage = f"{multiprocessing.current_process().name} pid={os.getpid()}"
        log(
            f"{stage} before_kv_sizing {snapshot(self.gpu_id)} gc_count={gc.get_count()}"
        )
        if os.environ.get("MEMDIAG_FIND") == "1":
            report_cycle_garbage(stage)
        if os.environ.get("MEMDIAG_GC") == "1":
            log(f"{stage} collected={collect()} after_gc {snapshot(self.gpu_id)}")
            result = profile_original(self, pre_model_load_memory)
        else:
            result = profile_original(self, pre_model_load_memory)
            log(f"{stage} collected={collect()} after_gc {snapshot(self.gpu_id)}")
        log(f"{stage} kv_bytes_gib={result / GIB:.3f}")
        return result

    configurator._profile_available_bytes = profile

    runner = module.SGLModelRunner
    graphs_original = runner.init_cuda_graphs

    def init_cuda_graphs(self, *args, **kwargs):
        stage = f"{multiprocessing.current_process().name} pid={os.getpid()}"
        log(f"{stage} before_graphs {snapshot(self.gpu_id)}")
        result = graphs_original(self, *args, **kwargs)
        log(f"{stage} after_graphs {snapshot(self.gpu_id)}")
        return result

    runner.init_cuda_graphs = init_cuda_graphs


def patch_ratio(module):
    """Talker only: count new-token-ratio resets and decays and log the current ratio."""
    if multiprocessing.current_process().name != "stage-talker_ar":
        return
    import threading
    import time

    tracker_cls = module.NewTokenRatioTracker
    counts = {"reset": 0, "decay": 0}
    trackers = []
    reset_original = tracker_cls.reset
    decay_original = tracker_cls.decay_step
    skip_reset = os.environ.get("MEMDIAG_NO_RATIO_RESET") == "1"

    def reset(self):
        counts["reset"] += 1
        if self not in trackers:
            trackers.append(self)
        if not skip_reset:
            reset_original(self)

    def decay_step(self):
        counts["decay"] += 1
        if self not in trackers:
            trackers.append(self)
        decay_original(self)

    tracker_cls.reset = reset
    tracker_cls.decay_step = decay_step

    def report():
        while True:
            time.sleep(5)
            current = trackers[0].current if trackers else None
            log(
                f"ratio talker resets={counts['reset']} decays={counts['decay']} "
                f"current={current} skip_reset={skip_reset}"
            )

    threading.Thread(target=report, daemon=True).start()


def is_talker_process():
    return multiprocessing.current_process().name == "stage-talker_ar"


def patch_talker_builder(module):
    """Answer tokens and prompt rows of every talker request at build."""
    if not is_talker_process():
        return
    build_original = module.build_talker_request_data

    def build_talker_request_data(payload, **kwargs):
        req_data = build_original(payload, **kwargs)
        log(
            f"admission build rid={payload.request_id} "
            f"answer_tokens={len(payload.prefetched_chunks)} "
            f"prompt_rows={len(req_data.req.origin_input_ids)} "
            f"thinker_done={bool(payload.prefetched_stream_done)}"
        )
        return req_data

    module.build_talker_request_data = build_talker_request_data


def patch_talker_runner(module):
    """Frames and finish reason of every talker request at finish."""
    if not is_talker_process():
        return
    runner = module.QwenTalkerModelRunner
    finished_original = runner.on_request_finished

    def on_request_finished(self, request_id, req_data):
        log(
            f"admission finish rid={request_id} frames={len(req_data.req.output_ids)} "
            f"reason={req_data.finish_reason}"
        )
        return finished_original(self, request_id, req_data)

    runner.on_request_finished = on_request_finished


def patch_prefill_adder(module):
    """Every talker admission attempt: free tokens, reservation terms, verdict."""
    if not is_talker_process():
        return
    adder_cls = module.PrefillAdder
    add_original = adder_cls.add_one_req

    def add_one_req(self, req, has_chunked_req, truncation_align_size):
        max_new = min(
            max(req.sampling_params.max_new_tokens - len(req.output_ids), 0),
            module.CLIP_MAX_NEW_TOKENS,
        )
        extend = len(req.full_untruncated_fill_ids) - len(req.prefix_indices)
        running = len(self.running_batch.reqs) if self.running_batch else 0
        before = self.rem_total_tokens
        offset = self.rem_total_token_offset
        result = add_original(self, req, has_chunked_req, truncation_align_size)
        log(
            f"admission try rid={req.rid} running={running} "
            f"admitted_this_pass={len(self.can_run_list)} ratio={self.new_token_ratio:.4f} "
            f"rem_total={before:.0f} offset={offset:.0f} extend={extend} "
            f"max_new={max_new} clip={module.CLIP_MAX_NEW_TOKENS} result={result.name}"
        )
        return result

    adder_cls.add_one_req = add_one_req


def patch_retract(module):
    """Every talker decode retraction."""
    if not is_talker_process():
        return
    batch_cls = module.ScheduleBatch
    retract_original = batch_cls.retract_decode

    def retract_decode(self):
        running = len(self.reqs)
        retracted, ratio, aborted = retract_original(self)
        log(
            f"admission retract running={running} retracted={len(retracted)} "
            f"aborted={len(aborted)} new_ratio={ratio:.4f}"
        )
        return retracted, ratio, aborted

    batch_cls.retract_decode = retract_decode


TARGETS = {TARGET: patch}
if os.environ.get("MEMDIAG_ADMISSION") == "1":
    TARGETS["sglang_omni.models.qwen3_omni.request_builders"] = patch_talker_builder
    TARGETS["sglang_omni.models.qwen3_omni.talker_model_runner"] = patch_talker_runner
    TARGETS["sglang.srt.managers.schedule_policy"] = patch_prefill_adder
    TARGETS["sglang.srt.managers.schedule_batch"] = patch_retract
if os.environ.get("MEMDIAG_RATIO") == "1":
    TARGETS["sglang.srt.managers.scheduler_components.new_token_ratio_tracker"] = (
        patch_ratio
    )


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        exec_original = spec.loader.exec_module

        def exec_module(module):
            exec_original(module)
            TARGETS[name](module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, PatchOnImport())
