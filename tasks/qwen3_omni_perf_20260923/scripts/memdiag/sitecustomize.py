"""Memory diagnostics for colocated SGLang AR stages, loaded through PYTHONPATH.

Wraps OmniKVCacheConfigurator._profile_available_bytes and SGLModelRunner.init_cuda_graphs
once sglang_omni.model_runner.sglang_model_runner is imported, and prints MEMDIAG lines:
NVML process bytes, torch allocated and reserved, before KV sizing and around graph capture.
MEMDIAG_GC=1 runs gc.collect() and torch.cuda.empty_cache() before the KV sizing reads memory
(the order sglang's own KVCacheConfigurator._profile_available_bytes uses); otherwise the
collection runs after sizing, so the pool is sized exactly as on main and the log shows how
much was collectable. MEMDIAG_FIND=1 first lists the CUDA tensors that only reference cycles
keep alive at sizing, with the frames and objects holding them (that collection frees them).
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


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        exec_original = spec.loader.exec_module

        def exec_module(module):
            exec_original(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, PatchOnImport())
