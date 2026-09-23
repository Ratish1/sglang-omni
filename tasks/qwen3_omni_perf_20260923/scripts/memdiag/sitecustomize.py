"""Memory diagnostics for colocated SGLang AR stages, loaded through PYTHONPATH.

Wraps OmniKVCacheConfigurator._profile_available_bytes and SGLModelRunner.init_cuda_graphs
once sglang_omni.model_runner.sglang_model_runner is imported, and prints MEMDIAG lines:
NVML process bytes, torch allocated and reserved, before KV sizing and around graph capture.
MEMDIAG_GC=1 runs gc.collect() and torch.cuda.empty_cache() before the KV sizing reads memory
(the order sglang's own KVCacheConfigurator._profile_available_bytes uses); otherwise the
collection runs after sizing, so the pool is sized exactly as on main and the log shows how
much was collectable.
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


def patch(module):
    configurator = module.OmniKVCacheConfigurator
    profile_original = configurator._profile_available_bytes

    def profile(self, pre_model_load_memory):
        stage = f"{multiprocessing.current_process().name} pid={os.getpid()}"
        log(
            f"{stage} before_kv_sizing {snapshot(self.gpu_id)} gc_count={gc.get_count()}"
        )
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
