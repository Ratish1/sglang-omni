"""What every StageOutputCache keeps alive against what its budget counts.

Put this directory on PYTHONPATH. Once sglang_omni.scheduling.stage_cache is imported,
StageOutputCache.put is wrapped: after each admission it prints one CACHEHOLD line to stderr
with the process name, the cache, its entries, the bytes the budget counts, the bytes of the
distinct tensor storages the entries keep alive (CPU and all devices), and the process's VmRSS.
The walk reaches tensors inside dicts, lists, tuples and dataclasses.
"""

import dataclasses
import importlib.abc
import importlib.util
import multiprocessing
import os
import sys

TARGET = "sglang_omni.scheduling.stage_cache"
MIB = 1 << 20


def held_storages(value, storages, seen):
    import torch

    if id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        storages[(value.device.type, storage.data_ptr())] = storage.nbytes()
    elif isinstance(value, dict):
        for item in value.values():
            held_storages(item, storages, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            held_storages(item, storages, seen)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            held_storages(getattr(value, field.name), storages, seen)


def rss_mib():
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return -1


def patch(module):
    cache_cls = module.StageOutputCache
    put_original = cache_cls.put

    def put(self, key, data):
        put_original(self, key, data)
        with self.lock:
            entries = [entry.data for entry in self.cache.values()]
            counted = self.current_bytes
        storages = {}
        seen = set()
        for value in entries:
            held_storages(value, storages, seen)
        cpu_held = sum(n for (device, _), n in storages.items() if device == "cpu")
        print(
            f"CACHEHOLD {multiprocessing.current_process().name} pid={os.getpid()} "
            f"cache={id(self):x} entries={len(entries)} counted_mib={counted / MIB:.1f} "
            f"held_cpu_mib={cpu_held / MIB:.1f} "
            f"held_all_mib={sum(storages.values()) / MIB:.1f} rss_mib={rss_mib()}",
            file=sys.stderr,
            flush=True,
        )

    cache_cls.put = put


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
