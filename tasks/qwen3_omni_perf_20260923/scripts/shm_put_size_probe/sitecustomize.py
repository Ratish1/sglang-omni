"""Size and time of every SHM relay put, per stage process.

Put this directory on PYTHONPATH and set OMNI_SHM_PUT_PROBE to an output directory. At import the
probe wraps sglang_omni.relay.shm.shm_create_from_tensor in every stage process and writes one
JSON line per put to <dir>/put_<pid>.jsonl: process, payload bytes, start time, wall and thread
CPU microseconds.
"""

import functools
import importlib.abc
import importlib.util
import json
import multiprocessing
import os
import sys
import threading
import time

OUT = os.environ.get("OMNI_SHM_PUT_PROBE", "")
TARGET = "sglang_omni.relay.shm"
write_lock = threading.Lock()


def wrap(fn):
    @functools.wraps(fn)
    def wrapped(tensor):
        start_ns = time.time_ns()
        wall = time.perf_counter()
        cpu = time.thread_time()
        try:
            return fn(tensor)
        finally:
            line = json.dumps(
                {
                    "process": multiprocessing.current_process().name,
                    "bytes": int(tensor.numel() * tensor.element_size()),
                    "start_ns": start_ns,
                    "wall_us": (time.perf_counter() - wall) * 1e6,
                    "cpu_us": (time.thread_time() - cpu) * 1e6,
                }
            )
            with write_lock:
                with open(os.path.join(OUT, f"put_{os.getpid()}.jsonl"), "a") as handle:
                    handle.write(line + "\n")

    return wrapped


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
            module.shm_create_from_tensor = wrap(module.shm_create_from_tensor)

        spec.loader.exec_module = exec_module
        return spec


if OUT:
    os.makedirs(OUT, exist_ok=True)
    sys.meta_path.insert(0, PatchOnImport())
