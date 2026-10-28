"""Size and time of every SHM relay put and of the payload packing before it, per stage process.

Put this directory on PYTHONPATH and set OMNI_SHM_PUT_PROBE to an output directory. At import the
probe wraps, in every stage process and where the tree has them: sglang_omni.relay.shm's
shm_create_from_tensor and shm_create_from_chunks (the put's block creation and copy), and
sglang_omni.comm.stage_io's pack_tensors and pack_tensor_chunks (main packs with a torch.cat
there; the one-copy put concatenates inside shm_create_from_chunks instead). One JSON line per
call to <dir>/put_<pid>.jsonl: process, function, payload bytes, start time, wall and thread CPU
microseconds.
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
TARGETS = {
    "sglang_omni.relay.shm": ("shm_create_from_tensor", "shm_create_from_chunks"),
    "sglang_omni.comm.stage_io": ("pack_tensors", "pack_tensor_chunks"),
}
write_lock = threading.Lock()


def payload_bytes(value):
    if hasattr(value, "numel"):
        return int(value.numel() * value.element_size())
    if isinstance(value, tuple):
        return payload_bytes(value[0])
    return sum(payload_bytes(chunk) for chunk in value)


def wrap(fn, packs):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        start_ns = time.time_ns()
        wall = time.perf_counter()
        cpu = time.thread_time()
        result = fn(*args, **kwargs)
        line = json.dumps(
            {
                "process": multiprocessing.current_process().name,
                "function": fn.__name__,
                "bytes": payload_bytes(result if packs else args[0]),
                "start_ns": start_ns,
                "wall_us": (time.perf_counter() - wall) * 1e6,
                "cpu_us": (time.thread_time() - cpu) * 1e6,
            }
        )
        with write_lock:
            with open(os.path.join(OUT, f"put_{os.getpid()}.jsonl"), "a") as handle:
                handle.write(line + "\n")
        return result

    return wrapped


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
            for function_name in TARGETS[name]:
                if hasattr(module, function_name):
                    setattr(
                        module,
                        function_name,
                        wrap(
                            getattr(module, function_name),
                            function_name.startswith("pack"),
                        ),
                    )

        spec.loader.exec_module = exec_module
        return spec


if OUT:
    os.makedirs(OUT, exist_ok=True)
    sys.meta_path.insert(0, PatchOnImport())
