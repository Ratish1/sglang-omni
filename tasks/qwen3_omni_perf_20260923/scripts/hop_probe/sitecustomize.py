"""Wall and thread CPU time of every step of a stage-to-stage payload hop, both transports.

Put this directory on PYTHONPATH and set OMNI_HOP_PROBE to an output directory. At import the
probe wraps, in every stage process: Stage.send_to_stage and Stage.on_data_ready (runtime),
StageControlPlane.send_to_stage, stage_io's serialize and deserialize of the direct CUDA IPC payload,
write_payload, pack_tensors, read_payload and read_transfer_buffer, the SHM relay's put_async and
its get operation's wait, and Qwen3-Omni's encoder cache lookup and store and single-payload
encode. One JSON line per call to <dir>/hop_<pid>.jsonl: process, function, request id, start
time, wall ms, and thread CPU ms for synchronous functions (an async function's wall includes
the time other coroutines ran while it waited).
"""

import functools
import importlib.abc
import importlib.util
import inspect
import json
import multiprocessing
import os
import sys
import threading
import time

OUT = os.environ.get("OMNI_HOP_PROBE", "")
write_lock = threading.Lock()

TARGETS = {
    "sglang_omni.pipeline.stage.runtime": [
        ("Stage", "send_to_stage"),
        ("Stage", "on_data_ready"),
    ],
    "sglang_omni.pipeline.control_plane": [("StageControlPlane", "send_to_stage")],
    "sglang_omni.comm.stage_io": [
        (None, "serialize_direct_cuda_ipc_payload"),
        (None, "deserialize_direct_cuda_ipc_payload"),
        (None, "write_payload"),
        (None, "pack_tensors"),
        (None, "pack_tensor_chunks"),
        (None, "read_payload"),
        (None, "read_transfer_buffer"),
    ],
    "sglang_omni.relay.shm": [
        ("ShmRelay", "put_async"),
        ("ShmRelay", "put_chunks_async"),
        ("ShmGetOperation", "wait_for_completion"),
    ],
    "sglang_omni.models.qwen3_omni.stages": [
        (None, "lookup_cached_encoder_output"),
        (None, "store_cached_encoder_output"),
        (None, "run_single_encoder_payload"),
    ],
}


def write(record):
    path = os.path.join(OUT, f"hop_{os.getpid()}.jsonl")
    with write_lock:
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def request_id_of(args, kwargs):
    for value in (kwargs.get("request_id"), kwargs.get("payload"), kwargs.get("msg")):
        if isinstance(value, str):
            return value
        if value is not None and isinstance(getattr(value, "request_id", None), str):
            return value.request_id
    for value in args:
        if isinstance(value, str) and len(value) >= 16:
            return value
        if isinstance(getattr(value, "request_id", None), str):
            return value.request_id
    return None


def record(name, args, kwargs, start_ns, wall_s, cpu_s):
    write(
        {
            "process": multiprocessing.current_process().name,
            "function": name,
            "request_id": request_id_of(args, kwargs),
            "start_ns": start_ns,
            "wall_ms": wall_s * 1e3,
            "cpu_ms": None if cpu_s is None else cpu_s * 1e3,
        }
    )


def wrap(fn, name):
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapped(*args, **kwargs):
            start_ns = time.time_ns()
            wall = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                record(name, args, kwargs, start_ns, time.perf_counter() - wall, None)

        return async_wrapped

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        start_ns = time.time_ns()
        wall = time.perf_counter()
        cpu = time.thread_time()
        try:
            return fn(*args, **kwargs)
        finally:
            record(
                name,
                args,
                kwargs,
                start_ns,
                time.perf_counter() - wall,
                time.thread_time() - cpu,
            )

    return wrapped


def patch(module):
    # A tree has only the functions of its own version (pack_tensors on main,
    # pack_tensor_chunks and put_chunks_async on the one-copy put), so a target
    # the tree lacks is skipped.
    for owner_name, attribute in TARGETS[module.__name__]:
        owner = module if owner_name is None else getattr(module, owner_name)
        if not hasattr(owner, attribute):
            continue
        name = attribute if owner_name is None else f"{owner_name}.{attribute}"
        setattr(owner, attribute, wrap(getattr(owner, attribute), name))


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
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


if OUT:
    os.makedirs(OUT, exist_ok=True)
    sys.meta_path.insert(0, PatchOnImport())
