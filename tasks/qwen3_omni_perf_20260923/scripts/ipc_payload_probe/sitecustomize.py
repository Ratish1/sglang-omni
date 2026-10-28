"""Per-step timing of the direct CUDA IPC payload hop (the plan's item 3).

Put this directory on PYTHONPATH and set OMNI_IPC_PROBE to an output directory. In every
stage process, sglang_omni.comm.stage_io is wrapped at import: serialize_direct_cuda_ipc_payload
and deserialize_direct_cuda_ipc_payload get one record per call with the wall and thread CPU
time of the whole call and of each step inside it (the request walks, extract_cuda_tensors,
every pickle.dumps and pickle.loads, every ipc_pickle, restore_tensors), timed only at the
outermost level so a recursive walk counts once. After the call, outside the timed region,
the record adds the header's pickled size split by field and every tensor's shape, dtype,
device and storage size. One JSON line per call goes to <dir>/ipc_<pid>.jsonl.
"""

import functools
import importlib.abc
import importlib.util
import json
import os
import pickle as std_pickle
import sys
import threading
import time

OUT = os.environ.get("OMNI_IPC_PROBE", "")
local = threading.local()
write_lock = threading.Lock()


def clocks():
    return time.perf_counter(), time.thread_time()


def write(record):
    path = os.path.join(OUT, f"ipc_{os.getpid()}.jsonl")
    with write_lock:
        with open(path, "a") as handle:
            handle.write(json.dumps(record) + "\n")


def timed_step(fn, name, size_of=None):
    """Time fn while a recorded call runs on this thread, outermost call only."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        steps = getattr(local, "steps", None)
        if steps is None or getattr(local, "in_step", False):
            return fn(*args, **kwargs)
        local.in_step = True
        wall_start, cpu_start = clocks()
        try:
            result = fn(*args, **kwargs)
        finally:
            local.in_step = False
        wall_end, cpu_end = clocks()
        step = {
            "step": name,
            "wall_ms": (wall_end - wall_start) * 1e3,
            "cpu_ms": (cpu_end - cpu_start) * 1e3,
        }
        if size_of is not None:
            try:
                step.update(size_of(args, result))
            except Exception as exc:
                step["size_error"] = repr(exc)
        steps.append(step)
        return result

    return wrapped


def pickled_size(value):
    try:
        return len(std_pickle.dumps(value))
    except Exception as exc:
        return f"unpicklable {type(exc).__name__}"


def tensor_facts(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "storage_bytes": int(tensor.untyped_storage().nbytes()),
    }


def field_sizes(value, depth):
    """Pickled size and type of each field, to the given depth, CPU parts only."""
    import torch

    if isinstance(value, torch.Tensor):
        return {"type": "tensor", **tensor_facts(value)}
    if isinstance(value, dict) and depth > 0:
        return {
            "type": "dict",
            "pickled": pickled_size(value),
            "fields": {
                str(key): field_sizes(item, depth - 1) for key, item in value.items()
            },
        }
    if isinstance(value, (list, tuple)) and depth > 0:
        return {
            "type": type(value).__name__,
            "pickled": pickled_size(value),
            "items": [field_sizes(item, depth - 1) for item in value[:8]],
            "length": len(value),
        }
    return {"type": type(value).__name__, "pickled": pickled_size(value)}


def describe_serialize(module, payload, result):
    data_without_tensors, tensors = module.extract_cuda_tensors_unprobed(payload.data)
    request = payload.request
    return {
        "request_id": payload.request_id,
        "header_bytes": len(result["header"]),
        "tensor_entry_bytes": [len(item["tensor_bytes"]) for item in result["tensors"]],
        "request": {
            "pickled": pickled_size(request),
            "inputs": field_sizes(request.inputs, 2),
            "params": field_sizes(request.params, 1),
            "metadata": field_sizes(request.metadata, 2),
        },
        "data_without_tensors": field_sizes(data_without_tensors, 4),
        "cuda_tensors": {
            path: tensor_facts(tensor) for path, tensor in tensors.items()
        },
    }


def describe_deserialize(module, data_ref, result):
    return {
        "request_id": result.request_id,
        "header_bytes": len(data_ref["header"]),
        "tensor_entry_bytes": [
            len(item["tensor_bytes"]) for item in data_ref["tensors"]
        ],
    }


def recorded(module, fn, kind, describe):
    @functools.wraps(fn)
    def wrapped(arg):
        local.steps = []
        wall_start, cpu_start = clocks()
        try:
            result = fn(arg)
        finally:
            wall_end, cpu_end = clocks()
            steps = local.steps
            local.steps = None
        record = {
            "kind": kind,
            "pid": os.getpid(),
            "thread": threading.current_thread().name,
            "time": time.time(),
            "wall_ms": (wall_end - wall_start) * 1e3,
            "cpu_ms": (cpu_end - cpu_start) * 1e3,
            "steps": steps,
            "switch_interval_s": sys.getswitchinterval(),
        }
        try:
            record.update(describe(module, arg, result))
        except Exception as exc:
            record["describe_error"] = repr(exc)
        write(record)
        return result

    return wrapped


class PickleShim:
    """The module's pickle with dumps and loads timed; everything else passes through."""

    def __init__(self, real):
        self.real = real
        self.dumps = timed_step(
            real.dumps, "pickle.dumps", lambda args, result: {"bytes": len(result)}
        )
        self.loads = timed_step(
            real.loads, "pickle.loads", lambda args, result: {"bytes": len(args[0])}
        )

    def __getattr__(self, name):
        return getattr(self.real, name)


def patch_stage_io(module):
    module.extract_cuda_tensors_unprobed = module.extract_cuda_tensors
    for name in (
        "contains_cuda_tensor",
        "contains_cpu_tensor",
        "extract_cuda_tensors",
        "ipc_pickle",
        "restore_tensors",
    ):
        setattr(module, name, timed_step(getattr(module, name), name))
    module.pickle = PickleShim(module.pickle)
    module.serialize_direct_cuda_ipc_payload = recorded(
        module,
        module.serialize_direct_cuda_ipc_payload,
        "serialize",
        describe_serialize,
    )
    module.deserialize_direct_cuda_ipc_payload = recorded(
        module,
        module.deserialize_direct_cuda_ipc_payload,
        "deserialize",
        describe_deserialize,
    )


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != "sglang_omni.comm.stage_io" or getattr(self, "done", False):
            return None
        self.done = True
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                patch_stage_io(module)
            except Exception as exc:
                print(f"ipc_payload_probe patch failed: {exc!r}", file=sys.stderr)

        spec.loader.exec_module = exec_and_patch
        return spec


if OUT:
    os.makedirs(OUT, exist_ok=True)
    sys.meta_path.insert(0, PatchOnImport())
    print(
        f"ipc_payload_probe on, out={OUT}, pid={os.getpid()}",
        file=sys.stderr,
        flush=True,
    )
