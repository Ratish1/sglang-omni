"""Per-key timing of the Qwen3-TTS vocoder graph capture, without touching runtime code.

Put this directory on PYTHONPATH and set OMNI_CAPTURE_TIMING to an output prefix. Every
server process then wraps the capture runner when it is first imported and appends one
JSON line per captured key to <prefix>.<pid>.jsonl: precompile, each warmup decode (stream
synchronized, so the first one carries cuDNN plan selection and lazy init), the whole key,
gc.collect and empty_cache. State is per thread, so runners capturing on different
threads do not mix their rows.
"""

import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time

TARGET = "sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph"
PREFIX = os.environ.get("OMNI_CAPTURE_TIMING")


def patch(module):
    import gc

    import torch

    runner = module.Qwen3TTSIncrementalCodecCudaGraphRunner
    decoder = module.Qwen3TTSIncrementalDecoder
    # note(ratish): held open for the life of the server process
    out = open(f"{PREFIX}.{os.getpid()}.jsonl", "a", buffering=1)  # noqa: SIM115
    write_lock = threading.Lock()
    local = threading.local()

    def write(record):
        with write_lock:
            out.write(json.dumps(record) + "\n")

    decode, precompile = decoder.decode, decoder.precompile
    warmup, capture_graph, capture = (
        runner._warmup_capture_shape,
        runner._capture_graph,
        runner.capture,
    )

    def stream_timed(name, fn):
        def wrapped(self, *args, **kwargs):
            if not getattr(local, "warmup", False):
                return fn(self, *args, **kwargs)
            stream = torch.cuda.current_stream()
            stream.synchronize()
            began = time.perf_counter()
            result = fn(self, *args, **kwargs)
            stream.synchronize()
            local.row.setdefault(name, []).append(time.perf_counter() - began)
            return result

        return wrapped

    def host_timed(name, fn):
        def wrapped(*args, **kwargs):
            if not getattr(local, "capture", False):
                return fn(*args, **kwargs)
            began = time.perf_counter()
            result = fn(*args, **kwargs)
            local.row[name] = local.row.get(name, 0.0) + time.perf_counter() - began
            if name == "empty_cache_s":
                write(local.row)
            return result

        return wrapped

    def warmup_timed(self, *args, **kwargs):
        local.warmup = True
        began = time.perf_counter()
        try:
            return warmup(self, *args, **kwargs)
        finally:
            local.warmup = False
            local.row["warmup_total_s"] = time.perf_counter() - began

    def capture_graph_timed(self, key, **kwargs):
        local.row = {
            "mode": self._mode,
            "frames": key.fresh_frames,
            "batch": key.batch_bucket,
            "compiled": key.fresh_frames in self._compile_fresh_frames,
            "at": time.time(),
        }
        began = time.perf_counter()
        try:
            return capture_graph(self, key, **kwargs)
        finally:
            local.row["key_total_s"] = time.perf_counter() - began

    def capture_timed(self):
        local.capture = True
        local.row = {}
        began = time.perf_counter()
        try:
            return capture(self)
        finally:
            local.capture = False
            write(
                {
                    "mode": self._mode,
                    "runner_total_s": time.perf_counter() - began,
                    "enabled": self._enabled,
                    "thread": threading.current_thread().name,
                }
            )

    decoder.decode = stream_timed("warmup_s", decode)
    decoder.precompile = stream_timed("precompile_s", precompile)
    runner._warmup_capture_shape = warmup_timed
    runner._capture_graph = capture_graph_timed
    runner.capture = capture_timed
    gc.collect = host_timed("gc_collect_s", gc.collect)
    torch.cuda.empty_cache = host_timed("empty_cache_s", torch.cuda.empty_cache)


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if PREFIX:
    sys.meta_path.insert(0, PatchOnImport())
