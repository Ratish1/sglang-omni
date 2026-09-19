"""Per call ledger of the Fun-CosyVoice3 vocoder for a profiling boot.

Wraps, after their modules are imported, the vocoder calls that decide the
graph design, and writes one JSON line per call to
$COSY_CALL_LEDGER_DIR/ledger_<pid>.jsonl:

  step            FunCosyVoice3StreamingVocoderScheduler.run_step: plan, and per
                  participant token_offset, hop_len, tokens so far, started,
                  wait since ready
  hop             CosyVoice3Vocoder.hop_batch
  hop_cached      CosyVoice3Vocoder.hop_batch_cached, where the tree has it
  final           CosyVoice3Vocoder.leftover_batch
  buffered_flow   FunCosyVoice3Flow.inference (the buffered groups)
  graph_run       FlowCudaGraphRunner.run: key, captured, hit
  graph_capture   FlowCudaGraphRunner.capture: shapes, wall, reserved growth
  hift            CosyVoice3Vocoder.hift_delta: history and new frames
  hift_batch      CosyVoice3Vocoder.mel2wav_batch

Shapes are read from host metadata only. host_ms is the wall time of the Python
call; gpu_ms is the time between two CUDA events recorded on the current stream
at entry and exit, resolved by a background thread without synchronizing, so it
is the device time of the work queued between them. No call adds a sync.
"""

from __future__ import annotations

import atexit
import collections
import importlib.abc
import importlib.machinery
import json
import os
import sys
import threading
import time

STAGES = "sglang_omni.models.fun_cosyvoice3.stages"
STREAMING = "sglang_omni.models.fun_cosyvoice3.streaming_vocoder"
FRAMES_PER_TOKEN = 2
LOOKAHEAD = 3
BUCKET = 16

_local = threading.local()
_ledger = None


class _Ledger:
    def __init__(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._file = open(
            os.path.join(directory, f"ledger_{os.getpid()}.jsonl"), "a", buffering=1
        )
        self._pending: collections.deque = collections.deque()
        self._lock = threading.Lock()
        threading.Thread(
            target=self._drain, name="cosy-call-ledger", daemon=True
        ).start()
        atexit.register(self._flush)

    def record(self, entry: dict, start=None, end=None) -> None:
        with self._lock:
            self._pending.append((entry, start, end))

    def _drain(self) -> None:
        while True:
            time.sleep(0.2)
            self._flush(wait=False)

    def _flush(self, wait: bool = True) -> None:
        with self._lock:
            items = list(self._pending)
            self._pending.clear()
        unresolved = []
        for entry, start, end in items:
            if end is not None:
                try:
                    if not wait and not end.query():
                        unresolved.append((entry, start, end))
                        continue
                    entry["gpu_ms"] = start.elapsed_time(end)
                except Exception as exc:
                    entry["gpu_ms_error"] = type(exc).__name__
            self._file.write(json.dumps(entry) + "\n")
        if unresolved:
            with self._lock:
                self._pending.extendleft(reversed(unresolved))


def _flow_rows(items, lookahead: int) -> dict:
    prompt = [int(item.prompt_token.shape[1]) for item in items]
    window = [int(item.token.shape[1]) for item in items]
    frames = [
        FRAMES_PER_TOKEN * (p + max(w - lookahead, 0)) for p, w in zip(prompt, window)
    ]
    return {
        "rows": len(items),
        "prompt_tokens": prompt,
        "window_tokens": window,
        "row_frames": frames,
        "total_frames": sum(frames),
        "widest_frames": max(frames) if frames else 0,
    }


def _timed(kind: str, describe, *, after=None):
    def wrap(function):
        def wrapper(*args, **kwargs):
            import torch

            entry = {"kind": kind, "t_wall": time.time(), "pid": os.getpid()}
            try:
                entry.update(describe(*args, **kwargs))
            except Exception as exc:
                entry["describe_error"] = f"{type(exc).__name__}: {exc}"
            start = end = None
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            began = time.perf_counter()
            result = None
            try:
                result = function(*args, **kwargs)
                return result
            finally:
                entry["host_ms"] = (time.perf_counter() - began) * 1e3
                if end is not None:
                    end.record()
                if after is not None:
                    try:
                        entry.update(after(result, *args, **kwargs))
                    except Exception as exc:
                        entry["after_error"] = f"{type(exc).__name__}: {exc}"
                _ledger.record(entry, start, end)

        wrapper.__wrapped__ = function
        return wrapper

    return wrap


def _step_context() -> dict:
    return {"step": getattr(_local, "step", None)}


def _patch_stages(module) -> None:
    vocoder = module.CosyVoice3Vocoder
    vocoder.hop_batch = _timed(
        "hop",
        lambda self, items: {**_flow_rows(items, LOOKAHEAD), **_step_context()},
    )(vocoder.hop_batch)
    # The hop cache branch only: the frames a row computes are the ones past
    # what its stream already holds.
    if hasattr(vocoder, "hop_batch_cached"):
        vocoder.hop_batch_cached = _timed(
            "hop_cached",
            lambda self, items, streams: {
                **_flow_rows(items, LOOKAHEAD),
                "new_frames": [stream.reserved - stream.frames for stream in streams],
                **_step_context(),
            },
        )(vocoder.hop_batch_cached)
    vocoder.leftover_batch = _timed(
        "final", lambda self, items: {**_flow_rows(items, 0), **_step_context()}
    )(vocoder.leftover_batch)
    vocoder.hift_delta = _timed(
        "hift",
        lambda self, tts_mel, *, hift_mel, speech_offset, finalize: {
            "new_frames": int(tts_mel.shape[2]),
            "history_frames": 0 if hift_mel is None else int(hift_mel.shape[2]),
            "finalize": bool(finalize),
            **_step_context(),
        },
    )(vocoder.hift_delta)
    vocoder.mel2wav_batch = _timed(
        "hift_batch",
        lambda self, mels: {
            "rows": len(mels),
            "row_frames": [int(mel.shape[2]) for mel in mels],
        },
    )(vocoder.mel2wav_batch)
    flow = module.FunCosyVoice3Flow
    flow.inference = _timed(
        "buffered_flow", lambda self, inputs: _flow_rows(inputs, 0)
    )(flow.inference)
    runner = module.FlowCudaGraphRunner

    def describe_run(self, noisy_mel, *inputs):
        batch, frames = int(noisy_mel.shape[0]), int(noisy_mel.shape[2])
        bucket = -(-frames // BUCKET) * BUCKET
        return {
            "batch": batch,
            "frames": frames,
            "bucket": bucket,
            "captured": (batch, bucket) in self.graphs,
        }

    runner.run = _timed(
        "graph_run",
        describe_run,
        after=lambda result, *a, **k: {"hit": result is not None},
    )(runner.run)

    def describe_capture(self, capture_shapes):
        import torch

        return {
            "shapes": [list(shape) for shape in capture_shapes],
            "reserved_before": torch.cuda.memory_reserved(self.device),
        }

    def after_capture(result, self, capture_shapes):
        import torch

        return {"reserved_after": torch.cuda.memory_reserved(self.device)}

    runner.capture = _timed("graph_capture", describe_capture, after=after_capture)(
        runner.capture
    )


def _patch_streaming(module) -> None:
    scheduler = module.FunCosyVoice3StreamingVocoderScheduler
    original = scheduler.run_step

    def run_step(self, participants, plan):
        now = self.clock()
        step = {
            "id": time.monotonic_ns(),
            "plan": str(plan),
            "participants": [
                {
                    "token_offset": int(state.token_offset),
                    "hop_len": int(state.hop_len),
                    "tokens": len(state.tokens),
                    "prompt_tokens": (
                        None
                        if state.prompt_token is None
                        else int(state.prompt_token.shape[-1])
                    ),
                    "started": state.first_emit_at is not None,
                    "wait_ms": (
                        None
                        if state.ready_since is None
                        else (now - state.ready_since) * 1e3
                    ),
                    "history_frames": (
                        None if state.hift_mel is None else int(state.hift_mel.shape[2])
                    ),
                }
                for _, state in participants
            ],
        }
        _local.step = step["id"]
        try:
            return _timed("step", lambda *a, **k: step)(original)(
                self, participants, plan
            )
        finally:
            _local.step = None

    run_step.__wrapped__ = original
    scheduler.run_step = run_step


_PATCHES = {STAGES: _patch_stages, STREAMING: _patch_streaming}


class _PostImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        patch = _PATCHES.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    global _ledger
    if _ledger is not None:
        return
    _ledger = _Ledger(os.environ["COSY_CALL_LEDGER_DIR"])
    for name, patch in _PATCHES.items():
        if name in sys.modules:
            patch(sys.modules[name])
    sys.meta_path.insert(0, _PostImportFinder())
