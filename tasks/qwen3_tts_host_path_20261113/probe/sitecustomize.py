"""Boot probe for the Qwen3-TTS host path checks, loaded through PYTHONPATH by every server process.

QWEN3_TTS_FIRST_EVENTS=1 keeps one request event per request, stage and name: main sends a stream chunk
per request per step where the step table sends one message, so recording every chunk would slow main more.
QWEN3_TTS_PROBE_DIR samples the code table's free rows and holders and counts retraction replays.
QWEN3_TTS_TIME_PREPROCESSING=1 records each preprocessing step's wall and thread CPU time per call.
QWEN3_TTS_PREPROCESSING_PRIORITY=greatest moves in-process preprocessing to the device's greatest
stream priority, the talker's.
"""

import importlib.abc
import json
import os
import sys
import threading
import time
from collections import defaultdict

FIRST_EVENTS = os.environ.get("QWEN3_TTS_FIRST_EVENTS") == "1"
PROBE_DIRECTORY = os.environ.get("QWEN3_TTS_PROBE_DIR")
TIME_PREPROCESSING = os.environ.get("QWEN3_TTS_TIME_PREPROCESSING") == "1"
PREPROCESSING_PRIORITY = os.environ.get("QWEN3_TTS_PREPROCESSING_PRIORITY")
SAMPLE_INTERVAL_S = 1.0
TIMING_DUMP_INTERVAL_S = 5.0
GIL_GAUGE_SLEEP_S = 0.001
TIMED_REQUEST_BUILDER_FUNCTIONS = (
    "preprocess_qwen3_tts_payload",
    "prepare_qwen3_tts_request",
    "prepare_qwen3_tts_base_request",
)

tables = []
replays = {"calls": 0, "frames": 0}
sampler_started = threading.Event()
timings = defaultdict(list)
timing_dump_started = threading.Event()


def keep_first_events(module):
    from sglang_omni.profiler.event_recorder_types import get_active_stage

    recorder_type = module.RequestEventRecorder
    emit = recorder_type.emit
    seen = set()

    def emit_first(self, *, request_id, stage, event_name, **kwargs):
        key = (request_id, stage or get_active_stage(), event_name)
        if key in seen:
            return
        else:
            seen.add(key)
        emit(self, request_id=request_id, stage=stage, event_name=event_name, **kwargs)

    recorder_type.emit = emit_first


def sample_tables():
    os.makedirs(PROBE_DIRECTORY, exist_ok=True)
    path = os.path.join(PROBE_DIRECTORY, f"rows_{os.getpid()}.jsonl")
    with open(path, "a") as sink:
        while True:
            for index, table in enumerate(list(tables)):
                record = {
                    "time": time.time(),
                    "table": index,
                    "rows": len(table.holders),
                    "free": len(table.free_rows),
                    "held": sum(table.holders),
                    "holding_rows": sum(1 for count in table.holders if count),
                    **replays,
                }
                sink.write(json.dumps(record) + "\n")
            sink.flush()
            time.sleep(SAMPLE_INTERVAL_S)


def register_tables(module):
    table_type = module.Qwen3TTSCodeTable
    init = table_type.__init__

    def init_registered(self, *args, **kwargs):
        init(self, *args, **kwargs)
        tables.append(self)
        if not sampler_started.is_set():
            sampler_started.set()
            threading.Thread(
                target=sample_tables, name="row-probe", daemon=True
            ).start()
        else:
            pass

    table_type.__init__ = init_registered


def count_replays(module):
    runner_type = module.Qwen3TTSModelRunner
    if not hasattr(runner_type, "replayed_decode_inputs"):
        return
    else:
        pass
    replayed_decode_inputs = runner_type.replayed_decode_inputs

    def replayed_counted(self, data, first_frame, stop_frame):
        replays["calls"] += 1
        replays["frames"] += stop_frame - first_frame
        return replayed_decode_inputs(self, data, first_frame, stop_frame)

    runner_type.replayed_decode_inputs = replayed_counted


def dump_timings():
    os.makedirs(PROBE_DIRECTORY, exist_ok=True)
    path = os.path.join(PROBE_DIRECTORY, f"timings_{os.getpid()}.json")
    while True:
        time.sleep(TIMING_DUMP_INTERVAL_S)
        snapshot = {name: list(calls) for name, calls in list(timings.items())}
        with open(path + ".tmp", "w") as sink:
            json.dump(snapshot, sink)
        os.replace(path + ".tmp", path)


def timed(name, function):
    """function, recording (start time, wall ns, thread CPU ns) per call under name."""

    def run(*args, **kwargs):
        if not timing_dump_started.is_set():
            timing_dump_started.set()
            threading.Thread(
                target=dump_timings, name="timing-probe", daemon=True
            ).start()
        else:
            pass
        started = time.time()
        wall = time.perf_counter_ns()
        cpu = time.thread_time_ns()
        try:
            return function(*args, **kwargs)
        finally:
            timings[name].append(
                (
                    started,
                    time.perf_counter_ns() - wall,
                    time.thread_time_ns() - cpu,
                )
            )

    return run


def time_cache_keys(module):
    keys_type = module.PromptRowCacheKeys
    keys_type.key_ids = timed("key_ids", keys_type.key_ids)


def time_reference_service(module):
    service_type = module.ReferenceEncodeService
    service_type.get_or_encode = timed("get_or_encode", service_type.get_or_encode)


def time_prompt_builder(module):
    builder_type = module.Qwen3TTSPromptBuilderMixin
    builder_type.build_voice_clone_inputs = timed(
        "build_voice_clone_inputs", builder_type.build_voice_clone_inputs
    )


def gauge_gil():
    """How long a thread waits for the GIL after a 1 ms sleep, kept per second as
    'gil_wait' entries of (second start, mean wait ns, p90 wait ns)."""
    sleep_ns = int(GIL_GAUGE_SLEEP_S * 1e9)
    while True:
        started = time.time()
        waits = []
        while time.time() - started < 1.0:
            wall = time.perf_counter_ns()
            time.sleep(GIL_GAUGE_SLEEP_S)
            waits.append(time.perf_counter_ns() - wall - sleep_ns)
        waits.sort()
        timings["gil_wait"].append(
            (started, sum(waits) // len(waits), waits[int(len(waits) * 0.9)])
        )


def patch_request_builders(module):
    if TIME_PREPROCESSING:
        for name in TIMED_REQUEST_BUILDER_FUNCTIONS:
            setattr(module, name, timed(name, getattr(module, name)))
        threading.Thread(target=gauge_gil, name="gil-gauge", daemon=True).start()
    else:
        pass
    if PREPROCESSING_PRIORITY == "greatest":
        raise_preprocessing_priority(module)
    else:
        pass


def raise_preprocessing_priority(module):
    import dataclasses

    import torch

    set_context = module.set_qwen3_tts_preprocessing_context

    def set_context_raised(*args, **kwargs):
        set_context(*args, **kwargs)
        context = module._PREPROCESSING_CONTEXT
        if context is None or context.stream is None:
            return
        else:
            pass
        device_module = torch.get_device_module(context.stream.device)
        _, greatest = device_module.Stream.priority_range()
        raised = device_module.Stream(device=context.stream.device, priority=greatest)
        raised.wait_stream(context.stream)
        module._PREPROCESSING_CONTEXT = dataclasses.replace(context, stream=raised)

    module.set_qwen3_tts_preprocessing_context = set_context_raised


class PatchAfterImport(importlib.abc.MetaPathFinder):
    """Runs a patch on a module right after its first import."""

    def __init__(self, patches):
        self.patches = patches

    def find_spec(self, name, path, target=None):
        patch = self.patches.get(name)
        if patch is None:
            return None
        else:
            pass
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            else:
                pass
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
            else:
                pass
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


PATCHES = {}
if FIRST_EVENTS:
    PATCHES["sglang_omni.profiler.event_recorder_writer"] = keep_first_events
else:
    pass
if PROBE_DIRECTORY:
    PATCHES["sglang_omni.models.qwen3_tts.code_table"] = register_tables
    PATCHES["sglang_omni.models.qwen3_tts.model_runner"] = count_replays
else:
    pass
if TIME_PREPROCESSING or PREPROCESSING_PRIORITY == "greatest":
    PATCHES["sglang_omni.models.qwen3_tts.request_builders"] = patch_request_builders
else:
    pass
if TIME_PREPROCESSING:
    PATCHES["sglang_omni.preprocessing.cache_key"] = time_cache_keys
    PATCHES["sglang_omni.scheduling.reference_encoder"] = time_reference_service
    PATCHES["sglang_omni.models.qwen3_tts.sglang_model"] = time_prompt_builder
else:
    pass
if PATCHES:
    sys.meta_path.insert(0, PatchAfterImport(PATCHES))
else:
    pass
