"""Two request flow probes for a Qwen3-TTS server, without touching runtime code.

Put this directory on PYTHONPATH and set OMNI_FLOW_PROBE to an output prefix. Every server
process appends JSON lines to <prefix>.<pid>.jsonl:

  kind "prepare"   one per request: wall and thread CPU seconds of the whole preprocessing
                   call. wall minus cpu is time the thread waited (device, lock, queue).
  kind "reference" one per uncached reference: the whole encode, and inside it audio
                   normalize, resample and speaker encoder, each wall and cpu. The rest
                   of its wall is the wait for the reference codes.
  kind "ref_code"  one per reference encode on the batcher thread, wall and cpu;
                   "ref_code_sync" is the batch's stream synchronize.
  kind "replay"    one per vocoder graph lookup: runner mode, width, live rows, bucket,
                   hit or miss, host wall.
  kind "cohort"    one per vocoder cohort decode: width, rows, initial or follow-up
                   stream, path (graph, windows with their split, eager), host wall.
"""

import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time

PREFIX = os.environ.get("OMNI_FLOW_PROBE")
BUILDERS = "sglang_omni.models.qwen3_tts.request_builders"
RUNNER = "sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph"
VOCODER = "sglang_omni.models.qwen3_tts.streaming_vocoder"

write_lock = threading.Lock()
local = threading.local()
sink = []


def write(record):
    record["t"] = time.time()
    record["thread"] = threading.current_thread().name
    with write_lock:
        if not sink:
            # note(ratish): held open for the life of the server process
            sink.append(
                open(f"{PREFIX}.{os.getpid()}.jsonl", "a", buffering=1)  # noqa: SIM115
            )
        sink[0].write(json.dumps(record) + "\n")


def timed(fn, *args, **kwargs):
    wall, cpu = time.perf_counter(), time.thread_time()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - wall, time.thread_time() - cpu


def patch_builders(module):
    prepare = module.prepare_qwen3_tts_request
    hook = module.Qwen3TTSAdhocReferenceHook
    batcher = module.Qwen3TTSRefCodeBatcher
    encode_one, encode_waveform, synchronize = (
        hook.encode_one,
        batcher.encode_waveform,
        batcher.synchronize_outcomes,
    )

    def prepare_timed(*args, **kwargs):
        result, wall, cpu = timed(prepare, *args, **kwargs)
        write({"kind": "prepare", "wall_s": wall, "cpu_s": cpu})
        return result

    def part(name, fn):
        # note(ratish): installed once; each thread adds to its own open record
        def wrapped(*args, **kwargs):
            result, wall, cpu = timed(fn, *args, **kwargs)
            parts = getattr(local, "parts", None)
            if parts is not None:
                parts[f"{name}_wall_s"] = parts.get(f"{name}_wall_s", 0.0) + wall
                parts[f"{name}_cpu_s"] = parts.get(f"{name}_cpu_s", 0.0) + cpu
            return result

        return wrapped

    def encode_one_timed(self, item):
        with write_lock:
            if not getattr(self, "flow_probe_installed", False):
                import librosa

                self._wrapper._normalize_audio_inputs = part(
                    "normalize", self._wrapper._normalize_audio_inputs
                )
                self._model.extract_speaker_embedding = part(
                    "speaker", self._model.extract_speaker_embedding
                )
                librosa.resample = part("resample", librosa.resample)
                self.flow_probe_installed = True
        local.parts = {}
        try:
            result, wall, cpu = timed(encode_one, self, item)
            write({"kind": "reference", "wall_s": wall, "cpu_s": cpu, **local.parts})
        finally:
            local.parts = None
        return result

    def encode_waveform_timed(self, waveform, sample_rate):
        result, wall, cpu = timed(encode_waveform, self, waveform, sample_rate)
        write(
            {
                "kind": "ref_code",
                "encode_wall_s": wall,
                "encode_cpu_s": cpu,
                "samples": int(len(waveform)),
                "graph": self._graph_runner is not None,
            }
        )
        return result

    def synchronize_timed(self, outcomes):
        result, wall, _ = timed(synchronize, self, outcomes)
        write({"kind": "ref_code_sync", "wall_s": wall, "batch": len(outcomes)})
        return result

    module.prepare_qwen3_tts_request = prepare_timed
    hook.encode_one = encode_one_timed
    batcher.encode_waveform = encode_waveform_timed
    batcher.synchronize_outcomes = synchronize_timed


def patch_runner(module):
    runner = module.Qwen3TTSIncrementalCodecCudaGraphRunner
    decode_slots = runner.decode_slots

    def decode_slots_timed(self, codes, slots):
        rows, width = int(codes.shape[0]), int(codes.shape[2])
        result, wall, _ = timed(decode_slots, self, codes, slots)
        bucket = next((size for size in self._batch_sizes if size >= rows), None)
        local.replays = getattr(local, "replays", 0) + (result is not None)
        write(
            {
                "kind": "replay",
                "mode": self._mode,
                "width": width,
                "rows": rows,
                "bucket": bucket,
                "hit": result is not None,
                "wall_s": wall,
            }
        )
        return result

    runner.decode_slots = decode_slots_timed


def patch_vocoder(module):
    scheduler = module.Qwen3TTSStreamingVocoderScheduler
    cohort, windows = (
        scheduler.decode_incremental_cohort,
        scheduler.decode_incremental_windows,
    )

    def windows_timed(self, gpu_input, plans, incremental, runner, split):
        local.split = list(split)
        return windows(self, gpu_input, plans, incremental, runner, split)

    def cohort_timed(self, gpu_input, plans, incremental, stream):
        local.split, local.replays = None, 0
        result, wall, _ = timed(cohort, self, gpu_input, plans, incremental, stream)
        if local.split is not None:
            path = "windows"
        elif local.replays:
            path = "graph"
        else:
            path = "eager"
        write(
            {
                "kind": "cohort",
                "width": int(plans[0].fresh_frames),
                "rows": len(plans),
                "initial_stream": stream is self._decode_stream,
                "path": path,
                "split": local.split,
                "wall_s": wall,
            }
        )
        return result

    scheduler.decode_incremental_windows = windows_timed
    scheduler.decode_incremental_cohort = cohort_timed


PATCHES = {BUILDERS: patch_builders, RUNNER: patch_runner, VOCODER: patch_vocoder}


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES or name in getattr(self, "seen", ()):
            return None
        self.seen = (*getattr(self, "seen", ()), name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            PATCHES[name](module)

        spec.loader.exec_module = exec_and_patch
        return spec


if PREFIX:
    sys.meta_path.insert(0, PatchOnImport())
