"""NVTX ranges over the Qwen3-TTS first chunk path, for an nsys profile of the server.

Put this directory on PYTHONPATH and set OMNI_TTFC_NVTX=1. No runtime code changes: the
functions are wrapped at import. Every range is thread local and nested; the request id
rides in the name where the path has one, so ttfc_census.py can join one request's
stages into its critical path.

  pre.prepare rid=R          the whole preprocessing call (preprocessing thread)
    pre.reference              one uncached reference encode
      pre.normalize            audio normalize
      pre.resample             librosa resample
      pre.speaker              speaker embedding: mel, copies, encoder
        pre.mel                  mel spectrogram (CPU)
        pre.spk_encoder          ECAPA forward (GPU, eager)
  ref.encode n=S             reference codes on the batcher thread (S samples)
  ref.sync b=N               the batch's stream synchronize
  sched.batch extend|decode bs=B toks=T   one scheduler forward
    mark sched.prefill rid=R   one per request of an extend batch
  sched.result extend|decode bs=B         process_batch_result
  voc.chunks n=N             one chunk batch on the vocoder thread: ingest, then the pump
    voc.chunk rid=R            one chunk: validate and ingest
  voc.initial rows=N         one initial worker batch
  voc.followup rows=N        one follow-up worker batch
    voc.cohort w=W rows=N init=0|1   one same width cohort
      voc.windows n=K          the window chain of a cohort
        voc.replay mode w=W rows=N   one graph lookup (hit or miss, see the census)
    voc.commit rid=R           the first chunk is committed at the end of this range
    voc.commit_followup rid=R
"""

import importlib.abc
import importlib.util
import os
import sys
import threading

ENABLED = os.environ.get("OMNI_TTFC_NVTX") == "1"
BUILDERS = "sglang_omni.models.qwen3_tts.request_builders"
RUNNER = "sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph"
VOCODER = "sglang_omni.models.qwen3_tts.streaming_vocoder"
SCHEDULER = "sglang_omni.scheduling.omni_scheduler"
REFERENCE = "qwen_tts.core.models.modeling_qwen3_tts"

install_lock = threading.Lock()


def ranged(fn, name):
    """Wrap fn in an NVTX range named by name(*args, **kwargs)."""
    import torch

    def wrapped(*args, **kwargs):
        torch.cuda.nvtx.range_push(name(*args, **kwargs))
        try:
            return fn(*args, **kwargs)
        finally:
            torch.cuda.nvtx.range_pop()

    return wrapped


def fixed(label):
    return lambda *args, **kwargs: label


def log_torch_settings():
    """The global knobs that move an eager stage: threads, cudnn, tf32, allocator."""
    import json

    import torch

    settings = {
        "pid": os.getpid(),
        "torch_threads": torch.get_num_threads(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    print(f"ttfc_nvtx settings {json.dumps(settings)}", file=sys.stderr, flush=True)


def patch_builders(module):
    log_torch_settings()
    hook = module._Qwen3TTSAdhocReferenceHook
    batcher = module._Qwen3TTSRefCodeBatcher
    module._prepare_qwen3_tts_request = ranged(
        module._prepare_qwen3_tts_request,
        lambda payload, **kwargs: f"pre.prepare rid={payload.request_id}",
    )
    encode_one = hook.encode_one

    def encode_one_ranged(self, item):
        with install_lock:
            if not getattr(self, "ttfc_nvtx_installed", False):
                import librosa

                self._wrapper._normalize_audio_inputs = ranged(
                    self._wrapper._normalize_audio_inputs, fixed("pre.normalize")
                )
                self._model.extract_speaker_embedding = ranged(
                    self._model.extract_speaker_embedding, fixed("pre.speaker")
                )
                librosa.resample = ranged(librosa.resample, fixed("pre.resample"))
                self.ttfc_nvtx_installed = True
        return encode_one(self, item)

    hook.encode_one = ranged(encode_one_ranged, fixed("pre.reference"))
    batcher._encode_waveform = ranged(
        batcher._encode_waveform,
        lambda self, waveform, sample_rate: f"ref.encode n={len(waveform)}",
    )
    batcher._synchronize_outcomes = ranged(
        batcher._synchronize_outcomes,
        lambda self, outcomes: f"ref.sync b={len(outcomes)}",
    )


def patch_reference(module):
    module.mel_spectrogram = ranged(module.mel_spectrogram, fixed("pre.mel"))
    encoder = module.Qwen3TTSSpeakerEncoder
    encoder.forward = ranged(encoder.forward, fixed("pre.spk_encoder"))


def batch_label(prefix, batch):
    mode = "extend" if batch.forward_mode.is_extend() else "decode"
    label = f"{prefix} {mode} bs={len(batch.reqs)}"
    if mode == "extend":
        label += f" toks={batch.extend_num_tokens}"
    return label


def patch_scheduler(module):
    import torch

    scheduler = module.OmniScheduler
    run_batch = scheduler._run_batch

    def run_batch_marked(self, batch, pp_proxy_tensors=None):
        if batch.forward_mode.is_extend():
            for req in batch.reqs:
                torch.cuda.nvtx.mark(f"sched.prefill rid={req.rid}")
        return run_batch(self, batch, pp_proxy_tensors)

    scheduler._run_batch = ranged(
        run_batch_marked,
        lambda self, batch, pp_proxy_tensors=None: batch_label("sched.batch", batch),
    )
    scheduler.process_batch_result = ranged(
        scheduler.process_batch_result,
        lambda self, batch, result: batch_label("sched.result", batch),
    )


def patch_runner(module):
    runner = module.Qwen3TTSIncrementalCodecCudaGraphRunner
    runner.decode_slots = ranged(
        runner.decode_slots,
        lambda self, codes, slots: (
            f"voc.replay {self._mode} w={int(codes.shape[2])} rows={int(codes.shape[0])}"
        ),
    )


def patch_vocoder(module):
    scheduler = module.Qwen3TTSStreamingVocoderScheduler
    scheduler.on_stream_chunk_batch = ranged(
        scheduler.on_stream_chunk_batch,
        lambda self, items: f"voc.chunks n={len(items)}",
    )
    scheduler._ingest_stream_item = ranged(
        scheduler._ingest_stream_item,
        lambda self, request_id, item: f"voc.chunk rid={request_id}",
    )
    scheduler._run_initial_batch = ranged(
        scheduler._run_initial_batch,
        lambda self, batch: f"voc.initial rows={len(batch)}",
    )
    scheduler._run_followup_batch = ranged(
        scheduler._run_followup_batch,
        lambda self, batch: f"voc.followup rows={len(batch)}",
    )
    scheduler._decode_incremental_cohort = ranged(
        scheduler._decode_incremental_cohort,
        lambda self, gpu_input, plans, incremental, stream: (
            f"voc.cohort w={int(plans[0].fresh_frames)} rows={len(plans)} "
            f"init={int(stream is self._decode_stream)}"
        ),
    )
    scheduler._decode_incremental_windows = ranged(
        scheduler._decode_incremental_windows,
        lambda self, gpu_input, plans, incremental, runner, split: (
            f"voc.windows n={len(split)}"
        ),
    )
    scheduler._commit_initial = ranged(
        scheduler._commit_initial,
        lambda self, request_id, state, plan, delta: f"voc.commit rid={request_id}",
    )
    scheduler._commit_followup = ranged(
        scheduler._commit_followup,
        lambda self, request_id, *args, **kwargs: (
            f"voc.commit_followup rid={request_id}"
        ),
    )


PATCHES = {
    BUILDERS: patch_builders,
    REFERENCE: patch_reference,
    SCHEDULER: patch_scheduler,
    RUNNER: patch_runner,
    VOCODER: patch_vocoder,
}


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


if ENABLED:
    sys.meta_path.insert(0, PatchOnImport())
