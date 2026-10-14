"""Measure real reference encoders and exercise their memory and backend contracts."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import soundfile
import torch

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.fishaudio_s2_pro.stages import (
    FishReferenceEncodeHook,
    configure_preprocessing_threads,
    load_codec,
)
from sglang_omni.models.higgs_tts.audio_codec import HiggsAudioCodec
from sglang_omni.utils.checkpoint import resolve_checkpoint


def fish_probe(gpu_index: int, output: Path) -> None:
    torch.cuda.set_device(gpu_index)
    torch.cuda.set_per_process_memory_fraction(
        (6 << 30) / torch.cuda.get_device_properties(gpu_index).total_memory,
        gpu_index,
    )
    configure_preprocessing_threads(8)
    checkpoint = resolve_checkpoint("fishaudio/s2-pro")
    cpu_codec = load_codec(checkpoint, "cpu")
    cuda_codec = load_codec(checkpoint, f"cuda:{gpu_index}")
    torch.backends.cudnn.allow_tf32 = False
    cpu_hook = FishReferenceEncodeHook(codec=cpu_codec, checkpoint_id=checkpoint)
    cuda_hook = FishReferenceEncodeHook(
        codec=cuda_codec, checkpoint_id=checkpoint, device=f"cuda:{gpu_index}"
    )
    results = {
        "intraop_threads": torch.get_num_threads(),
        "loaded_allocated": torch.cuda.memory_allocated(gpu_index),
        "loaded_reserved": torch.cuda.memory_reserved(gpu_index),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "samples": [],
    }
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    for sample in samples[:48]:
        waveform, sample_rate = soundfile.read(sample.ref_audio, dtype="float32")
        audio = torch.from_numpy(waveform).reshape(1, -1)
        started = time.perf_counter()
        cpu_codes = cpu_hook.encode_reference_waveform(audio, sample_rate)
        cpu_seconds = time.perf_counter() - started
        torch.cuda.reset_peak_memory_stats(gpu_index)
        started = time.perf_counter()
        cuda_codes = cuda_hook.encode_reference_waveform(audio, sample_rate)
        results["samples"].append(
            {
                "sample": sample.sample_id,
                "seconds": audio.numel() / sample_rate,
                "cpu_seconds": cpu_seconds,
                "gpu_seconds": time.perf_counter() - started,
                "codes_equal": torch.equal(cpu_codes, cuda_codes),
                "changed_codes": int(torch.count_nonzero(cpu_codes != cuda_codes)),
                "code_count": cpu_codes.numel(),
                "peak_allocated": torch.cuda.max_memory_allocated(gpu_index),
                "peak_reserved": torch.cuda.max_memory_reserved(gpu_index),
            }
        )
        output.write_text(json.dumps(results, indent=2))
    results["long_references"] = []
    for seconds in (36, 64, 96, 128):
        long_audio = audio.repeat(1, 1 + int(seconds * sample_rate / audio.numel()))
        long_audio = long_audio[:, : int(seconds * sample_rate)]
        torch.cuda.reset_peak_memory_stats(gpu_index)
        row = {"seconds": seconds}
        try:
            codes = cuda_hook.encode_reference_waveform(long_audio, sample_rate)
            row["codes_shape"] = list(codes.shape)
        except torch.OutOfMemoryError as error:
            row["oom"] = str(error)
        finally:
            row["peak_allocated"] = torch.cuda.max_memory_allocated(gpu_index)
            row["peak_reserved"] = torch.cuda.max_memory_reserved(gpu_index)
            results["long_references"].append(row)
            output.write_text(json.dumps(results, indent=2))


def higgs_probe(gpu_index: int, output: Path) -> None:
    torch.cuda.set_device(gpu_index)
    codec = HiggsAudioCodec.from_pretrained(
        "bosonai/higgs-audio-v3-tts-4b",
        device=f"cuda:{gpu_index}",
        dtype=torch.bfloat16,
    )
    codec.model.acoustic_encoder = torch.compile(
        codec.model.acoustic_encoder, mode="default", dynamic=True
    )
    codec.encode_reference(torch.zeros(codec.SAMPLE_RATE))
    samples = load_seedtts_samples("zhaochenyang20/seed-tts-eval-arrow", split="en")
    results = {"samples": []}
    for sample in samples[:48]:
        waveform, sample_rate = soundfile.read(sample.ref_audio, dtype="float32")
        audio = torch.from_numpy(waveform)
        codes = []
        timings = []
        for _ in range(2):
            started = time.perf_counter()
            codes.append(codec.encode_reference(audio, sample_rate=sample_rate))
            timings.append(time.perf_counter() - started)
        results["samples"].append(
            {
                "sample": sample.sample_id,
                "seconds": audio.numel() / sample_rate,
                "cold_seconds": timings[0],
                "warm_seconds": timings[1],
                "aa_equal": torch.equal(codes[0], codes[1]),
                "aa_changed_codes": int(torch.count_nonzero(codes[0] != codes[1])),
                "code_count": codes[0].numel(),
            }
        )
        torch.save(codes[0], output.parent / f"{sample.sample_id}.pt")
        output.write_text(json.dumps(results, indent=2))
    context_method = getattr(codec, "encode_kernel_context", None)
    if context_method is None:
        return
    else:
        pass
    encode_inside = threading.Event()
    allow_encode = threading.Event()
    observed_decode_flags = []
    failures = []

    def hold_encode(module: torch.nn.Module, arguments: tuple[torch.Tensor]) -> None:
        encode_inside.set()
        assert allow_encode.wait(timeout=10)

    def record_decode(module: torch.nn.Module, arguments: tuple[torch.Tensor]) -> None:
        observed_decode_flags.append(
            {
                "cudnn": torch.backends.cudnn.enabled,
                "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
            }
        )

    encode_hook = codec.model.acoustic_encoder.register_forward_pre_hook(hold_encode)
    decode_hook = codec.model.acoustic_decoder.register_forward_pre_hook(record_decode)

    def encode() -> None:
        try:
            codec.encode_reference(audio, sample_rate=sample_rate)
        except Exception as error:
            failures.append(str(error))

    encode_thread = threading.Thread(target=encode)
    encode_thread.start()
    try:
        assert encode_inside.wait(timeout=10)
        codec.decode(codes[0][:25])
    finally:
        allow_encode.set()
        encode_thread.join(timeout=15)
        encode_hook.remove()
        decode_hook.remove()
    results["concurrent_decode_flags"] = observed_decode_flags
    results["concurrent_failures"] = failures
    results["flags_after"] = {
        "cudnn": torch.backends.cudnn.enabled,
        "cudnn_sdp": torch.backends.cuda.cudnn_sdp_enabled(),
    }
    output.write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("fish", "higgs"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    if arguments.model == "fish":
        fish_probe(arguments.gpu, arguments.output)
    else:
        higgs_probe(arguments.gpu, arguments.output)


if __name__ == "__main__":
    main()
