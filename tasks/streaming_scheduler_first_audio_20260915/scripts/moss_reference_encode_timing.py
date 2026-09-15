"""Time every step of the MOSS-TTS Local reference encode, offline on one GPU.

Usage, from the repository root with no server on the GPU:
    python moss_reference_encode_timing.py --model OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5
"""

import argparse
import math
import statistics
import threading
import time

import torch
import torchaudio

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.moss_tts.audio_tokenizer import (
    load_moss_audio_encoder,
    resolve_moss_audio_dtype,
)
from sglang_omni.models.moss_tts_local import stages
from sglang_omni.preprocessing.cache_key import reference_path_cache_key
from sglang_omni.utils.cpu import effective_cpu_count


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1]


def report(name: str, values: list[float], unit: str = "ms") -> None:
    print(
        f"  {name:<34} n={len(values):<5} mean={statistics.mean(values):8.2f} "
        f"p50={pct(values, 50):8.2f} p90={pct(values, 90):8.2f} "
        f"p95={pct(values, 95):8.2f} p99={pct(values, 99):8.2f} "
        f"max={max(values):8.2f} {unit}"
    )


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=16)
    args = parser.parse_args()

    intraop_threads = stages._configure_pipeline_threads(16)
    print(
        f"effective cpus {effective_cpu_count()}, torch intra-op threads "
        f"{intraop_threads}, torchaudio {torchaudio.__version__}"
    )
    processor = stages._load_moss_tts_local_processor(args.model)
    n_vq = int(processor.model_config.n_vq)
    encoder = load_moss_audio_encoder(
        stages._resolve_audio_tokenizer_model_path(processor, None),
        device="cuda:0",
        compute_dtype=resolve_moss_audio_dtype(
            "bfloat16", name="compute_dtype", allow_none=True
        ),
        attention_backend="auto",
    )
    samples = load_seedtts_samples(args.meta, split="en")
    paths = list(dict.fromkeys(sample.ref_audio for sample in samples))
    print(f"references: {len(paths)} unique of {len(samples)} samples")
    for path in paths[:8]:
        encoder.encode_paths([path], num_quantizers=n_vq)
    torch.cuda.synchronize()

    print("\n1. one reference at a time")
    steps: dict[str, list[float]] = {
        name: []
        for name in (
            "torchaudio.info x2",
            "path cache key",
            "torchaudio.load",
            "resample",
            "prepare waveform + H2D",
            "encoder forward",
            "codes D2H",
            "total",
        )
    }
    durations, sample_rates = [], []
    for path in paths[: args.samples]:
        total_start = time.perf_counter()
        start = time.perf_counter()
        torchaudio.info(path)
        torchaudio.info(path)
        steps["torchaudio.info x2"].append(elapsed_ms(start))
        start = time.perf_counter()
        reference_path_cache_key(path)
        steps["path cache key"].append(elapsed_ms(start))
        start = time.perf_counter()
        waveform, sample_rate = torchaudio.load(path)
        steps["torchaudio.load"].append(elapsed_ms(start))
        sample_rates.append(int(sample_rate))
        start = time.perf_counter()
        if int(sample_rate) != encoder.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform=waveform,
                orig_freq=int(sample_rate),
                new_freq=encoder.sample_rate,
            )
        steps["resample"].append(elapsed_ms(start))
        durations.append(waveform.shape[-1] / encoder.sample_rate)
        start = time.perf_counter()
        prepared = encoder._prepare_waveform(waveform, encoder.sample_rate)
        torch.cuda.synchronize()
        steps["prepare waveform + H2D"].append(elapsed_ms(start))
        start = time.perf_counter()
        with torch.inference_mode():
            encoded = encoder.model.batch_encode([prepared], num_quantizers=n_vq)
        torch.cuda.synchronize()
        steps["encoder forward"].append(elapsed_ms(start))
        start = time.perf_counter()
        encoded.audio_codes.detach().to(device="cpu", dtype=torch.long)
        steps["codes D2H"].append(elapsed_ms(start))
        steps["total"].append(elapsed_ms(total_start))
    report("reference seconds", durations, unit="s")
    print(
        f"  source sample rates: {sorted(set(sample_rates))} -> {encoder.sample_rate}"
    )
    for name, values in steps.items():
        report(name, values)

    print("\n2. batches as the encode worker runs them (load_paths + encode_waveforms)")
    for batch_size in (1, 2, 4, 8):
        load_ms, encode_ms, padding = [], [], []
        for offset in range(
            0, min(args.samples, len(paths)) - batch_size + 1, batch_size
        ):
            group = paths[offset : offset + batch_size]
            start = time.perf_counter()
            loaded = encoder.load_paths(group)
            load_ms.append(elapsed_ms(start))
            lengths = [waveform.shape[-1] for waveform, _ in loaded]
            padding.append(batch_size * max(lengths) / sum(lengths))
            start = time.perf_counter()
            encoder.encode_waveforms(loaded, num_quantizers=n_vq)
            torch.cuda.synchronize()
            encode_ms.append(elapsed_ms(start))
        print(f"  batch {batch_size}")
        report("load_paths", load_ms)
        report("encode_waveforms (prep+fwd+D2H)", encode_ms)
        report("padded / real samples", padding, unit="x")

    print("\n3. bursts of 16 concurrent requests through the production encoder")
    batched = stages._BatchedReferenceEncoder(encoder, n_vq=n_vq)
    service = stages._MossLocalReferenceEncoder(
        batched, n_vq=n_vq, max_items=8192, max_bytes=64 * 1024 * 1024
    )
    worker_batches: list[tuple[int, float, float]] = []
    original_load = encoder.load_paths
    original_encode = encoder.encode_waveforms

    def load_paths(group):
        start = time.perf_counter()
        loaded = original_load(group)
        worker_batches.append((len(group), elapsed_ms(start), 0.0))
        return loaded

    def encode_waveforms(waveforms, *, num_quantizers):
        start = time.perf_counter()
        codes = original_encode(waveforms, num_quantizers=num_quantizers)
        count, load, _ = worker_batches[-1]
        worker_batches[-1] = (count, load, elapsed_ms(start))
        return codes

    encoder.load_paths = load_paths
    encoder.encode_waveforms = encode_waveforms
    latencies: list[float] = []
    rank_latencies: dict[int, list[float]] = {rank: [] for rank in range(16)}
    burst_paths = paths[args.samples :]
    for round_index in range(args.rounds):
        group = burst_paths[round_index * 16 : (round_index + 1) * 16]
        if len(group) < 16:
            break
        barrier = threading.Barrier(16)
        results: list[float] = []
        lock = threading.Lock()

        def request(path: str) -> None:
            barrier.wait()
            start = time.perf_counter()
            service.encode(path)
            with lock:
                results.append(elapsed_ms(start))

        threads = [threading.Thread(target=request, args=(path,)) for path in group]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        latencies.extend(results)
        for rank, value in enumerate(sorted(results)):
            rank_latencies[rank].append(value)
    report("request latency", latencies)
    for rank in (0, 7, 8, 15):
        report(f"completion rank {rank + 1} of 16", rank_latencies[rank])
    report("worker batch size", [float(b[0]) for b in worker_batches], unit="refs")
    report("worker load_paths", [b[1] for b in worker_batches])
    report("worker encode_waveforms", [b[2] for b in worker_batches])


if __name__ == "__main__":
    main()
