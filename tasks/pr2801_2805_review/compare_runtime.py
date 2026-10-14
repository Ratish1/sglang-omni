"""Compare native MOSS session PCM and decode latency across code locations."""

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.moss_tts.audio_tokenizer import load_moss_audio_vocoder


def measure_codec(
    model_path: str,
    *,
    quantizer_count: int,
    stream_slots: int,
    initial_frames: int,
    steady_frames: int,
    session_module: str,
    output_prefix: Path,
) -> None:
    codec = load_moss_audio_vocoder(
        model_path,
        device="cuda:0",
        decoder_dtype=torch.float32,
        compute_dtype=torch.bfloat16,
    ).model
    runtime = importlib.import_module(session_module)
    session = runtime.CodecStreamSession(
        codec, stream_slots=stream_slots, n_vq=quantizer_count
    )
    waveforms: list[torch.Tensor] = []
    measurements: dict[str, list[float]] = {}
    try:
        captured = session.warmup_cuda_graph([initial_frames, steady_frames])
        assert captured == sorted({initial_frames, steady_frames})
        leased_slots = [session.acquire() for _ in range(stream_slots)]
        vocabulary = codec.quantizer.quantizers[0].codebook.num_embeddings
        for active_count in (1, 3, 16):
            slots = leased_slots[:active_count]
            codes = torch.randint(
                0,
                vocabulary,
                (quantizer_count, active_count, steady_frames),
                generator=torch.Generator().manual_seed(2805 + active_count),
            )
            slot_codes = {slot: codes[:, index] for index, slot in enumerate(slots)}
            latencies_ms: list[float] = []
            for _ in range(3):
                session.reset_slots(leased_slots)
                for _ in range(16):
                    session.step(slot_codes)
                torch.cuda.synchronize()
                for _ in range(64):
                    started = time.perf_counter_ns()
                    decoded = session.step(slot_codes)
                    latencies_ms.append((time.perf_counter_ns() - started) / 1e6)
                waveforms.extend(decoded[slot].clone() for slot in slots)
            measurements[str(active_count)] = latencies_ms
        torch.save(waveforms, output_prefix.with_suffix(".pt"))
        report = {
            "model": model_path,
            "quantizer_count": quantizer_count,
            "channels": codec.number_channels,
            "sample_rate": codec.sampling_rate,
            "steady_frames": steady_frames,
            "stream_slots": stream_slots,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "graph_steps": sum(session.cg_graph_t.values()),
            "eager_steps": sum(session.cg_eager_t.values()),
            "latencies_ms": measurements,
            "median_ms": {
                count: float(np.median(values))
                for count, values in measurements.items()
            },
        }
        output_prefix.with_suffix(".json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    name: value
                    for name, value in report.items()
                    if name != "latencies_ms"
                }
            )
        )
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantizers", type=int, required=True)
    parser.add_argument("--slots", type=int, required=True)
    parser.add_argument("--initial-frames", type=int, required=True)
    parser.add_argument("--steady-frames", type=int, required=True)
    parser.add_argument("--session-module", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    options = parser.parse_args()
    measure_codec(
        options.model,
        quantizer_count=options.quantizers,
        stream_slots=options.slots,
        initial_frames=options.initial_frames,
        steady_frames=options.steady_frames,
        session_module=options.session_module,
        output_prefix=options.output_prefix,
    )
