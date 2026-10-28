"""Block 6 component benchmark: the Qwen3-TTS vocoder's eager whole-utterance decode, which
serves every non-streaming CustomVoice and VoiceDesign batch, under three backend choices.

Each arm runs in its own process, since cuDNN's plan caches live per process. The workload is
block 6's CustomVoice ZH part01 in case order, batched by --batch-cycle (the vocoder batches 1 to
8 requests); each request's codes are a slice of real speech, SeedTTS ZH references encoded by the
checkpoint's speech tokenizer and laid end to end. The decoder is built as the vocoder builds it
(load_qwen3_tts_tokenizer, then fuse_vocoder_decoder) and called as vocode_payloads calls it.

Per arm: each batch's decode time, the first and second half of the batches apart (the first
half meets most new shapes); for every fortieth batch the waveform's relative L2 error against a
float32 decode of the same codes (math SDPA, no TF32, unfused SnakeBeta); last, the device filled
to a few MiB with a cached block left for activations, and one batch of a new shape decoded.

arms: main (torch's order: cuDNN SDPA first on H100), no_cudnn_sdpa (flash, efficient, math, the
list higgs_tts's encode uses), no_cudnn (that list and cuDNN convolutions off).

PYTHONPATH=<main tree> python3 eager_decode_backends.py --arm main --device cuda:0 \
    --frames <block 6 scan jsonl> --out <json>
"""

import argparse
import contextlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

CHUNK_FRAMES = 300
CONTEXT_FRAMES = 25
CHECK_EVERY_BATCHES = 40
MIB = 1024**2


def arm_context(arm: str) -> contextlib.AbstractContextManager:
    stack = contextlib.ExitStack()
    if arm == "main":
        return stack
    else:
        pass
    stack.enter_context(
        sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        )
    )
    if arm == "no_cudnn":
        stack.enter_context(torch.backends.cudnn.flags(enabled=False))
    else:
        pass
    return stack


def chunk_shapes(batch_requests: int, padded_frames: int) -> list[tuple[int, int]]:
    shapes = []
    start_frame = 0
    while start_frame < padded_frames:
        end_frame = min(start_frame + CHUNK_FRAMES, padded_frames)
        context_frames = (
            CONTEXT_FRAMES if start_frame - CONTEXT_FRAMES > 0 else start_frame
        )
        shapes.append((batch_requests, end_frame - start_frame + context_frames))
        start_frame = end_frame
    return shapes


def code_pool(model_path: str, device: str, reference_count: int) -> torch.Tensor:
    tokenizer = load_qwen3_tts_tokenizer(
        model_path, device=device, dtype="bfloat16", attn_implementation="sdpa"
    )
    samples = load_seedtts_samples(
        "zhaochenyang20/seed-tts-eval-arrow", reference_count, split="zh"
    )
    codes = []
    # note (ratish): no cuDNN SDPA kernel launches before the decode, which the full device mode reads.
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        for start in range(0, len(samples), 16):
            encoded = tokenizer.encode(
                [sample.ref_audio for sample in samples[start : start + 16]]
            )
            codes.extend(code.cpu() for code in encoded.audio_codes)
    return torch.cat(codes, dim=0)


def relative_error(waveform: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(waveform - reference) / np.linalg.norm(reference))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arm", choices=("main", "no_cudnn_sdpa", "no_cudnn"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--references", type=int, default=400)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--full-device-mib",
        type=int,
        default=0,
        help="fill the device to this many free MiB before the first decode (a cached "
        "--room-gib block left for activations) and count each batch's failures by kind",
    )
    parser.add_argument("--room-gib", type=int, default=8)
    parser.add_argument(
        "--batch-cycle",
        default="8",
        help="requests per batch, cycled, as comma-separated counts (the vocoder batches 1 to 8)",
    )
    arguments = parser.parse_args()
    device = arguments.device
    torch.cuda.set_device(torch.device(device))

    pool = code_pool(arguments.model, device, arguments.references)
    frames = [
        int(json.loads(line)["frames"])
        for line in arguments.frames.read_text().splitlines()
        if json.loads(line).get("frames")
    ]
    batch_cycle = [int(count) for count in arguments.batch_cycle.split(",")]
    batches = []
    offset = 0
    start = 0
    while start < len(frames):
        batch_requests = batch_cycle[len(batches) % len(batch_cycle)]
        batch = []
        for request_frames in frames[start : start + batch_requests]:
            if offset + request_frames > int(pool.shape[0]):
                offset = 0
            else:
                pass
            batch.append(pool[offset : offset + request_frames].clone())
            offset += request_frames
        batches.append(batch)
        start += batch_requests
    shapes = {
        shape
        for batch in batches
        for shape in chunk_shapes(len(batch), max(int(code.shape[0]) for code in batch))
    }
    print(
        f"{arguments.arm}: {len(frames)} requests in {len(batches)} batches, "
        f"{len(shapes)} distinct (batch, chunk frames) decoder shapes, code pool {int(pool.shape[0])} frames"
    )

    tokenizer = load_qwen3_tts_tokenizer(
        arguments.model, device=device, dtype="bfloat16", attn_implementation="sdpa"
    )
    fuse_vocoder_decoder(tokenizer.model.decoder)
    reference_tokenizer = load_qwen3_tts_tokenizer(
        arguments.model, device=device, dtype="float32", attn_implementation="sdpa"
    )

    if arguments.full_device_mib:
        room = torch.empty(arguments.room_gib * 1024 * MIB, dtype=torch.uint8, device=device)
        del room
        free_bytes, _ = torch.cuda.mem_get_info(device)
        filler = torch.empty(
            free_bytes - arguments.full_device_mib * MIB, dtype=torch.uint8, device=device
        )
        free_bytes, _ = torch.cuda.mem_get_info(device)
        print(
            f"full device from the start: reserved {torch.cuda.memory_reserved(device) / MIB:.0f} MiB, "
            f"device free {free_bytes / MIB:.1f} MiB",
            flush=True,
        )
        failures: dict[str, list[int]] = {}
        for index, batch in enumerate(batches):
            try:
                with arm_context(arguments.arm):
                    tokenizer.decode([{"audio_codes": code} for code in batch])
            except RuntimeError as error:
                kind = str(error).split("\n")[0][:160]
                failures.setdefault(kind, []).append(index)
            else:
                pass
        free_bytes, _ = torch.cuda.mem_get_info(device)
        result = {
            "arm": arguments.arm,
            "full_device_mib": arguments.full_device_mib,
            "batches": len(batches),
            "failed_batches": sum(len(indices) for indices in failures.values()),
            "failures": {
                kind: {"batches": len(indices), "first": indices[:5]}
                for kind, indices in failures.items()
            },
            "device_free_after_mib": free_bytes / MIB,
        }
        del filler
        print(json.dumps(result, indent=1))
        arguments.out.write_text(json.dumps(result, indent=1))
        return
    else:
        pass

    decode_ms = []
    errors = []
    for index, batch in enumerate(batches):
        items = [{"audio_codes": code} for code in batch]
        torch.cuda.synchronize()
        start_seconds = time.perf_counter()
        with arm_context(arguments.arm):
            waveforms, _ = tokenizer.decode(items)
        decode_ms.append((time.perf_counter() - start_seconds) * 1000)
        if index % CHECK_EVERY_BATCHES == 0:
            with (
                sdpa_kernel([SDPBackend.MATH]),
                torch.backends.cudnn.flags(enabled=True, allow_tf32=False),
            ):
                references, _ = reference_tokenizer.decode(items)
            errors.extend(
                relative_error(waveform, reference)
                for waveform, reference in zip(waveforms, references)
            )
        else:
            pass

    half = len(decode_ms) // 2
    result = {
        "arm": arguments.arm,
        "batches": len(decode_ms),
        "decode_seconds": sum(decode_ms) / 1000,
        "first_half_seconds": sum(decode_ms[:half]) / 1000,
        "second_half_seconds": sum(decode_ms[half:]) / 1000,
        "batch_ms_p50": statistics.median(decode_ms),
        "batch_ms_p99": sorted(decode_ms)[int(len(decode_ms) * 0.99)],
        "batch_ms_max": max(decode_ms),
        "batches_over_100_ms": sum(1 for value in decode_ms if value > 100),
        "relative_l2_vs_float32_p50": statistics.median(errors),
        "relative_l2_vs_float32_max": max(errors),
        "checked_requests": len(errors),
    }

    room = torch.empty(3 * 1024 * MIB, dtype=torch.uint8, device=device)
    del room
    for headroom_mib, new_frames in ((8, 151), (4, 163), (2, 167)):
        free_bytes, _ = torch.cuda.mem_get_info(device)
        filler = torch.empty(
            free_bytes - headroom_mib * MIB, dtype=torch.uint8, device=device
        )
        items = [
            {"audio_codes": pool[start : start + new_frames - 4 * start // 100]}
            for start in (0, 100, 200)
        ]
        try:
            with arm_context(arguments.arm):
                tokenizer.decode(items)
        except RuntimeError as error:
            outcome = f"failed: {str(error)[:240]}"
        else:
            outcome = "ran"
        free_bytes, _ = torch.cuda.mem_get_info(device)
        result[f"full_device_{headroom_mib}_mib"] = (
            f"{outcome} (device free after {free_bytes / MIB:.1f} MiB)"
        )
        del filler
    print(json.dumps(result, indent=1))
    arguments.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
