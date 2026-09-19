"""Graph memory of one production vocoder graph runner, key by key.

Builds the real Qwen3TTSIncrementalCodecCudaGraphRunner (warm mode, the production widths
and batch buckets, one shared pool) against a real arena, with the decode compiled for
every width or for none, and prints the device's reserved and allocated memory after each
key's capture, in the runner's own capture order. Run one arm per process.

usage: python vocoder_runner_memory.py --model DIR --arm eager|compiled [--widths 1-8]
"""

from __future__ import annotations

import argparse
import time

import torch

from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=("eager", "compiled"), required=True)
    parser.add_argument("--widths", default="1-8")
    parser.add_argument("--slots", type=int, default=32)
    args = parser.parse_args()
    low, high = (int(part) for part in args.widths.split("-"))
    widths = tuple(range(low, high + 1))
    device = torch.device("cuda", 0)
    tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
        args.model, device=str(device), dtype="bfloat16", attn_implementation=None
    )
    decoder = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder)
    arena = Qwen3TTSCodecStateArena(
        decoder, num_slots=args.slots, device=device, dtype=torch.bfloat16
    )
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        decoder,
        device=device,
        dtype=torch.bfloat16,
        num_quantizers=16,
        mode="warm",
        fresh_frames=widths,
        batch_sizes=(1, 2, 4, 8),
        min_free_gb=0.0,
        compile_fresh_frames=widths if args.arm == "compiled" else (),
        arena=arena,
    )
    mib = 2**20
    print(f"arm {args.arm}, widths {widths}, torch {torch.__version__}")
    print(
        f"before capture: reserved {torch.cuda.memory_reserved(device) / mib:.0f} MiB, "
        f"allocated {torch.cuda.memory_allocated(device) / mib:.0f} MiB"
    )
    original = runner._capture_graph
    started = time.perf_counter()

    def capture_and_report(key, **kwargs):
        captured = original(key, **kwargs)
        torch.cuda.synchronize()
        print(
            f"  after w{key.fresh_frames} b{key.batch_bucket}: "
            f"reserved {torch.cuda.memory_reserved(device) / mib:8.0f} MiB, "
            f"allocated {torch.cuda.memory_allocated(device) / mib:8.0f} MiB, "
            f"t {time.perf_counter() - started:6.1f} s",
            flush=True,
        )
        return captured

    runner._capture_graph = capture_and_report
    runner.capture()
    stats = runner.stats()
    print(f"enabled {stats['enabled']}, disable_reason {stats['disable_reason']}")
    print(
        f"graph footprint {stats['memory'].get('graph_footprint_bytes', 0) / mib:.0f} MiB"
    )


if __name__ == "__main__":
    main()
