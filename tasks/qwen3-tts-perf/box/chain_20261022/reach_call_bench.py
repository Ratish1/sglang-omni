"""The host cost of the first-chunk reach on the trained decoder, on the tree on PYTHONPATH: the vocoder's planner calls
first_chunk_reference_frames once per streaming first plan, which walks the decoder's modules through tail_frames.
Times 20,000 calls of each (median of 5 rounds, per call) on the served bf16 decoder, fused as served.

usage: PYTHONPATH=<D1 tree> python3 reach_call_bench.py [--card 3]
"""

import argparse
import statistics
import time

import torch

from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CALLS = 20000


def per_call_us(call) -> float:
    rounds = []
    for _ in range(5):
        start = time.perf_counter()
        for _ in range(CALLS):
            call()
        rounds.append(1e6 * (time.perf_counter() - start) / CALLS)
    return statistics.median(rounds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    args = parser.parse_args()
    device = torch.device("cuda", args.card)
    decoder = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None).model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    print(f"reach for one emitted frame: {incremental.first_chunk_reference_frames(1)} frames")
    print(f"tail_frames(1): {per_call_us(lambda: incremental.tail_frames(1)):.2f} us a call")
    print(f"first_chunk_reference_frames(1): {per_call_us(lambda: incremental.first_chunk_reference_frames(1)):.2f} us a call")


if __name__ == "__main__":
    main()
