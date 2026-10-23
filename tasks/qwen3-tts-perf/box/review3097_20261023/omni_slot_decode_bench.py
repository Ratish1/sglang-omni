"""Device time of Qwen3-Omni code2wav's served slot decode on the tree on PYTHONPATH: the bf16 code2wav as the scheduler
loads it (load_code2wav_model: channels-last, fused transformer; fuse_vocoder_decoder), Qwen3OmniIncrementalCodec on a
CodecStatePool, a stream's first 10-frame decode and a later one from warm slots, at rows 1 and 8, each captured in one
CUDA graph under no_grad and replayed 50 times per trial, 7 trials; prints the median microseconds per decode.

usage: PYTHONPATH=<tree> python3 omni_slot_decode_bench.py [--card 2]
"""

import argparse
import json
import statistics

import torch

from sglang_omni.models.qwen3_omni.components.code2wav import Qwen3OmniIncrementalCodec
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import load_code2wav_model
from sglang_omni.models.qwen3_omni.components.codec_state_pool import CodecStatePool
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
FRAMES = 10
REPLAYS = 50
TRIALS = 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    torch.manual_seed(0)
    model = load_code2wav_model(MODEL, device=str(device), dtype="bfloat16")
    fuse_vocoder_decoder(model.decoder)
    codec = Qwen3OmniIncrementalCodec(model)
    pool = CodecStatePool(
        codec.spec,
        num_slots=8,
        capture_slots=0,
        ring_frames=codec.spec.retained_context + FRAMES,
        device=device,
        dtype=model.dtype,
    )
    slots = [pool.acquire() for _ in range(8)]
    stream = torch.cuda.Stream(device)
    result = {}
    for rows in (1, 8):
        index = pool.stage_index(slots[:rows]).clone()
        codes = torch.randint(
            model.config.codebook_size, (rows, model.config.num_quantizers, FRAMES), device=device
        )
        for kind, first_decode in (("first", True), ("next", False)):
            with torch.cuda.stream(stream), torch.no_grad():
                codec.decode_slots(codes, pool, index, first_decode=True)
                for _ in range(3):
                    codec.decode_slots(codes, pool, index, first_decode=first_decode)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream), torch.no_grad():
                codec.decode_slots(codes, pool, index, first_decode=first_decode)
            times = []
            for _ in range(TRIALS):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(REPLAYS):
                    graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1000 / REPLAYS)
            result[f"{kind}_rows{rows}_us"] = round(statistics.median(times), 1)
            del graph
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
