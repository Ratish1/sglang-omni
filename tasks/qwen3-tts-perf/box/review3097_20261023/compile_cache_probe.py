"""Inductor's FX graph cache on the served Qwen3-TTS vocoder's compiled warm graph: one warm runner of width 8 at
batches 1 and 4, compiled as served, on the served bf16 decoder fused as served and a codec state arena. Prints the
capture time, Inductor's cache hit, miss and bypass counters and every bypass reason, so two fresh processes on one
tree show whether the second reuses the first's compile.

usage: PYTHONPATH=<tree> python3 compile_cache_probe.py [--card 2]
"""

import argparse
import json
import logging
import time

import torch
from torch._dynamo.utils import counters

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphPool,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


class Reasons(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if "ypass" in message or "cache" in message.lower():
            self.lines.append(message[:400])
        else:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=2)
    args = parser.parse_args()
    reasons = Reasons()
    for name in ("torch._inductor.codecache", "torch._functorch._aot_autograd.autograd_cache"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(reasons)
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    decoder = tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=4, device=device, dtype=torch.bfloat16)
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        incremental,
        device=device,
        dtype=torch.bfloat16,
        num_quantizers=int(tokenizer.model.encoder_valid_num_quantizers),
        mode="warm",
        fresh_frames=(8,),
        batch_sizes=(1, 4),
        min_free_gb=0.0,
        compile_fresh_frames=(8,),
        arena=arena,
        graph_pool=IncrementalCodecGraphPool(stream_priority=0),
    )
    start = time.perf_counter()
    with torch.inference_mode():
        runner.capture()
    torch.cuda.synchronize(device)
    print(
        json.dumps(
            {
                "capture_s": round(time.perf_counter() - start, 1),
                "build": str(runner.stats()["build"]),
                "inductor": {key: value for key, value in counters["inductor"].items() if "cache" in key},
                "aot_autograd": {key: value for key, value in counters["aot_autograd"].items() if "cache" in key},
                "reasons": sorted(set(reasons.lines))[:20],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
