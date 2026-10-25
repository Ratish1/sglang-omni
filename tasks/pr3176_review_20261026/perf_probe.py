"""Matched trained graph latency; both arms have four real slot rows."""

import json
import statistics
from pathlib import Path

import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphPool,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

apply_qwen_tts_transformers_compatibility_patches()
from qwen_tts import Qwen3TTSTokenizer

with torch.inference_mode():
    torch.manual_seed(3176)
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    decoder = tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    device = torch.device("cuda", 0)
    codes = torch.randint(
        decoder.config.codebook_size,
        (4, decoder.config.num_quantizers, 8),
        device=device,
    )
    runners = {}
    for name, frames in (("buffered", 0), ("ring", 8)):
        arena = Qwen3TTSCodecStateArena(
            incremental,
            num_slots=4,
            device=device,
            dtype=torch.bfloat16,
            in_place_frames=frames,
        )
        slots = [arena.acquire() for _ in range(4)]
        runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
            incremental,
            device=device,
            dtype=torch.bfloat16,
            num_quantizers=decoder.config.num_quantizers,
            mode="warm",
            fresh_frames=(8,),
            batch_sizes=(4,),
            min_free_gb=0,
            compile_fresh_frames=(8,),
            arena=arena,
            graph_pool=IncrementalCodecGraphPool(stream_priority=0),
        )
        runner.capture()
        assert runner.enabled, runner.stats()
        for _ in range(12):
            assert runner.decode_slots(codes, slots) is not None
        runners[name] = runner
    torch.cuda.synchronize()
    timings = {name: [] for name in runners}
    for turn in range(10):
        for name in ("buffered", "ring") if turn % 2 else ("ring", "buffered"):
            graph = next(iter(runners[name].graphs.values())).graph
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end) / 20)
    medians = {name: statistics.median(rows) for name, rows in timings.items()}
    result = {
        "batch": 4,
        "frames": 8,
        "compiled": True,
        "live_slots_per_arm": 4,
        "samples_ms": timings,
        "median_ms": medians,
        "improvement_pct": 100 * (1 - medians["ring"] / medians["buffered"]),
    }
    Path("/data/review3176oct26/perf.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
