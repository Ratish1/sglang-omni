# Item 3 feasibility, the second form: torch's caching_precompile on the codec decode. First process: compile with the
# flag on across every size class, save with torch.compiler.save_cache_artifacts (the Dynamo cache under the Inductor
# cache dir). A later process with the same cache: the same calls; prints seconds per shape, new programs (0 means no
# trace ran) and whether the waveforms equal the first process's.
# usage: PYTHONPATH=<tree> TORCHINDUCTOR_CACHE_DIR=... python3 codec_precompile.py --waves FILE
import argparse
import os
import time

import torch
import torch._dynamo as dynamo
from torch._dynamo.utils import counters

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SHAPES = [(8, 8), (8, 6), (8, 1), (1, 8), (1, 1), (8, 7), (8, 3), (4, 8), (4, 4), (4, 6), (2, 8), (2, 2), (2, 3), (2, 1), (1, 2), (1, 3)]


def zero(arena):
    storage = arena.storage
    storage.frame_positions.zero_()
    for pool in (*storage.conv_histories.values(), *storage.transconv_overlaps.values()):
        pool.zero_()
    for ring in (*arena.key_rings, *arena.value_rings):
        ring.zero_()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--waves", required=True)
    args = parser.parse_args()
    torch._dynamo.config.caching_precompile = True
    began = time.perf_counter()
    decoder = load_qwen3_tts_tokenizer(MODEL, device="cuda:0", dtype="bfloat16", attn_implementation="sdpa", float32_encoder=False).model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=16, device=torch.device("cuda:0"), dtype=torch.bfloat16, in_place_frames=8)
    print(f"build {time.perf_counter() - began:.2f} s", flush=True)
    began = time.perf_counter()
    compiled = torch.compile(incremental.decode_tensors, dynamic=False, fullgraph=True)
    print(f"torch.compile {time.perf_counter() - began:.2f} s, dynamo cache {dict(counters['dynamo_cache'])}", flush=True)
    results = {}
    for batch, width in SHAPES:
        zero(arena)
        generator = torch.Generator().manual_seed(batch * 100 + width)
        codes = torch.randint(0, 1000, (batch, 16, width), generator=generator).cuda()
        state = arena.gather_by_index(torch.arange(batch, device="cuda:0"))
        for tensor in (codes, state.frame_positions, state.slot_index, state.kv_rings.slot_rows):
            dynamo.maybe_mark_dynamic(tensor, 0)
        dynamo.maybe_mark_dynamic(codes, 2)
        programs = counters["stats"]["unique_graphs"]
        began = time.perf_counter()
        waveform = compiled(codes, state)
        torch.cuda.synchronize()
        print(f"({batch}, {width}) {time.perf_counter() - began:.3f} s, new programs {counters['stats']['unique_graphs'] - programs}", flush=True)
        results[(batch, width)] = waveform.float().cpu().clone()
    print(f"programs {counters['stats']['unique_graphs']}", flush=True)
    if os.path.exists(args.waves):
        saved = torch.load(args.waves)
        print("equal to the first process:", sum(torch.equal(saved[key], results[key]) for key in results), "of", len(results), flush=True)
    else:
        began = time.perf_counter()
        artifacts = torch.compiler.save_cache_artifacts()
        print(f"saved {time.perf_counter() - began:.2f} s, {None if artifacts is None else len(artifacts[0]) / 1e6} MB", flush=True)
        torch.save(results, args.waves)


if __name__ == "__main__":
    main()
    os._exit(0)
