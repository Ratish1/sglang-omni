# Item 3 feasibility: torch 2.14's aot_compile on the codec decode, the program a warm boot re-traces. Save: build the
# decoder as the vocoder does, trace at --shape with batch and width marked dynamic, compile, serialize, then decode with
# the AOT function and with the ordinary compiled kernel. Load (a fresh process): load the saved function and decode the
# same codes at several sizes; prints seconds per step and whether the waveforms equal the save phase's.
# usage: PYTHONPATH=<tree> python3 codec_aot.py save|load --path FILE
import argparse
import os
import time

import torch
import torch._dynamo as dynamo

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def build():
    began = time.perf_counter()
    decoder = load_qwen3_tts_tokenizer(MODEL, device="cuda:0", dtype="bfloat16", attn_implementation="sdpa", float32_encoder=False).model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=16, device=torch.device("cuda:0"), dtype=torch.bfloat16, in_place_frames=8)
    print(f"build {time.perf_counter() - began:.2f} s", flush=True)
    return incremental, arena


def inputs(arena, batch, width, seed):
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 1000, (batch, 16, width), generator=generator).cuda()
    return codes, arena.gather_by_index(torch.arange(batch, device="cuda:0"))


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
    parser.add_argument("phase", choices=["save", "load"])
    parser.add_argument("--path", required=True)
    parser.add_argument("--shape", default="8x8")
    args = parser.parse_args()
    incremental, arena = build()
    sizes = [(8, 8), (2, 8), (2, 2), (4, 4)]
    if args.phase == "save":
        batch, width = (int(value) for value in args.shape.split("x"))
        codes, state = inputs(arena, batch, width, 1)
        for tensor in (codes, state.frame_positions, state.slot_index, state.kv_rings.slot_rows):
            dynamo.maybe_mark_dynamic(tensor, 0)
        dynamo.maybe_mark_dynamic(codes, 2)
        compiled = torch.compile(incremental.decode_tensors, dynamic=False, fullgraph=True)
        began = time.perf_counter()
        function = compiled.aot_compile(((codes, state), {}))
        print(f"aot_compile {time.perf_counter() - began:.2f} s", flush=True)
        began = time.perf_counter()
        function.save_compiled_function(args.path)
        print(f"saved {time.perf_counter() - began:.2f} s, {os.path.getsize(args.path) / 1e6:.1f} MB", flush=True)
    else:
        began = time.perf_counter()
        with open(args.path, "rb") as handle:
            function = torch.compiler.load_compiled_function(handle)
        print(f"loaded {time.perf_counter() - began:.2f} s", flush=True)
    results = {}
    for batch, width in sizes:
        zero(arena)
        codes, state = inputs(arena, batch, width, 7)
        began = time.perf_counter()
        waveform = function(incremental, codes, state)
        torch.cuda.synchronize()
        first = time.perf_counter() - began
        results[(batch, width)] = waveform.float().cpu().clone()
        print(f"({batch}, {width}) first call {first * 1000:.1f} ms, rms {waveform.float().pow(2).mean().sqrt().item():.5f}", flush=True)
    if args.phase == "save":
        torch.save(results, args.path + ".waves")
    else:
        saved = torch.load(args.path + ".waves")
        print("equal to the save phase:", {key: torch.equal(saved[key], results[key]) for key in results}, flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
