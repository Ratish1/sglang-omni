# The launches of one compiled decode on the tree on PYTHONPATH: one program traced at --trace (a shape) with the size
# marks the vocoder uses, then under the torch profiler each --shapes decode, the kernels in order with grid and block.
# Prints each shape's launch count, the Triton launch count, the kernels that repeat a name stem, and writes a JSON.
# usage: PYTHONPATH=<tree> python3 codec_launches.py --trace 8x8 --shapes 8x8,2x8,2x2 --out FILE
import argparse
import collections
import json
import os
import re
import tempfile

import torch
import torch._dynamo as dynamo
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def zero(arena):
    storage = arena.storage
    storage.frame_positions.zero_()
    for pool in (*storage.conv_histories.values(), *storage.transconv_overlaps.values()):
        pool.zero_()
    for ring in (*arena.key_rings, *arena.value_rings):
        ring.zero_()


def inputs(arena, batch, width):
    zero(arena)
    generator = torch.Generator().manual_seed(batch * 100 + width)
    codes = torch.randint(0, 1000, (batch, 16, width), generator=generator).cuda()
    return codes, arena.gather_by_index(torch.arange(batch, device="cuda:0"))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="8x8")
    parser.add_argument("--shapes", default="8x8,2x8,2x2")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    decoder = load_qwen3_tts_tokenizer(MODEL, device="cuda:0", dtype="bfloat16", attn_implementation="sdpa", float32_encoder=False).model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=16, device=torch.device("cuda:0"), dtype=torch.bfloat16, in_place_frames=8)
    batch, width = (int(value) for value in args.trace.split("x"))
    codes, state = inputs(arena, batch, width)
    for tensor in (codes, state.frame_positions, state.slot_index, state.kv_rings.slot_rows):
        dynamo.maybe_mark_dynamic(tensor, 0)
    dynamo.maybe_mark_dynamic(codes, 2)
    program = torch.compile(incremental.decode_tensors, dynamic=False, fullgraph=True).aot_compile(((codes, state), {}))
    program(incremental, codes, state)
    result = {}
    for shape in args.shapes.split(","):
        batch, width = (int(value) for value in shape.split("x"))
        for _ in range(3):
            codes, state = inputs(arena, batch, width)
            program(incremental, codes, state)
        torch.cuda.synchronize()
        codes, state = inputs(arena, batch, width)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as profiler:
            waveform = program(incremental, codes, state)
            torch.cuda.synchronize()
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            profiler.export_chrome_trace(handle.name)
            events = json.load(open(handle.name))["traceEvents"]
        kernels = sorted((event for event in events if event.get("cat") == "kernel"), key=lambda event: event["ts"])
        table = [{"name": event["name"], "grid": event["args"].get("grid"), "block": event["args"].get("block"), "us": event["dur"]} for event in kernels]
        result[shape] = {"kernels": table, "rms": float(waveform.float().pow(2).mean().sqrt())}
        triton = [item for item in table if item["name"].startswith("triton_")]
        stems = collections.Counter(re.sub(r"_\d+$", "", item["name"])[:90] for item in triton)
        print(f"{shape}: {len(table)} launches, {len(triton)} Triton, Triton us {sum(item['us'] for item in triton):.1f}, all us {sum(item['us'] for item in table):.1f}, rms {result[shape]['rms']:.5f}", flush=True)
        for stem, count in stems.most_common(6):
            print(f"    {count} x {stem}", flush=True)
    with open(args.out, "w") as handle:
        json.dump(result, handle)
    os._exit(0)


if __name__ == "__main__":
    main()
