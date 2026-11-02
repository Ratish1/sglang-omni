# Item 3 feasibility, every size class through aot_compile. Save: walk SHAPES, each shape takes the first program whose
# guards pass, else traces one with aot_compile and saves it; each shape's waveform from a zeroed arena; then the same
# shapes through the ordinary torch.compile the vocoder uses. Load (a fresh process): load every saved program, walk the
# same shapes, no trace may run; waveforms compared with the save phase's AOT and ordinary runs.
# usage: PYTHONPATH=<tree> python3 codec_aot_classes.py save|load --directory DIR
import argparse
import glob
import os
import time

import torch
import torch._dynamo as dynamo

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
SHAPES = [(2, 2), (2, 3), (2, 1), (1, 2), (1, 1), (8, 8), (8, 7), (8, 6), (8, 5), (8, 4), (8, 3), (8, 2), (8, 1), (4, 8), (4, 6), (4, 3), (4, 1), (2, 8), (2, 6), (1, 8), (1, 6), (1, 3)]


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
    state = arena.gather_by_index(torch.arange(batch, device="cuda:0"))
    for tensor in (codes, state.frame_positions, state.slot_index, state.kv_rings.slot_rows):
        dynamo.maybe_mark_dynamic(tensor, 0)
    dynamo.maybe_mark_dynamic(codes, 2)
    return codes, state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["save", "load"])
    parser.add_argument("--directory", required=True)
    args = parser.parse_args()
    os.makedirs(args.directory, exist_ok=True)
    decoder = load_qwen3_tts_tokenizer(MODEL, device="cuda:0", dtype="bfloat16", attn_implementation="sdpa", float32_encoder=False).model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=16, device=torch.device("cuda:0"), dtype=torch.bfloat16, in_place_frames=8)
    programs = []
    if args.phase == "load":
        began = time.perf_counter()
        for path in sorted(glob.glob(f"{args.directory}/program_*.aot")):
            with open(path, "rb") as handle:
                programs.append(torch.compiler.load_compiled_function(handle))
        print(f"loaded {len(programs)} programs in {time.perf_counter() - began:.2f} s", flush=True)
    else:
        pass
    aot_waves = {}
    for batch, width in SHAPES:
        codes, state = inputs(arena, batch, width)
        began = time.perf_counter()
        index = next((position for position, program in enumerate(programs) if program.guard_check(incremental, codes, state)), None)
        routed = time.perf_counter() - began
        traced = 0.0
        if index is None:
            assert args.phase == "save", (batch, width, "no saved program's guards pass")
            began = time.perf_counter()
            program = torch.compile(incremental.decode_tensors, dynamic=False, fullgraph=True).aot_compile(((codes, state), {}))
            traced = time.perf_counter() - began
            program.save_compiled_function(f"{args.directory}/program_{len(programs)}.aot")
            programs.append(program)
            index = len(programs) - 1
            zero(arena)
        else:
            pass
        began = time.perf_counter()
        waveform = programs[index](incremental, codes, state)
        torch.cuda.synchronize()
        print(f"({batch}, {width}) program {index} guards {routed * 1000:.2f} ms traced {traced:.2f} s first call {(time.perf_counter() - began) * 1000:.1f} ms", flush=True)
        aot_waves[(batch, width)] = waveform.float().cpu().clone()
    print(f"programs {len(programs)}", flush=True)
    if args.phase == "save":
        torch.save(aot_waves, f"{args.directory}/aot.waves")
        compiled = torch.compile(incremental.decode_tensors, dynamic=False, fullgraph=True)
        ordinary = {}
        for batch, width in SHAPES:
            codes, state = inputs(arena, batch, width)
            ordinary[(batch, width)] = compiled(codes, state).float().cpu().clone()
        torch.cuda.synchronize()
        torch.save(ordinary, f"{args.directory}/ordinary.waves")
        print("ordinary compile equal to AOT:", sum(torch.equal(ordinary[key], aot_waves[key]) for key in ordinary), "of", len(ordinary), flush=True)
    else:
        saved = torch.load(f"{args.directory}/aot.waves")
        ordinary = torch.load(f"{args.directory}/ordinary.waves")
        print("equal to the save phase:", sum(torch.equal(saved[key], aot_waves[key]) for key in saved), "of", len(saved), "; to the ordinary compile:", sum(torch.equal(ordinary[key], aot_waves[key]) for key in saved), flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)
