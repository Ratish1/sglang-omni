"""Triton GSan over the served Qwen3-TTS vocoder's slot conv stack on a codec state arena, one case per process, as
#3097's review found the in-place conv state exchanges racing on the repeated scratch slot. Cases: scratch (every row
the scratch slot, as capture warmup gathers), padded (one live slot, then the scratch slot, as a replay pads a short
cohort) and distinct (live slots only). Eager decodes of fixed random codes over the served ramp widths on the served
bf16 decoder, fused as served, gathered and scattered through the arena as a replay does. With --gsan the process runs
instrumented on its own card (python3 -m triton.tools.gsan pins its pool to device 0) and a race report aborts it;
without, it prints the live rows' waveform hash, so two trees are compared on the rows a request receives.

usage: PYTHONPATH=<tree> python3 pad_slot_gsan_probe.py CASE [--gsan] [--card 2]
"""

import argparse
import contextlib
import hashlib
import json

import torch
import triton
from triton.experimental.gsan._allocator import create_mem_pool

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
ROWS = 4
RAMP = (1, 2, 4, 8)


def decode_case(case: str, card: int) -> dict:
    device = torch.device("cuda", card)
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    decoder = tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(decoder)
    incremental = Qwen3TTSIncrementalDecoder(decoder)
    num_quantizers = int(tokenizer.model.encoder_valid_num_quantizers)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=ROWS, device=device, dtype=torch.bfloat16)
    live = [arena.acquire() for _ in range(ROWS)]
    scratch = arena.scratch_slot
    slots = {
        "scratch": [scratch] * ROWS,
        "padded": [live[0]] + [scratch] * (ROWS - 1),
        "distinct": live,
    }[case]
    index = torch.tensor(slots, dtype=torch.long, device=device)
    live_rows = [row for row, slot in enumerate(slots) if slot != scratch]
    generator = torch.Generator(device=device).manual_seed(0)
    digest = hashlib.sha256()
    with torch.inference_mode():
        for width in RAMP:
            codes = torch.randint(0, 1024, (ROWS, num_quantizers, width), device=device, generator=generator)
            state = arena.gather_by_index(index)
            pad_slot = vars(state).get("pad_slot")
            waveform = incremental.decode(codes, state)
            arena.scatter_by_index(index, state)
            digest.update(waveform[live_rows].float().cpu().numpy().tobytes())
    torch.cuda.synchronize(device)
    return {
        "case": case,
        "slots": slots,
        "state_pad_slot": pad_slot,
        "channels_last": incremental.channels_last_weights is not None,
        "live_rows_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=("scratch", "padded", "distinct"))
    parser.add_argument("--gsan", action="store_true")
    parser.add_argument("--card", type=int, default=2)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    if args.gsan:
        triton.knobs.compilation.instrumentation_mode = "gsan"
        context = torch.cuda.use_mem_pool(create_mem_pool())
    else:
        context = contextlib.nullcontext()
    with context:
        result = decode_case(args.case, args.card)
    print(json.dumps({"gsan": args.gsan, **result, "finished": True}), flush=True)


if __name__ == "__main__":
    main()
