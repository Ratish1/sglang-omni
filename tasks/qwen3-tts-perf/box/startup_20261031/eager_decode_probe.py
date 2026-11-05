"""One eager or launched stateful decode on the real codec, in its own process, to place the illegal address block 131 saw after the
window graphs were torn down. Prints one JSON line. Steps decode consecutive speech windows against one slot.

usage: cd TREE && PYTHONPATH=TREE python3 eager_decode_probe.py --state alive|torn --slot stream|reserved --frames 64 --path direct|launch [--pageable] [--steps 3]
"""

import argparse
import json
import traceback
from pathlib import Path

import soundfile
import torch

from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    IncrementalDecodeBatch,
    IncrementalDecodePlan,
    Qwen3TTSStreamingVocoderScheduler,
)

apply_qwen_tts_transformers_compatibility_patches()
from qwen_tts import Qwen3TTSTokenizer  # noqa: E402


def speech_codes(tokenizer) -> torch.Tensor:
    pieces = []
    for path in sorted((Path.cwd() / "docs/_static/audio").glob("*.wav")):
        samples, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        pieces.append(tokenizer.encode(samples.mean(axis=1), sr=sample_rate).audio_codes[0].cpu())
    return torch.cat(pieces, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=("alive", "torn"), required=True)
    parser.add_argument("--slot", choices=("stream", "reserved"), required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--path", choices=("direct", "launch"), required=True)
    parser.add_argument("--pageable", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    row = dict(vars(args))
    tokenizer = Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", device_map="cuda:0", dtype=torch.bfloat16,
                                                  attn_implementation="sdpa")
    scheduler = Qwen3TTSStreamingVocoderScheduler(
        tokenizer, device="cuda:0", tts_model_size="1b7", async_decode=True, enable_stateful_codec_decoder=True,
        incremental_codec_cuda_graph=True, incremental_codec_compile=True, fused_snake_activation=True, initial_max_batch_size=8,
        followup_max_batch_size=8, followup_worker_count=1, codec_state_slots=16, has_reference_prefixed_first_chunks=True,
    )
    scheduler.warmup_now()
    arena = scheduler.codec_arena
    window = scheduler.initial_window_decode_graphs
    row["ring_frames"] = arena.ring_frames
    row["in_place_frames"] = arena.ring_frames - scheduler.incremental_decoder.state_spec().retained_context
    row["window_enabled"] = window.enabled
    with torch.inference_mode():
        speech = speech_codes(tokenizer)
    done = 0
    try:
        with torch.inference_mode(), torch.cuda.stream(scheduler.decode_stream):
            if args.state == "torn":
                window.disable_runtime("probe")
            else:
                pass
            if args.pageable:
                scheduler.pinned_staging_disabled = True
            else:
                pass
            slot = arena.acquire() if args.slot == "stream" else list(arena.reserved_slots)[0]
            arena.zero_slot(slot)
            row["slot_index"] = slot
            position = 0
            for _ in range(args.steps):
                codes = speech[position : position + args.frames].T.unsqueeze(0)
                if args.path == "direct":
                    state = arena.gather([slot])
                    waveform = scheduler.incremental_decoder.decode(codes.to(scheduler.device), state)
                    arena.scatter([slot], state)
                    samples = waveform.float().cpu()
                else:
                    plan = IncrementalDecodePlan(
                        decoder_input=codes, slot=slot, fresh_frames=args.frames, reference_trim_frames=0,
                        generated_frames=position + args.frames, emitted_generated_frames=position,
                    )
                    handle = scheduler.launch_decode_plans(
                        [plan], stream=scheduler.decode_stream,
                        incremental=IncrementalDecodeBatch(decoder=scheduler.incremental_decoder, arena=arena, slots=[slot]),
                    )
                    samples = handle.resolve_partial()[0][0]
                row[f"step{done}_rms"] = float(samples.pow(2).mean().sqrt())
                position += args.frames
                done += 1
            scheduler.decode_stream.synchronize()
        row["outcome"] = "ok"
    except BaseException as exc:
        row["outcome"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        row["traceback"] = traceback.format_exc()[-1500:]
    row["steps_done"] = done
    print("PROBE " + json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
