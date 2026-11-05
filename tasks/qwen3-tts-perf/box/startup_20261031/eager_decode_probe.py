"""One eager or launched stateful decode on the real codec, in its own process, to place the illegal address block 131 saw after the
window graphs were torn down. Prints one JSON line. Steps decode consecutive speech windows against one slot.

usage: cd TREE && PYTHONPATH=TREE python3 eager_decode_probe.py --state alive|torn|clear|capture_fail --slot stream|reserved --frames 64
       --path direct|launch|tail|warm [--pageable] [--steps 3] [--no-empty-cache] [--own-capture-streams]
torn disables the window runner, clear frees the cuBLAS workspaces with every graph alive, capture_fail makes the window capture fail partway.
"""

import argparse
import json
import traceback
from pathlib import Path

import soundfile
import torch

from sglang_omni.models.qwen3_tts.compat import apply_qwen_tts_transformers_compatibility_patches
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import IncrementalCodecGraphPool
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
    parser.add_argument("--state", choices=("alive", "torn", "clear", "capture_fail"), required=True)
    parser.add_argument("--no-empty-cache", action="store_true")
    parser.add_argument("--own-capture-streams", action="store_true")
    parser.add_argument("--slot", choices=("stream", "reserved"), required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--path", choices=("direct", "launch", "tail", "warm"), required=True)
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
    window = scheduler.initial_window_decode_graphs
    cold = scheduler.initial_incremental_decode_graphs
    tail = scheduler.initial_tail_decode_graphs
    if args.own_capture_streams:
        handle = window.graph_backend.graph_pool_handle()
        for runner in (cold, window, tail):
            runner.graph_pool = IncrementalCodecGraphPool(
                stream_priority=scheduler.decode_stream_priority, handle=handle,
                capture_stream=torch.cuda.Stream(device=scheduler.device, priority=scheduler.decode_stream_priority),
            )
    else:
        pass
    if args.state == "capture_fail":
        real_headroom = window.require_headroom
        headroom_calls = []

        def failing_headroom(free_bytes, *, key=None):
            headroom_calls.append(1)
            if len(headroom_calls) == 5:
                raise RuntimeError("injected capture failure")
            else:
                real_headroom(free_bytes, key=key)

        window.require_headroom = failing_headroom
    else:
        pass
    scheduler.warmup_now()
    if args.no_empty_cache:
        torch.cuda.empty_cache = lambda: None
    else:
        pass
    arena = scheduler.codec_arena
    row["footprint_mib"] = {
        runner.mode: runner.memory_stats.get("graph_footprint_bytes", 0) // 1048576 for runner in (cold, window, tail)
    }
    row["reserved_mib"] = torch.cuda.memory_reserved() // 1048576
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
            elif args.state == "clear":
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.empty_cache()
            else:
                pass
            if args.pageable:
                scheduler.pinned_staging_disabled = True
            else:
                pass
            row["window_enabled_after"] = window.enabled
            slot = arena.acquire() if args.slot == "stream" else list(arena.reserved_slots)[0]
            arena.zero_slot(slot)
            row["slot_index"] = slot
            position = 0
            for _ in range(args.steps):
                codes = speech[position : position + args.frames].T.unsqueeze(0)
                if args.path == "tail":
                    waveform = tail.decode_slots(codes.to(scheduler.device), [slot], valid_frames=[args.frames])
                    samples = waveform.float().cpu()
                elif args.path == "warm":
                    holder = scheduler.followup_incremental_graph_holders[0]
                    with torch.cuda.stream(scheduler.followup_decode_streams[0]):
                        waveform = holder.decode_slots(codes.to(scheduler.device), [slot])
                        samples = waveform.float().cpu()
                elif args.path == "direct":
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
