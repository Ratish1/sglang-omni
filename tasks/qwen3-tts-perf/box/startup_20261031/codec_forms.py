# Block 43 of 10-31: the incremental codec's compile forms (startup_20261031/PLAN.md item 1).
# Builds the vocoder exactly as serving does (create_vocoder_executor, which captures every runner), with the form
# chosen by SGLANG_OMNI_CODEC_COMPILE_FORM on the forms branch (main ignores it), then per compiled graph key:
# the device time per replay (CUDA events over --replays replays, best of two passes) and the waveform of one replay
# from a zeroed arena on seeded codes, one arena slot per row, saved for the identity check against main. Writes one
# JSON and one .pt.
# usage: python3 codec_forms.py --model M --out PREFIX [--no-autotune-pointwise]
import argparse
import json
import os
import time

began = time.perf_counter()

import torch
import torch._inductor.config
from torch._dynamo.utils import compile_times, counters

from sglang_omni.models.qwen3_tts import incremental_codec
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)
from sglang_omni.models.qwen3_tts.stages import create_vocoder_executor


def zero_arena(arena):
    storage = arena.storage
    storage.frame_positions.zero_()
    for pool in (*storage.conv_histories.values(), *storage.transconv_overlaps.values()):
        pool.zero_()
    for ring in (*arena.key_rings, *arena.value_rings):
        ring.zero_()


def replay_ms(graph, replays):
    best = None
    for _ in range(2):
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        value = start.elapsed_time(end) / replays
        best = value if best is None else min(best, value)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--out", required=True)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--no-autotune-pointwise", action="store_true")
    parser.add_argument("--warmups", type=int, default=None)
    args = parser.parse_args()
    if args.no_autotune_pointwise:
        torch._inductor.config.triton.autotune_pointwise = False
    else:
        pass
    if args.warmups is not None:
        Qwen3TTSIncrementalCodecCudaGraphRunner.WARMUP_ITERATIONS = args.warmups
    else:
        pass
    imported = time.perf_counter()
    captures = []
    original_capture = Qwen3TTSIncrementalCodecCudaGraphRunner.capture

    def timed_capture(self):
        start = time.perf_counter()
        original_capture(self)
        torch.cuda.synchronize()
        captures.append(
            {"mode": self.mode, "seconds": round(time.perf_counter() - start, 2), "graphs": len(self.graphs)}
        )

    Qwen3TTSIncrementalCodecCudaGraphRunner.capture = timed_capture
    precompiles = []
    original_precompile = incremental_codec.Qwen3TTSIncrementalDecoder.precompile

    def timed_precompile(self, codes, state):
        start = time.perf_counter()
        graphs_before = counters["stats"]["unique_graphs"]
        original_precompile(self, codes, state)
        torch.cuda.synchronize()
        precompiles.append(
            {
                "shape": [int(codes.shape[0]), int(codes.shape[-1])],
                "seconds": round(time.perf_counter() - start, 2),
                "new_graphs": counters["stats"]["unique_graphs"] - graphs_before,
            }
        )

    incremental_codec.Qwen3TTSIncrementalDecoder.precompile = timed_precompile
    build_start = time.perf_counter()
    scheduler = create_vocoder_executor(
        args.model, gpu_id=0, dtype="bfloat16", shares_engine_speech_tokenizer=True
    )
    built = time.perf_counter()
    decoder = scheduler.incremental_decoder
    compiled_widths = {shape[1] for shape in decoder.compiled_shapes}
    runners = [
        ("cold", scheduler.initial_incremental_decode_graphs),
        ("warm0", scheduler.followup_incremental_graph_holders[0]),
    ]
    generator = torch.Generator(device="cpu").manual_seed(20261031)
    timings = []
    waveforms = {}
    for name, runner in runners:
        for key in sorted(runner.graphs, key=lambda item: (item.fresh_frames, item.batch_bucket)):
            if key.fresh_frames not in compiled_widths:
                continue
            else:
                pass
            entry = runner.graphs[key]
            codes = torch.randint(0, 1000, tuple(entry.static_codes.shape), generator=generator)
            zero_arena(scheduler.codec_arena)
            entry.static_codes.copy_(codes)
            # note (ratish): one slot per row, as decode_slots fills it; rows sharing the scratch slot race on its ring rows.
            entry.static_index.copy_(torch.arange(key.batch_bucket))
            entry.graph.replay()
            torch.cuda.synchronize()
            label = f"{name}:{key.fresh_frames}x{key.batch_bucket}"
            waveforms[label] = entry.waveform.detach().float().cpu().clone()
            timings.append(
                {"runner": name, "frames": key.fresh_frames, "batch": key.batch_bucket, "ms": round(replay_ms(entry.graph, args.replays), 4)}
            )
    result = {
        "form": os.environ.get("SGLANG_OMNI_CODEC_COMPILE_FORM", "main"),
        "oblivious": os.environ.get("SGLANG_OMNI_CODEC_OBLIVIOUS", "0"),
        "autotune_pointwise": not args.no_autotune_pointwise,
        "import_s": round(imported - began, 2),
        "build_s": round(built - build_start, 2),
        "captures": captures,
        "precompile_total_s": round(sum(item["seconds"] for item in precompiles), 2),
        "precompiles": precompiles,
        "unique_graphs": counters["stats"]["unique_graphs"],
        "compiled_shapes": sorted(decoder.compiled_shapes),
        "replays": timings,
        "compile_times": compile_times(repr="str", aggregate=True),
    }
    with open(f"{args.out}.json", "w") as handle:
        json.dump(result, handle, indent=1)
    torch.save(waveforms, f"{args.out}.pt")
    print(json.dumps({key: result[key] for key in ("form", "oblivious", "build_s", "precompile_total_s", "unique_graphs")}))
    os._exit(0)


if __name__ == "__main__":
    main()
