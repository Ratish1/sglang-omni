# Block 76: the kernels of the codec graphs main already compiles, on the tree on PYTHONPATH. Builds the vocoder as
# serving does (create_vocoder_executor, as codec_forms.py), then per graph key of the cold runner, the width-8 keys of
# warm0 and window: the replay's device time (CUDA events, best of two passes of 200) and, under the torch profiler over
# 20 replays from a zeroed arena, every kernel in execution order with its grid, block and median duration. Writes one
# JSON.
# usage: PYTHONPATH=<tree> python3 codec_kernels.py --out PREFIX
import argparse
import json
import os
import statistics
import tempfile

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_tts.stages import create_vocoder_executor

REPLAYS_PROFILED = 20


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


def kernel_table(graph):
    with profile(activities=[ProfilerActivity.CUDA]) as profiler:
        for _ in range(REPLAYS_PROFILED):
            graph.replay()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        profiler.export_chrome_trace(handle.name)
        events = json.load(open(handle.name))["traceEvents"]
    kernels = sorted((event for event in events if event.get("cat") == "kernel"), key=lambda event: event["ts"])
    per_replay = len(kernels) // REPLAYS_PROFILED
    assert per_replay * REPLAYS_PROFILED == len(kernels), (len(kernels), REPLAYS_PROFILED)
    table = []
    for position in range(per_replay):
        samples = kernels[position::per_replay]
        names = {event["name"] for event in samples}
        assert len(names) == 1, (position, names)
        args = samples[0].get("args", {})
        table.append(
            {
                "name": samples[0]["name"],
                "grid": args.get("grid"),
                "block": args.get("block"),
                "shared": args.get("shared memory"),
                "registers": args.get("registers per thread"),
                "us": round(statistics.median(event["dur"] for event in samples), 3),
            }
        )
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    scheduler = create_vocoder_executor(args.model, gpu_id=0, dtype="bfloat16", shares_engine_speech_tokenizer=True)
    runners = [
        ("cold", scheduler.initial_incremental_decode_graphs, None),
        ("warm0", scheduler.followup_incremental_graph_holders[0], 8),
        ("window", scheduler.initial_window_decode_graphs, 8),
    ]
    generator = torch.Generator(device="cpu").manual_seed(20261102)
    result = {"compiled_shapes": sorted(scheduler.incremental_decoder.compiled_shapes), "keys": {}}
    for name, runner, only_width in runners:
        for key in sorted(runner.graphs, key=lambda item: (item.fresh_frames, item.batch_bucket)):
            if only_width is not None and key.fresh_frames != only_width:
                continue
            else:
                pass
            entry = runner.graphs[key]
            entry.static_codes.copy_(torch.randint(0, 1000, tuple(entry.static_codes.shape), generator=generator))
            entry.static_index.copy_(torch.arange(key.batch_bucket))
            zero_arena(scheduler.codec_arena)
            milliseconds = replay_ms(entry.graph, 200)
            zero_arena(scheduler.codec_arena)
            result["keys"][f"{name}:{key.fresh_frames}x{key.batch_bucket}"] = {
                "ms": round(milliseconds, 4),
                "kernels": kernel_table(entry.graph),
            }
            print(f"{name}:{key.fresh_frames}x{key.batch_bucket} {milliseconds:.4f} ms", flush=True)
    with open(f"{args.out}.json", "w") as handle:
        json.dump(result, handle)
    os._exit(0)


if __name__ == "__main__":
    main()
