"""Replay time per captured width against one eager bootstrap decode.

Usage: python codec_window_bench.py <model_path> [--widths 4,8,16,32,64]
       [--totals 50,100,150] [--reps 50] [--out result.json]

Loads the Qwen3-TTS speech tokenizer on the current CUDA device, builds the
incremental decoder, an arena and one cold graph runner with the given window
widths, then measures, per width and batch bucket 1 and 4, the device time of
one replay and the host time of issuing it; for each total, the device and host
time of one eager decode of that width and of the window sequence the runner
plans for it. The capture footprint comes from the runner's stats.

The eager decode is what a reference prefixed bootstrap runs today. The window
sequence is what it runs once windows are enabled. The widths whose sequence
stays at or under the eager device time while cutting the host time are the
ones worth capturing.
"""

import argparse
import json
import statistics
import time

import torch

from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    Qwen3TTSIncrementalCodecCudaGraphRunner,
    plan_decode_windows,
)


def timed(fn, reps):
    """Device ms per call from CUDA events, host ms per call from the wall clock."""
    fn()
    torch.cuda.synchronize()
    device_ms = []
    host_ms = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        host_start = time.perf_counter()
        start.record()
        fn()
        end.record()
        host_ms.append((time.perf_counter() - host_start) * 1e3)
        end.synchronize()
        device_ms.append(start.elapsed_time(end))
    return {
        "device_ms_p50": statistics.median(device_ms),
        "host_ms_p50": statistics.median(host_ms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--widths", default="4,8,16,32,64")
    parser.add_argument("--totals", default="50,100,150")
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--buckets", default="1,4")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="capture every width from the compiled decoder step, as the warm "
        "runner does for its steady stride",
    )
    parser.add_argument("--out", default="codec_window_bench.json")
    args = parser.parse_args()
    widths = tuple(int(w) for w in args.widths.split(","))
    totals = tuple(int(t) for t in args.totals.split(","))
    buckets = tuple(int(b) for b in args.buckets.split(","))

    device = torch.device("cuda", torch.cuda.current_device())
    tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
        args.model_path, device=str(device), dtype="bfloat16", attn_implementation=None
    )
    decoder = Qwen3TTSIncrementalDecoder(tokenizer.model.decoder)
    config = getattr(tokenizer.model, "config", None)
    decoder_config = getattr(config, "decoder_config", config)
    num_quantizers = int(decoder_config.num_quantizers)
    dtype = next(tokenizer.model.decoder.parameters()).dtype
    arena = Qwen3TTSCodecStateArena(decoder, num_slots=8, device=device, dtype=dtype)
    all_widths = tuple(sorted({1, 2, *widths}))
    if args.compile:
        # note(ratish): the decoder compiles one shape per width and bucket
        # through a single function, and Dynamo refuses more than eight per
        # function by default; the warm runner never asks for more, this does.
        import torch._dynamo

        shapes = len(all_widths) * len(buckets) + 1
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, shapes
        )
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, shapes
        )
    # note(ratish): the same construction the vocoder uses for its window
    # runner, with every candidate cap's rungs captured at once.
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        decoder,
        device=device,
        dtype=dtype,
        num_quantizers=num_quantizers,
        mode="window",
        fresh_frames=all_widths,
        batch_sizes=buckets,
        compile_fresh_frames=all_widths if args.compile else (),
        arena=arena,
    )
    capture_start = time.perf_counter()
    runner.capture()
    stats = runner.stats()
    spec = decoder.state_spec()
    transformer = tokenizer.model.decoder.pre_transformer
    result = {
        "compiled": bool(args.compile),
        "buckets": list(buckets),
        "capture_s": time.perf_counter() - capture_start,
        "captured_keys": stats["build"]["captured_keys"],
        "graph_footprint_bytes": stats["memory"].get("graph_footprint_bytes"),
        "decoder": {
            "transformer_layers": spec.num_layers,
            "window_size": int(transformer.window_size),
            "retained_context": spec.retained_context,
            "conv_history_frames": [
                (key, length) for key, _, length in spec.conv_histories
            ],
        },
        "replay": {},
        "eager": {},
        "windowed": {},
    }
    if not stats["enabled"]:
        raise SystemExit(f"capture failed: {stats['disable_reason']}")
    print("decoder:", result["decoder"])
    print(
        f"captured {len(stats['build']['captured_keys'])} keys in "
        f"{result['capture_s']:.1f} s, footprint "
        f"{(result['graph_footprint_bytes'] or 0) / 2**20:.0f} MiB"
    )

    with torch.inference_mode():
        for width in all_widths:
            for bucket in buckets:
                codes = torch.randint(
                    0, 2048, (bucket, num_quantizers, width), device=device
                )
                slots = list(range(bucket))

                def replay(codes=codes, slots=slots):
                    if runner.decode_slots(codes, slots) is None:
                        raise RuntimeError(f"width {width} bucket {bucket} missed")

                result["replay"][f"{width}x{bucket}"] = timed(replay, args.reps)
                print(
                    f"replay width {width:3d} bucket {bucket}: {result['replay'][f'{width}x{bucket}']}"
                )

        for total in totals:
            codes = torch.randint(0, 2048, (1, num_quantizers, total), device=device)

            def eager(codes=codes):
                state = arena.gather([0])
                decoder.decode(codes, state)
                arena.scatter([0], state)

            result["eager"][total] = timed(eager, args.reps)
            print(f"eager total {total}: {result['eager'][total]}")

            for largest in widths:
                usable = tuple(w for w in all_widths if w <= largest)
                windows = plan_decode_windows(total, usable)
                if windows is None:
                    continue

                def windowed(codes=codes, windows=windows):
                    offset = 0
                    for width in windows:
                        if (
                            runner.decode_slots(
                                codes[:, :, offset : offset + width], [0]
                            )
                            is None
                        ):
                            raise RuntimeError(f"width {width} missed")
                        offset += width

                key = f"{total}/max{largest}"
                result["windowed"][key] = {
                    "windows": list(windows),
                    **timed(windowed, args.reps),
                }
                print(
                    f"windowed total {total} up to {largest}: {result['windowed'][key]}"
                )

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
