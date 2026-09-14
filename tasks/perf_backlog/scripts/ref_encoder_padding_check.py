"""Reference encoder: where bucket padded codes differ from the unpadded codes.

Usage: python ref_encoder_padding_check.py <model_path> --meta zhaochenyang20/seed-tts-eval-arrow
       [--lang en] [--samples 64] [--bucket-frames 48,64,96,128,192]
       [--out ref_encoder_padding_check.json]

Doc 40: graphs captured at bucket lengths reproduced 63 of 64 references with code
differences from quantizer 0 on. This separates the causes without graphs. For each
reference and each dtype (bfloat16 as the vocoder loads the tokenizer, float32 as the
checkpoint stores the encoder) it encodes eagerly with host side padding buffers and
num_quantizers=16 as in doc 39:

  plain     the waveform as is, today's path
  again     plain a second time, determinism
  aligned   zero padded to the next multiple of 1920 samples, so every conv layer's
            extra padding is zero
  bucket    zero padded to the smallest bucket that fits

and reports, per variant against plain, the references that match exactly and, for
the rest, the differing codes split into the last two frames and the frames before
them, with the first differing quantizer. It also compares the float32 plain codes to
the bfloat16 plain codes, and in float32 captures one graph per bucket and checks the
replayed codes against the eager bucket run, with the replay time.
"""

import argparse
import json
import statistics
import time

import torch
from transformers.models.mimi.modeling_mimi import MimiConv1d

from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts import stages as qwen3_stages

VALID_QUANTIZERS = 16


def move_padding_buffers_to_host(encoder):
    for module in encoder.modules():
        if isinstance(module, MimiConv1d):
            module.stride = module.stride.cpu()
            module.kernel_size = module.kernel_size.cpu()
            module.padding_total = module.padding_total.cpu()
            module.padding_right = module.padding_total // 2
            module.padding_left = module.padding_total - module.padding_right


def encode(encoder, values):
    """values (B, 1, T) in the encoder's dtype; codes (B, 16, frames)."""
    return encoder.encode(
        values, num_quantizers=VALID_QUANTIZERS, return_dict=True
    ).audio_codes


def padded(waveform, length, dtype):
    values = torch.zeros((1, 1, length), device=waveform.device, dtype=dtype)
    values[0, 0, : waveform.numel()].copy_(waveform.to(dtype=dtype))
    return values


def compare(want, got):
    """want, got (16, frames). None if equal, else the split of differing codes."""
    diff = want != got
    if not bool(diff.any()):
        return None
    frames = want.shape[1]
    tail = int(diff[:, max(0, frames - 2) :].sum())
    return {
        "positions": int(diff.sum()),
        "tail_two_frames": tail,
        "earlier_frames": int(diff.sum()) - tail,
        "first_quantizer": int(diff.nonzero()[:, 0].min()),
    }


def summarize(pairs):
    """pairs: list of compare() results, one per reference."""
    misses = [p for p in pairs if p is not None]
    return {
        "exact": len(pairs) - len(misses),
        "mismatched": len(misses),
        "positions_p50": (
            statistics.median(p["positions"] for p in misses) if misses else 0
        ),
        "tail_only": sum(1 for p in misses if p["earlier_frames"] == 0),
        "first_quantizer_0": sum(1 for p in misses if p["first_quantizer"] == 0),
        "examples": misses[:6],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--bucket-frames", default="48,64,96,128,192")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out", default="ref_encoder_padding_check.json")
    args = parser.parse_args()
    bucket_frames = tuple(sorted(int(f) for f in args.bucket_frames.split(",")))

    device = torch.device("cuda", torch.cuda.current_device())
    encoders = {}
    for name in ("bfloat16", "float32"):
        tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
            args.model_path, device=str(device), dtype=name, attn_implementation=None
        )
        move_padding_buffers_to_host(tokenizer.model.encoder)
        encoders[name] = tokenizer.model.encoder
    sample_rate = int(tokenizer.feature_extractor.sampling_rate)
    hop = int(tokenizer.model.encode_downsample_rate)

    seen = set()
    waveforms = []
    for sample in load_seedtts_samples(args.meta, split=args.lang):
        if sample.ref_audio in seen:
            continue
        seen.add(sample.ref_audio)
        audio = tokenizer.load_audio(sample.ref_audio, target_sr=sample_rate)
        waveforms.append(torch.from_numpy(audio).to(device))
        if len(waveforms) >= args.samples:
            break
    frames = [-(-w.numel() // hop) for w in waveforms]
    buckets = [min(b for b in bucket_frames if b >= f) for f in frames]
    result = {"samples": len(waveforms), "bucket_frames": list(bucket_frames)}
    plain_by_dtype = {}

    with torch.inference_mode():
        for name, encoder in encoders.items():
            dtype = getattr(torch, name)
            codes = {"plain": [], "again": [], "aligned": [], "bucket": []}
            for waveform, f, bucket in zip(waveforms, frames, buckets):
                n = waveform.numel()
                codes["plain"].append(
                    encode(encoder, padded(waveform, n, dtype))[0, :, :f]
                )
                codes["again"].append(
                    encode(encoder, padded(waveform, n, dtype))[0, :, :f]
                )
                codes["aligned"].append(
                    encode(encoder, padded(waveform, f * hop, dtype))[0, :, :f]
                )
                codes["bucket"].append(
                    encode(encoder, padded(waveform, bucket * hop, dtype))[0, :, :f]
                )
            plain_by_dtype[name] = codes["plain"]
            entry = {}
            for variant in ("again", "aligned", "bucket"):
                entry[variant] = summarize(
                    [compare(a, b) for a, b in zip(codes["plain"], codes[variant])]
                )
                print(f"{name} {variant}: {entry[variant]}")
            started = time.perf_counter()
            for _ in range(args.reps):
                encode(encoder, padded(waveforms[0], waveforms[0].numel(), dtype))
            torch.cuda.synchronize(device)
            entry["plain_host_ms"] = (time.perf_counter() - started) * 1e3 / args.reps
            result[name] = entry

        result["float32_vs_bfloat16_plain"] = summarize(
            [
                compare(a, b)
                for a, b in zip(plain_by_dtype["bfloat16"], plain_by_dtype["float32"])
            ]
        )
        print("float32 vs bfloat16 plain:", result["float32_vs_bfloat16_plain"])

        encoder = encoders["float32"]
        graphs = {}
        result["float32_graph"] = {}
        for bucket in bucket_frames:
            static_in = torch.zeros(
                (1, 1, bucket * hop), device=device, dtype=torch.float32
            )
            stream = torch.cuda.Stream(device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    encode(encoder, static_in)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = encode(encoder, static_in)
            torch.cuda.synchronize(device)
            graphs[bucket] = (static_in, graph, static_out)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            graph.replay()
            start.record()
            for _ in range(args.reps):
                graph.replay()
            end.record()
            end.synchronize()
            result["float32_graph"][str(bucket)] = {
                "replay_device_ms": start.elapsed_time(end) / args.reps
            }
        pairs = []
        for waveform, f, bucket in zip(waveforms, frames, buckets):
            static_in, graph, static_out = graphs[bucket]
            static_in.zero_()
            static_in[0, 0, : waveform.numel()].copy_(waveform.to(dtype=torch.float32))
            graph.replay()
            torch.cuda.synchronize(device)
            want = encode(encoder, padded(waveform, bucket * hop, torch.float32))[
                0, :, :f
            ]
            pairs.append(compare(want, static_out[0, :, :f]))
        result["float32_graph"]["against_eager_bucket"] = summarize(pairs)
        print(
            "float32 graph against eager bucket:",
            result["float32_graph"]["against_eager_bucket"],
        )
        print(
            "float32 graph replay ms:",
            {
                k: v
                for k, v in result["float32_graph"].items()
                if k != "against_eager_bucket"
            },
        )

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
