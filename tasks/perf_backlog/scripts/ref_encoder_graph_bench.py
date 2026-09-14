"""Reference encoder: eager against captured graphs on real references.

Usage: python ref_encoder_graph_bench.py <model_path> --meta zhaochenyang20/seed-tts-eval-arrow
       [--lang en] [--samples 64] [--bucket-frames 48,64,96,128,192] [--batches 1,2]
       [--reps 20] [--out ref_encoder_graph_bench.json]

Loads the Qwen3-TTS speech tokenizer the way the vocoder stage does, takes the first
distinct reference files of the corpus, and runs four encodes of each:

  wrapper   tokenizer.encode(...), what _Qwen3TTSRefCodeBatcher runs today
  host_pad  the same after the conv padding buffers move to the CPU
  q16       the encoder called directly with num_quantizers=16
  graph     a captured graph at the smallest bucket the reference fits, codes sliced
            to the reference's frames

host_pad and q16 must match the wrapper bit for bit. graph is compared to q16 per
sample: exact matches, and for the rest the mismatching positions and their first
quantizer index. Timing is host ms per call (wall) and device ms (events), median
over reps, plus the capture footprint per key.
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


def move_padding_buffers_to_host(encoder):
    """The conv padding arithmetic runs on these; on the device it costs a sync per layer."""
    count = 0
    for module in encoder.modules():
        if isinstance(module, MimiConv1d):
            module.stride = module.stride.cpu()
            module.kernel_size = module.kernel_size.cpu()
            module.padding_total = module.padding_total.cpu()
            module.padding_right = module.padding_total // 2
            module.padding_left = module.padding_total - module.padding_right
            count += 1
    return count


def encode_direct(encoder, waveform, dtype):
    """One waveform through MimiModel.encode with the valid quantizers only."""
    values = waveform.to(dtype=dtype).view(1, 1, -1)
    return encoder.encode(
        values, num_quantizers=VALID_QUANTIZERS, return_dict=True
    ).audio_codes[0]


class BucketGraph:
    def __init__(self, encoder, batch, length, device, dtype):
        self.batch = batch
        self.length = length
        self.static_in = torch.zeros((batch, 1, length), device=device, dtype=dtype)
        before = torch.cuda.memory_allocated(device)
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(2):
                encoder.encode(
                    self.static_in, num_quantizers=VALID_QUANTIZERS, return_dict=True
                )
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_out = encoder.encode(
                self.static_in, num_quantizers=VALID_QUANTIZERS, return_dict=True
            ).audio_codes
        torch.cuda.synchronize(device)
        self.footprint_bytes = torch.cuda.memory_allocated(device) - before

    def replay(self, waveforms, dtype):
        self.static_in.zero_()
        for row, waveform in enumerate(waveforms):
            self.static_in[row, 0, : waveform.numel()].copy_(waveform.to(dtype=dtype))
        self.graph.replay()
        return self.static_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--meta", default="zhaochenyang20/seed-tts-eval-arrow")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--bucket-frames", default="48,64,96,128,192")
    parser.add_argument("--batches", default="1,2")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--out", default="ref_encoder_graph_bench.json")
    args = parser.parse_args()
    bucket_frames = tuple(sorted(int(f) for f in args.bucket_frames.split(",")))
    batches = tuple(sorted(int(b) for b in args.batches.split(",")))

    device = torch.device("cuda", torch.cuda.current_device())
    tokenizer = qwen3_stages._load_qwen3_tts_tokenizer(
        args.model_path, device=str(device), dtype="bfloat16", attn_implementation=None
    )
    encoder = tokenizer.model.encoder
    dtype = next(encoder.parameters()).dtype
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
    result = {
        "samples": len(waveforms),
        "dtype": str(dtype),
        "frames_quantiles": {
            "min": min(frames),
            "p50": statistics.median(frames),
            "p90": sorted(frames)[int(0.9 * (len(frames) - 1))],
            "max": max(frames),
        },
        "bucket_frames": list(bucket_frames),
        "batches": list(batches),
        "wrapper": {},
        "host_pad": {},
        "q16": {},
        "graph": {},
    }
    print(f"{len(waveforms)} references, frames {result['frames_quantiles']}")

    with torch.inference_mode():
        wrapper_codes = [
            tokenizer.encode([w.cpu().numpy()], sr=sample_rate)
            .audio_codes[0]
            .transpose(0, 1)
            for w in waveforms
        ]
        result["wrapper"] = timed(
            lambda: tokenizer.encode([waveforms[0].cpu().numpy()], sr=sample_rate),
            args.reps,
        )
        print("wrapper:", result["wrapper"])

        moved = move_padding_buffers_to_host(encoder)
        host_pad_codes = [
            tokenizer.encode([w.cpu().numpy()], sr=sample_rate)
            .audio_codes[0]
            .transpose(0, 1)
            for w in waveforms
        ]
        result["host_pad"] = {
            "conv_layers_moved": moved,
            "exact": all(
                torch.equal(a, b) for a, b in zip(wrapper_codes, host_pad_codes)
            ),
            **timed(
                lambda: tokenizer.encode([waveforms[0].cpu().numpy()], sr=sample_rate),
                args.reps,
            ),
        }
        print("host_pad:", result["host_pad"])

        q16_codes = [
            encode_direct(encoder, w, dtype)[:, :f] for w, f in zip(waveforms, frames)
        ]
        result["q16"] = {
            "exact": all(torch.equal(a, b) for a, b in zip(wrapper_codes, q16_codes)),
            **timed(lambda: encode_direct(encoder, waveforms[0], dtype), args.reps),
        }
        print("q16:", result["q16"])

        graphs = {}
        for batch in batches:
            for bucket in bucket_frames:
                key = f"{bucket}x{batch}"
                graph = BucketGraph(encoder, batch, bucket * hop, device, dtype)
                graphs[(batch, bucket)] = graph
                sample = waveforms[0][: bucket * hop]
                entry = {
                    "footprint_mib": graph.footprint_bytes / 2**20,
                    **timed(
                        lambda g=graph, s=sample, b=batch: g.replay([s] * b, dtype),
                        args.reps,
                    ),
                }
                result["graph"][key] = entry
                print(f"graph {key}: {entry}")

        for batch in batches:
            exact = 0
            mismatches = []
            unbucketed = 0
            for index in range(0, len(waveforms), batch):
                group = waveforms[index : index + batch]
                need = max(-(-w.numel() // hop) for w in group)
                fit = [b for b in bucket_frames if b >= need]
                if not fit:
                    unbucketed += len(group)
                    continue
                graph = graphs[(batch, fit[0])]
                out = graph.replay(group, dtype)
                for row, waveform in enumerate(group):
                    f = -(-waveform.numel() // hop)
                    got = out[row, :, :f]
                    want = q16_codes[index + row]
                    if torch.equal(got, want):
                        exact += 1
                        continue
                    diff = (got != want).nonzero()
                    mismatches.append(
                        {
                            "sample": index + row,
                            "frames": f,
                            "bucket": fit[0],
                            "positions": int(diff.shape[0]),
                            "first_quantizer": int(diff[:, 0].min()),
                        }
                    )
            result["graph"][f"batch{batch}_check"] = {
                "exact": exact,
                "mismatched": len(mismatches),
                "unbucketed": unbucketed,
                "mismatches": mismatches[:16],
            }
            print(
                f"batch {batch}: exact {exact}, mismatched {len(mismatches)}, unbucketed {unbucketed}"
            )

    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
