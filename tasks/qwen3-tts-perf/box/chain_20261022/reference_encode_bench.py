"""Whether the Qwen3-TTS reference encoder's graphs should reach past 256 frames, on the tree on PYTHONPATH: the
trained speech tokenizer's encoder prepared as serving prepares it, the serving ladder captured and then the ladder
extended by --extra buckets (each in its own pool), the device memory each capture reserved; then per reference wav
(and the first reference cut to --cut frames) the eager encode as the request builder runs it on a miss against the
extended ladder's replay: wall time to the codes on the host side of a synchronize (median of 5), the eager encode's
device time from CUDA events and its kernel count, and the frames whose codes the two give equal.

usage: PYTHONPATH=<tree> python3 reference_encode_bench.py --card 3 --wavs <wav> [<wav> ...]
"""

import argparse
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import (
    DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES,
    Qwen3TTSReferenceEncoderCudaGraphRunner,
    move_conv_padding_to_host,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REPEATS = 5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--wavs", nargs="+", required=True)
    parser.add_argument("--extra", type=int, nargs="+", default=[384, 512, 768, 1024, 1152])
    parser.add_argument("--cut", type=int, nargs="+", default=[300, 500, 700, 900])
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    encoder = tokenizer.model.encoder
    move_conv_padding_to_host(encoder)
    hop = int(tokenizer.model.encode_downsample_rate)
    num_quantizers = int(tokenizer.model.encoder_valid_num_quantizers)
    dtype = next(encoder.parameters()).dtype
    stream = torch.cuda.Stream(device)
    runners = {}
    for name, ladder in (
        ("serving", DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES),
        ("extended", (*DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES, *args.extra)),
    ):
        torch.cuda.synchronize(device)
        reserved = torch.cuda.memory_reserved(device)
        runner = Qwen3TTSReferenceEncoderCudaGraphRunner(
            encoder, hop=hop, num_quantizers=num_quantizers, bucket_frames=ladder, stream=stream
        )
        runner.capture()
        torch.cuda.synchronize(device)
        print(
            f"{name} ladder {list(runner.graphs)}: reserved +{(torch.cuda.memory_reserved(device) - reserved) / 2**20:.0f} MiB"
            f"{'' if runner.disable_reason is None else ', ' + runner.disable_reason}"
        )
        runners[name] = runner
    extended = runners["extended"]

    def eager(samples: torch.Tensor) -> torch.Tensor:
        frames = -(-samples.numel() // hop)
        values = torch.zeros((1, 1, frames * hop), device=device, dtype=dtype)
        values[0, 0, : samples.numel()].copy_(samples)
        codes = encoder.encode(values, num_quantizers=num_quantizers, return_dict=True).audio_codes
        return codes[0, :, :frames].transpose(0, 1)

    def wall_ms(call, samples: torch.Tensor) -> tuple[float, torch.Tensor]:
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            codes = call(samples)
            stream.synchronize()
            times.append(1000 * (time.perf_counter() - start))
        return statistics.median(times), codes

    waveforms = [
        (path.rsplit("/", 1)[-1], torch.from_numpy(tokenizer._normalize_audio_inputs([path], sr=None)[0]))
        for path in args.wavs
    ]
    first_name, first = waveforms[0]
    waveforms += [(f"{first_name} cut to {frames}", first[: frames * hop]) for frames in args.cut]
    with torch.inference_mode(), torch.cuda.stream(stream):
        for name, samples in waveforms:
            frames = -(-samples.numel() // hop)
            eager_ms, eager_codes = wall_ms(eager, samples)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            eager(samples)
            end.record(stream)
            stream.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as profiler:
                eager(samples)
                stream.synchronize()
            kernels = sum(1 for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA)
            bucket = extended.bucket_for(frames)
            if bucket is None:
                print(f"{name}: {frames} frames, eager {eager_ms:.1f} ms wall, past the extended ladder")
                continue
            else:
                pass
            graph_ms, graph_codes = wall_ms(extended.encode, samples)
            equal = (graph_codes == eager_codes).all(dim=1)
            first_differing = int((~equal).nonzero()[0]) if not bool(equal.all()) else None
            print(
                f"{name}: {frames} frames, bucket {bucket}: eager {eager_ms:.1f} ms wall, {start.elapsed_time(end):.1f} ms "
                f"device, {kernels} kernels; graph {graph_ms:.1f} ms wall; codes equal on {int(equal.sum())} of {frames} "
                f"frames, first differing {first_differing}"
            )


if __name__ == "__main__":
    main()
