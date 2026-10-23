"""How close a tree's served bf16 streaming vocoder stays to the fp32 decode of the same codes, per clip, so two trees
are compared on one truth. Real codes: each distinct long reference of a list encoded by the speech tokenizer as
serving encodes. Two streams per clip: a SeedTTS-like one (--short-reference frames of reference, its first chunk from
frame 0 in one decode_tail, as the tail graphs decode it) and a long one (all but --generated frames of reference, its
first chunk from frame 0 in windows of up to 64 frames, as the window runner decodes it); either way --generated frames
follow in the served ramp (1, 2, 4, then 8 a chunk). Every decode goes through the codec state arena (gather, decode,
scatter) on a bf16 decoder fused as served, so a channels-last tree runs its slot conv stack. Truth: the decoder's own
forward in fp32 over the same codes. Writes per clip the SNR in dB of each stream's emitted audio against the truth,
over the first frame and over the stream.

usage: PYTHONPATH=<tree> python3 vocoder_truth_probe.py run LIST OUT.json [--card 3] [--clips 62]
       python3 vocoder_truth_probe.py compare BASE.json CHANGE.json
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import soundfile
import torch

from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import split_frames_by_width
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WINDOW_WIDTHS = (1, 2, 4, 8, 16, 32, 64)
RAMP = (1, 2, 4)
STEADY_STRIDE = 8


def snr_db(actual: torch.Tensor, truth: torch.Tensor) -> float:
    noise = (actual.double() - truth.double()).pow(2).sum().item()
    signal = truth.double().pow(2).sum().item()
    return math.inf if noise == 0 else 10 * math.log10(signal / noise)


def run(args: argparse.Namespace) -> None:
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    truth_decoder = load_qwen3_tts_tokenizer(
        MODEL, device=str(device), dtype="float32", attn_implementation=None
    ).model.decoder.eval()
    served_tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    served_decoder = served_tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(served_decoder)
    incremental = Qwen3TTSIncrementalDecoder(served_decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=4, device=device, dtype=torch.bfloat16)
    encoder = served_tokenizer.model.encoder
    hop = int(served_tokenizer.model.encode_downsample_rate)
    num_quantizers = int(served_tokenizer.model.encoder_valid_num_quantizers)
    samples_per_frame = int(incremental.total_upsample)
    list_dir = Path(args.list).parent

    def stream(codes, reference_frames: int, first_in_windows: bool) -> torch.Tensor:
        slot = arena.acquire()
        audio = []
        if first_in_windows:
            offset = 0
            for width in split_frames_by_width(reference_frames + 1, WINDOW_WIDTHS):
                state = arena.gather([slot])
                waveform = incremental.decode(codes[..., offset : offset + width], state)
                arena.scatter([slot], state)
                offset += width
            audio.append(waveform.reshape(-1)[-samples_per_frame:])
        else:
            state = arena.gather([slot])
            waveform = incremental.decode_tail(
                codes[..., : reference_frames + 1],
                torch.tensor([reference_frames + 1], device=device),
                state,
                1,
            )
            arena.scatter([slot], state)
            audio.append(waveform.reshape(-1))
        position = reference_frames + 1
        widths = iter(RAMP[1:])
        while position < int(codes.shape[-1]):
            width = next(widths, STEADY_STRIDE)
            end = min(position + width, int(codes.shape[-1]))
            state = arena.gather([slot])
            audio.append(incremental.decode(codes[..., position:end], state).reshape(-1))
            arena.scatter([slot], state)
            position = end
        arena.release(slot)
        return torch.cat(audio).float()

    results = {}
    seen = set()
    with torch.inference_mode():
        for row in Path(args.list).read_text(encoding="utf-8").splitlines():
            row_id, _, wav_name, _ = row.split("|", 3)
            digest = hashlib.sha256((list_dir / wav_name).read_bytes()).hexdigest()
            if digest in seen or len(results) == args.clips:
                continue
            else:
                seen.add(digest)
            data, rate = soundfile.read(list_dir / wav_name, dtype="float32", always_2d=False)
            waveform = torch.from_numpy(served_tokenizer._normalize_audio_inputs([data], sr=rate)[0])  # noqa: leading-underscore
            frames = -(-waveform.numel() // hop)
            values = torch.zeros((1, 1, frames * hop), device=device, dtype=torch.bfloat16)
            values[0, 0, : waveform.numel()].copy_(waveform)
            codes = encoder.encode(values, num_quantizers=num_quantizers, return_dict=True).audio_codes[..., :frames]
            result = {}
            for name, total, reference_frames, first_in_windows in (
                ("short", args.short_reference + args.generated, args.short_reference, False),
                ("long", frames, frames - args.generated, True),
            ):
                clip = codes[..., :total]
                truth = truth_decoder(clip).reshape(-1)[reference_frames * samples_per_frame :]
                served = stream(clip, reference_frames, first_in_windows)
                result[f"{name} first"] = snr_db(served[:samples_per_frame], truth[:samples_per_frame])
                result[f"{name} stream"] = snr_db(served, truth)
            results[row_id] = result
            print(f"{row_id}: " + ", ".join(f"{key} {value:.2f}" for key, value in result.items()), flush=True)
    Path(args.out).write_text(json.dumps(results, indent=1))
    for key in next(iter(results.values())):
        values = [result[key] for result in results.values()]
        print(f"median {key}: {statistics.median(values):.2f} dB over {len(values)} clips")


def compare(args: argparse.Namespace) -> None:
    base, change = json.loads(Path(args.base).read_text()), json.loads(Path(args.change).read_text())
    clips = sorted(set(base) & set(change))
    for key in base[clips[0]]:
        deltas = [change[clip][key] - base[clip][key] for clip in clips]
        mean = statistics.fmean(deltas)
        error = statistics.stdev(deltas) / math.sqrt(len(deltas))
        print(
            f"{key}: base median {statistics.median(base[clip][key] for clip in clips):.2f}, change median "
            f"{statistics.median(change[clip][key] for clip in clips):.2f} dB; change minus base mean {mean:+.3f} dB "
            f"(se {error:.3f}, 95 % {mean - 1.96 * error:+.3f} to {mean + 1.96 * error:+.3f}), lower on "
            f"{sum(delta < 0 for delta in deltas)} of {len(deltas)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("list")
    run_parser.add_argument("out")
    run_parser.add_argument("--card", type=int, default=3)
    run_parser.add_argument("--clips", type=int, default=62)
    run_parser.add_argument("--short-reference", type=int, default=100)
    run_parser.add_argument("--generated", type=int, default=96)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("base")
    compare_parser.add_argument("change")
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
