"""Which bf16 first decode of a long reference sits closer to the fp32 decode of the same codes: main's chained
windows from frame 0 (widths up to 64, as the window runner splits), or D1's one tail decode starting
first_chunk_reference_frames(1) before the emitted frame. Real codes: each long reference of a list encoded by the
speech tokenizer as serving does; its last --generated frames stand in for the generated stream, the rest is the
reference. Each bf16 path emits the first frame, then both continue with the same eager follow-ups of --stride
frames; the fp32 truth is the decoder's own whole forward over the same codes (in fp32, as served for non-streaming).
Per reference: SNR in dB of each path's emitted audio against the truth, over the first frame and over the whole
stream, and of D1 against the windows.

usage: PYTHONPATH=<D1 tree> python3 first_chunk_truth_probe.py LIST [--card 3] [--references 8] [--generated 96]
"""

import argparse
import math
import statistics
from pathlib import Path

import soundfile
import torch

from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import split_frames_by_width
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
WINDOW_WIDTHS = (1, 2, 4, 8, 16, 32, 64)


def snr_db(actual: torch.Tensor, truth: torch.Tensor) -> float:
    noise = (actual.double() - truth.double()).pow(2).sum().item()
    signal = truth.double().pow(2).sum().item()
    return math.inf if noise == 0 else 10 * math.log10(signal / noise)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("list")
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--references", type=int, default=8)
    parser.add_argument("--generated", type=int, default=96)
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    truth_tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="float32", attn_implementation=None)
    served_tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    truth_decoder = truth_tokenizer.model.decoder.eval()
    served_decoder = served_tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(served_decoder)
    incremental = Qwen3TTSIncrementalDecoder(served_decoder)
    encoder = served_tokenizer.model.encoder
    hop = int(served_tokenizer.model.encode_downsample_rate)
    num_quantizers = int(served_tokenizer.model.encoder_valid_num_quantizers)
    samples_per_frame = int(incremental.total_upsample)
    reach = incremental.first_chunk_reference_frames(1)
    list_dir = Path(args.list).parent
    rows = Path(args.list).read_text(encoding="utf-8").splitlines()[: args.references]
    results = []
    with torch.inference_mode():
        for row in rows:
            row_id, _, wav_name, _ = row.split("|", 3)
            data, rate = soundfile.read(list_dir / wav_name, dtype="float32", always_2d=False)
            audio = served_tokenizer._normalize_audio_inputs([data], sr=rate)  # noqa: leading-underscore
            waveform = torch.from_numpy(audio[0])
            frames = -(-waveform.numel() // hop)
            values = torch.zeros((1, 1, frames * hop), device=device, dtype=torch.bfloat16)
            values[0, 0, : waveform.numel()].copy_(waveform)
            codes = encoder.encode(values, num_quantizers=num_quantizers, return_dict=True).audio_codes[..., :frames]
            reference_frames = frames - args.generated
            truth = truth_decoder(codes).reshape(-1)[reference_frames * samples_per_frame :]

            windows_state = incremental.init_state(1, device=device, dtype=torch.bfloat16)
            offset = 0
            for width in split_frames_by_width(reference_frames + 1, WINDOW_WIDTHS):
                windows_first = incremental.decode(codes[..., offset : offset + width], windows_state)
                offset += width
            windows_audio = [windows_first.reshape(-1)[-samples_per_frame:]]

            start = reference_frames - reach
            capped_state = incremental.init_state(1, device=device, dtype=torch.bfloat16)
            capped_state.frame_positions.fill_(start)
            width = reference_frames + 1 - start
            capped_first = incremental.decode_tail(
                codes[..., start : reference_frames + 1],
                torch.tensor([width], device=device),
                capped_state,
                1,
            )
            capped_audio = [capped_first.reshape(-1)]

            position = reference_frames + 1
            while position < frames:
                end = min(position + args.stride, frames)
                windows_audio.append(incremental.decode(codes[..., position:end], windows_state).reshape(-1))
                capped_audio.append(incremental.decode(codes[..., position:end], capped_state).reshape(-1))
                position = end
            windows_stream = torch.cat(windows_audio).float()
            capped_stream = torch.cat(capped_audio).float()
            first = slice(0, samples_per_frame)
            result = {
                "windows first": snr_db(windows_stream[first], truth[first]),
                "d1 first": snr_db(capped_stream[first], truth[first]),
                "windows stream": snr_db(windows_stream, truth),
                "d1 stream": snr_db(capped_stream, truth),
                "d1 against windows": snr_db(capped_stream, windows_stream),
            }
            results.append(result)
            print(f"{row_id} ({frames} frames, start {start}): " + ", ".join(f"{k} {v:.1f} dB" for k, v in result.items()), flush=True)
    for key in results[0]:
        values = [result[key] for result in results]
        print(f"median {key}: {statistics.median(values):.1f} dB (min {min(values):.1f}, max {max(values):.1f})")


if __name__ == "__main__":
    main()
