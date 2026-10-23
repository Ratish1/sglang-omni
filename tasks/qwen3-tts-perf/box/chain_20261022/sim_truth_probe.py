"""Whether speaker similarity moves with a vocoder's distance to the fp32 decoder: every SeedTTS EN reference clip
encoded by the speech tokenizer as served, its first quarter (at least --min-reference frames) taken as the reference
and the rest streamed through the tree's served graph runners (tail graph for the first chunk, warm graphs in the
served ramp, width 8 compiled), and with --truth the same codes through the fp32 decoder's own forward. Writes each
stream as a 24 kHz wav under OUT_DIR/<arm>/audio/ with a generated.json the SeedTTS scorer reads, so
benchmark_tts_seedtts --similarity-only scores every arm against the clip it came from.

usage: PYTHONPATH=<tree> python3 sim_truth_probe.py run OUT_DIR ARM [--card 3] [--truth]
       python3 sim_truth_probe.py compare OUT_DIR BASE_ARM ARM [ARM ...]   (after each arm's similarity pass)
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import soundfile
import torch

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.codec_state_arena import Qwen3TTSCodecStateArena
from sglang_omni.models.qwen3_tts.incremental_codec import Qwen3TTSIncrementalDecoder
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphPool,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer
from sglang_omni.models.qwen3_tts.streaming_vocoder import QWEN3_TTS_TAIL_GRAPH_FRAMES
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
RAMP = (1, 2, 4)
STEADY_STRIDE = 8
SAMPLE_RATE = 24000


def write_arm(out_dir: Path, arm: str, audio: dict[str, torch.Tensor]) -> None:
    (out_dir / arm / "audio").mkdir(parents=True, exist_ok=True)
    generated = []
    for sample_id, samples in audio.items():
        wav_path = out_dir / arm / "audio" / f"{sample_id}.wav"
        soundfile.write(wav_path, samples.numpy(), SAMPLE_RATE, subtype="FLOAT")
        generated.append({"sample_id": sample_id, "wav_path": str(wav_path), "is_success": True})
    (out_dir / arm / "generated.json").write_text(json.dumps(generated))


def compare(out_dir: Path, arms: list[str]) -> None:
    similarity = {
        arm: {
            row["id"]: float(row["speaker_similarity"])
            for row in json.loads((out_dir / arm / "similarity_results.json").read_text())["per_sample"]
            if row["is_success"]
        }
        for arm in ["truth", *arms]
    }
    pairs = [("truth", arm) for arm in arms] + [(arms[0], arm) for arm in arms[1:]]
    for base, change in pairs:
        clips = sorted(set(similarity[base]) & set(similarity[change]))
        deltas = [similarity[change][clip] - similarity[base][clip] for clip in clips]
        mean = statistics.fmean(deltas)
        error = statistics.stdev(deltas) / math.sqrt(len(deltas))
        print(
            f"{change} minus {base}: SIM {statistics.fmean(similarity[base][clip] for clip in clips):.4f} -> "
            f"{statistics.fmean(similarity[change][clip] for clip in clips):.4f}, paired {mean:+.4f} "
            f"(se {error:.4f}, 95 % {mean - 1.96 * error:+.4f} to {mean + 1.96 * error:+.4f}) over {len(clips)} clips"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("out_dir")
    run_parser.add_argument("arm")
    run_parser.add_argument("--card", type=int, default=3)
    run_parser.add_argument("--truth", action="store_true")
    run_parser.add_argument("--min-reference", type=int, default=12)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("out_dir")
    compare_parser.add_argument("arms", nargs="+")
    args = parser.parse_args()
    if args.command == "compare":
        compare(Path(args.out_dir), args.arms)
        return
    else:
        pass
    out_dir = Path(args.out_dir)
    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    served_tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    truth_decoder = (
        load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="float32", attn_implementation=None).model.decoder.eval()
        if args.truth
        else None
    )
    served_decoder = served_tokenizer.model.decoder.eval()
    fuse_vocoder_decoder(served_decoder)
    incremental = Qwen3TTSIncrementalDecoder(served_decoder)
    arena = Qwen3TTSCodecStateArena(incremental, num_slots=4, device=device, dtype=torch.bfloat16)
    num_quantizers = int(served_tokenizer.model.encoder_valid_num_quantizers)

    def runner(mode: str, fresh_frames: tuple[int, ...], compiled: tuple[int, ...], emit_frames: int = 0):
        built = Qwen3TTSIncrementalCodecCudaGraphRunner(
            incremental,
            device=device,
            dtype=torch.bfloat16,
            num_quantizers=num_quantizers,
            mode=mode,
            fresh_frames=fresh_frames,
            batch_sizes=(1,),
            min_free_gb=0.0,
            compile_fresh_frames=compiled,
            arena=arena,
            graph_pool=IncrementalCodecGraphPool(stream_priority=0),
            emit_frames=emit_frames,
        )
        built.capture()
        return built

    with torch.inference_mode():
        tail = runner("tail", QWEN3_TTS_TAIL_GRAPH_FRAMES, (), emit_frames=1)
        warm = runner("warm", tuple(range(1, STEADY_STRIDE + 1)), (STEADY_STRIDE,))
    encoder = served_tokenizer.model.encoder
    hop = int(served_tokenizer.model.encode_downsample_rate)
    samples_per_frame = int(incremental.total_upsample)

    def replay(graphs, codes: torch.Tensor, slot: int, valid_frames: list[int] | None = None) -> torch.Tensor:
        waveform = graphs.decode_slots(codes, [slot], valid_frames=valid_frames)
        assert waveform is not None, f"{graphs.mode} graph missed width {int(codes.shape[-1])}"
        return waveform.reshape(-1).float().cpu()

    served_audio, truth_audio = {}, {}
    with torch.inference_mode():
        for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en"):
            data, rate = soundfile.read(sample.ref_audio, dtype="float32", always_2d=False)
            waveform = torch.from_numpy(served_tokenizer._normalize_audio_inputs([data], sr=rate)[0])  # noqa: leading-underscore
            frames = -(-waveform.numel() // hop)
            values = torch.zeros((1, 1, frames * hop), device=device, dtype=torch.bfloat16)
            values[0, 0, : waveform.numel()].copy_(waveform)
            codes = encoder.encode(values, num_quantizers=num_quantizers, return_dict=True).audio_codes[..., :frames]
            reference_frames = max(args.min_reference, frames // 4)
            if frames <= reference_frames + 1:
                continue
            else:
                pass
            slot = arena.acquire()
            audio = [replay(tail, codes[..., : reference_frames + 1], slot, [reference_frames + 1])]
            position = reference_frames + 1
            widths = iter(RAMP[1:])
            while position < frames:
                end = min(position + next(widths, STEADY_STRIDE), frames)
                audio.append(replay(warm, codes[..., position:end], slot))
                position = end
            arena.release(slot)
            served_audio[sample.sample_id] = torch.cat(audio)
            if truth_decoder is not None:
                truth_audio[sample.sample_id] = (
                    truth_decoder(codes).reshape(-1)[reference_frames * samples_per_frame :].float().cpu()
                )
            else:
                pass
    write_arm(out_dir, args.arm, served_audio)
    if truth_decoder is not None:
        write_arm(out_dir, "truth", truth_audio)
    else:
        pass
    print(f"{args.arm}: {len(served_audio)} clips written{' with the fp32 truth' if truth_decoder is not None else ''}")


if __name__ == "__main__":
    main()
