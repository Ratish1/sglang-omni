"""The Qwen3-TTS speaker embedding of SeedTTS EN references as serving computes it, on the tree on PYTHONPATH (one arm
per process): the prompt frontend loaded and its speaker encoder graphs captured at the reference encoder's bucket
ladder, each reference resampled to the encoder's rate as the request builder does. Per reference: the wall time of
embed() to a synchronized embedding (its first call and the median of the repeats); the embeddings are saved, so two
trees compare with --compare (cosine similarity and max |delta|). --meta takes the references of a long-reference
list instead of SeedTTS EN.

usage: PYTHONPATH=<tree> python3 speaker_embed_bench.py OUT.pt [--card 3] [--references 64] [--repeats 20] [--meta L]
       python3 speaker_embed_bench.py --compare A.pt B.pt
"""

import argparse
import statistics
import time
from pathlib import Path

import librosa
import soundfile
import torch

from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID
from benchmarks.dataset.seedtts import load_seedtts_samples
from sglang_omni.models.qwen3_tts.prompt_frontend import load_qwen3_tts_prompt_frontend
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
from sglang_omni.models.qwen3_tts.stages import load_qwen3_tts_tokenizer, register_qwen3_tts_hf_config
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?")
    parser.add_argument("--card", type=int, default=3)
    parser.add_argument("--references", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--meta", type=Path)
    args = parser.parse_args()
    if args.compare:
        first, second = (torch.load(path) for path in args.compare)
        cosines = torch.nn.functional.cosine_similarity(first["embeddings"].float(), second["embeddings"].float())
        delta = (first["embeddings"].float() - second["embeddings"].float()).abs().amax(dim=1)
        print(
            f"cosine min {cosines.min().item():.6f} median {cosines.median().item():.6f}; "
            f"max |delta| median {delta.median().item():.4e} max {delta.max().item():.4e}; "
            f"embed ms median of references {first['median_ms']:.3f} / {second['median_ms']:.3f}"
        )
        return
    else:
        pass

    torch.cuda.set_device(args.card)
    device = torch.device("cuda", args.card)
    register_qwen3_tts_hf_config()
    tokenizer = load_qwen3_tts_tokenizer(MODEL, device=str(device), dtype="bfloat16", attn_implementation=None)
    frontend = load_qwen3_tts_prompt_frontend(resolve_checkpoint(MODEL), device=device, dtype=torch.bfloat16)
    runner = frontend.speaker_encoder_graph_runner
    runner.capture(DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES, tokenizer.model.encode_downsample_rate)
    print(f"speaker encoder graphs: {runner.stats()}")
    rate = frontend.speaker_encoder_sample_rate
    if args.meta is None:
        references = [sample.ref_audio for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, args.references, split="en")]
    else:
        lines = [line for line in args.meta.read_text().splitlines() if line]
        references = sorted({str(args.meta.parent / line.split("|")[2]) for line in lines})
    embeddings = []
    medians_ms = []
    first_ms = []
    with torch.inference_mode():
        for reference in references:
            audio, source_rate = soundfile.read(reference, dtype="float32")
            waveform = librosa.resample(audio, orig_sr=source_rate, target_sr=rate) if source_rate != rate else audio
            times_ms = []
            for _ in range(args.repeats):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                embedding = runner.embed(waveform)
                torch.cuda.synchronize(device)
                times_ms.append(1000 * (time.perf_counter() - start))
            embeddings.append(embedding.float().cpu())
            medians_ms.append(statistics.median(times_ms[1:]))
            first_ms.append(times_ms[0])
            if args.meta is not None:
                print(f"{Path(reference).name}: {len(waveform)} samples, first {times_ms[0]:.1f} ms, then {medians_ms[-1]:.1f} ms")
            else:
                pass
    median_ms = statistics.median(medians_ms)
    print(
        f"embed wall ms over {len(references)} references: median {median_ms:.3f}, "
        f"min {min(medians_ms):.3f}, max {max(medians_ms):.3f}; first calls median {statistics.median(first_ms):.3f}, "
        f"max {max(first_ms):.3f}; replays {runner.replays}, misses {runner.misses}"
    )
    torch.save({"embeddings": torch.stack(embeddings), "median_ms": median_ms}, args.out)


if __name__ == "__main__":
    main()
