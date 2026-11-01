"""The reference encoder alone, built as the engine builds it (load_qwen3_tts_tokenizer with a float32 encoder, then
Qwen3TTSRefCodeBatcher, which compiles once and captures every bucket), encoding every SeedTTS EN reference clip once
through the batcher; writes {clip path: codes} to OUT.pt and the compiled reductions' picks found in the inductor cache.
--eager builds the batcher with no buckets (main's eager encode); --deterministic adds Inductor's deterministic option
to the encoder's compile options (no numerics-affecting benchmarking, reductions on one filtered config). A second pass
times the encodes. Compare runs with --compare A.pt B.pt ... [--identity ROOT], which prints clips equal per pair and,
with a seeded identity root, how the differing clips match its 63.

usage: python3 refenc_codes.py --gpu N --out PREFIX [--eager] [--deterministic]
       python3 refenc_codes.py --compare A.pt B.pt ... [--identity /data/next_20261101/block59/identity_codec]
"""

import argparse
import hashlib
import os
import time
from pathlib import Path

import soundfile
import torch

MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"


def parse_option(text: str) -> tuple[str, bool | int]:
    key, value = text.split("=", 1)
    if value in ("True", "False"):
        return key, value == "True"
    else:
        return key, int(value)


def encode_all(
    gpu: int, eager: bool, deterministic: bool, options: list[str]
) -> tuple[dict[str, torch.Tensor], float, float, dict[int, float]]:
    from benchmarks.dataset.seedtts import SEEDTTS_DATASET_ID, load_seedtts_samples
    from sglang_omni.models.qwen3_tts import reference_encoder_cuda_graph, stages
    from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import (
        DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES,
    )
    from sglang_omni.models.qwen3_tts.request_builders import Qwen3TTSRefCodeBatcher

    if deterministic:
        reference_encoder_cuda_graph.REFERENCE_ENCODER_INDUCTOR_OPTIONS["deterministic"] = True
    else:
        pass
    for option in options:
        key, value = parse_option(option)
        reference_encoder_cuda_graph.REFERENCE_ENCODER_INDUCTOR_OPTIONS[key] = value
    torch.cuda.set_device(gpu)
    tokenizer = stages.load_qwen3_tts_tokenizer(
        MODEL, device=f"cuda:{gpu}", dtype="bfloat16", attn_implementation=None, float32_encoder=True
    )
    began = time.perf_counter()
    batcher = Qwen3TTSRefCodeBatcher(
        tokenizer, graph_bucket_frames=() if eager else DEFAULT_QWEN3_TTS_REFERENCE_ENCODER_BUCKET_FRAMES
    )
    built = time.perf_counter() - began
    clips = sorted({sample.ref_audio for sample in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")})
    audios = [(Path(clip).name, *soundfile.read(clip, dtype="float32")) for clip in clips]
    codes = {}
    for name, audio, rate in audios:
        codes[name] = batcher.encode(audio, rate).to(torch.int16).cpu()
    # note (ratish): a second pass times the encodes alone, every kernel loaded and every plan built.
    began = time.perf_counter()
    for name, audio, rate in audios:
        batcher.encode(audio, rate)
    encode_ms = (time.perf_counter() - began) * 1000 / len(audios)
    batcher.close()
    replay_ms = {}
    runner = batcher.graph_runner
    if runner is not None:
        for frames, captured in sorted(runner.graphs.items()):
            with torch.cuda.stream(runner.stream):
                for _ in range(10):
                    captured.graph.replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(runner.stream)
                for _ in range(200):
                    captured.graph.replay()
                end.record(runner.stream)
            end.synchronize()
            replay_ms[frames] = round(start.elapsed_time(end) / 200, 4)
    else:
        pass
    return codes, built, encode_ms, replay_ms


def picks(cache: str) -> list[str]:
    found = []
    for path in sorted(Path(cache).rglob("*.best_config")):
        found.append(f"{path.parent.name}/{path.name}: {path.read_text()[:160]}")
    return found


def compare(paths: list[str], identity: Path | None) -> None:
    runs = {Path(path).stem: torch.load(path) for path in paths}
    names = list(runs)
    clips = sorted(runs[names[0]])
    print(f"clips {len(clips)}")
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            differing = [clip for clip in clips if not torch.equal(runs[first][clip], runs[second][clip])]
            print(f"{first} vs {second}: equal {len(clips) - len(differing)} of {len(clips)}")
    if identity is not None:
        from benchmarks.dataset.seedtts import SEEDTTS_DATASET_ID, load_seedtts_samples

        main = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (identity / "c1n" / "main").rglob("*.wav")}
        branch = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (identity / "c1n" / "branch").rglob("*.wav")}
        differing_samples = {name[: -len(".wav")] for name in main if main[name] != branch.get(name)}
        served = {Path(s.ref_audio).name for s in load_seedtts_samples(SEEDTTS_DATASET_ID, split="en")
                  if s.sample_id in differing_samples}
        print(f"identity root {identity}: clips with differing served audio {len(served)}")
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                coded = {clip for clip in clips if not torch.equal(runs[first][clip], runs[second][clip])}
                print(f"  {first} vs {second}: code-differing {len(coded)}, both {len(coded & served)}, "
                      f"served only {len(served - coded)}, codes only {len(coded - served)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--option", action="append", default=[], help="an Inductor option for the encoder, KEY=VALUE")
    parser.add_argument("--compare", nargs="+")
    parser.add_argument("--identity", type=Path)
    args = parser.parse_args()
    if args.compare:
        compare(args.compare, args.identity)
        return
    else:
        pass
    codes, built, encode_ms, replay_ms = encode_all(args.gpu, args.eager, args.deterministic, args.option)
    torch.save(codes, f"{args.out}.pt")
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
    found = picks(cache) if cache else []
    summary = (f"gpu {args.gpu} eager {args.eager} deterministic {args.deterministic} options {args.option} "
               f"clips {len(codes)} build {built:.1f} s encode {encode_ms:.3f} ms replay ms {replay_ms} "
               f"best_config files {len(found)}")
    with open(f"{args.out}.txt", "w") as handle:
        handle.write(f"{summary} cache {cache}\n")
        handle.writelines(line + "\n" for line in found)
    print(summary)
    os._exit(0)


if __name__ == "__main__":
    main()
