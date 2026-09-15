#!/usr/bin/env python3
"""Stage 1: every Fun-CosyVoice3 serving call on its own, at the operating
points the workload produces, with each GPU activity attributed to the Python
range that launched it.

Run on the H100 venv from a worktree at the tree under test, alone on the GPU,
no server:

  python tasks/cosyvoice_utilization_20260912/stage1/profile_components.py \
      --device cuda:0 --components flow,hift,preprocess --out stage1-out

A serving call is the unit that costs the pipeline time: one packed hop, one
packed final, one buffered Flow group, one HiFT delta, one HiFT batch, one
reference encode. Each point runs the real method of the vocoder the serving
factory builds (bf16 autocast, the same patches), with a real SeedTTS prompt and
random speech tokens of the chosen length, because time and kernels depend on
shapes and not on token values. The points come from the c16 call ledger of
readout 03: rows 1 and 16, prompt 125 tokens (p50), first hop window 28 tokens,
a late hop at offset 275 with hop 100, a batch with one runaway row of 2,048
tokens, finals of 125 (mean) and 500 tokens, HiFT histories up to 4,000 frames.

For each point the ledger gives: wall ms (synchronized median, no
instrumentation), device busy ms (union of device intervals) and its share of
wall, launches, graph launches, syncs and blocking copies with the host time
blocked in them, and per range (module forward or annotated function) the
launches, self and inclusive device ms and pointwise kernels, the top kernels
with their owning ranges, and ranges whose kernel sequence repeats identically.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage0"))

from common import (  # noqa: E402
    FLOW_AUDIO_SR,
    MODEL_ID,
    PROMPT_AUDIO_SR,
    build_streams,
    flow_input,
    load_vocoder,
    provenance,
)
from trace_ledger import measure, render_markdown  # noqa: E402

from sglang_omni.models.fun_cosyvoice3 import packed_dit, stages  # noqa: E402
from sglang_omni.models.fun_cosyvoice3.sglang_model import VOCAB_SIZE  # noqa: E402
from sglang_omni.models.fun_cosyvoice3.streaming import (  # noqa: E402
    PRE_LOOKAHEAD_LEN,
    TOKEN_MEL_RATIO,
)

PROMPT_TOKENS = 125
FIRST_HOP_WINDOW = 25 + PRE_LOOKAHEAD_LEN
LATE_HOP_WINDOW = 275 + 100 + PRE_LOOKAHEAD_LEN
RUNAWAY_WINDOW = 2048 + PRE_LOOKAHEAD_LEN


def flow_functions():
    return [
        (stages, "pack_flow_inputs"),
        (stages, "prepare_flow_conditioning"),
        (stages, "generate_flow_packed"),
        (stages, "generate_flow"),
        (stages, "solve_flow_euler"),
        (stages, "split_generated_mels"),
        (stages, "pack_rows"),
        (stages, "gather_rows"),
        (stages, "scatter_rows"),
        (stages, "solve_flow_euler_packed"),
        (packed_dit, "pack_rows"),
        (packed_dit, "gather_rows"),
        (packed_dit, "scatter_rows"),
        (packed_dit.RowAttention, "__call__"),
        (packed_dit.PackedDiT, "forward"),
        (packed_dit.PackedDiT, "_conv_pos_embed"),
        (packed_dit.PackedDiT, "_rope"),
        (stages.FlowCudaGraphRunner, "run"),
    ]


def hift_functions(hift):
    generator = type(hift)
    return [
        (generator, "inference"),
        (generator, "decode"),
        (generator, "_stft"),
        (generator, "_istft"),
        (stages.CosyVoice3Vocoder, "hift_delta"),
        (stages.CosyVoice3Vocoder, "mel2wav_batch"),
    ]


def profile_flow(vocoder, stream, device, out_dir, repeats):
    generator = torch.Generator().manual_seed(0)

    def rows(count, tokens):
        return [
            flow_input(
                replace(
                    stream,
                    tokens=torch.randint(
                        0,
                        VOCAB_SIZE,
                        (1, tokens),
                        dtype=torch.int32,
                        generator=generator,
                    ),
                ),
                tokens,
            )
            for _ in range(count)
        ]

    def autocast():
        return torch.autocast(device_type="cuda", dtype=vocoder.autocast_dtype)

    buffered_rows = rows(16, 125)
    frames = max(
        stages.pack_flow_inputs(vocoder.flow.flow, buffered_rows).total_mel_lengths
    )
    bucket = (
        -(-frames // stages.FLOW_CUDA_GRAPH_FRAME_BUCKET)
        * stages.FLOW_CUDA_GRAPH_FRAME_BUCKET
    )
    runner = stages.FlowCudaGraphRunner(
        vocoder.flow, device=torch.device(device), autocast_dtype=vocoder.autocast_dtype
    )
    runner.capture(((len(buffered_rows), bucket),))

    def buffered(graph):
        def call():
            vocoder.flow.cuda_graph_runner = runner if graph else None
            with autocast():
                return vocoder.flow.inference(buffered_rows)

        return call

    points = {
        "flow_hop_first_rows1": lambda items=rows(
            1, FIRST_HOP_WINDOW
        ): vocoder.hop_batch(items),
        "flow_hop_first_rows16": lambda items=rows(
            16, FIRST_HOP_WINDOW
        ): vocoder.hop_batch(items),
        "flow_hop_late_rows16": lambda items=rows(
            16, LATE_HOP_WINDOW
        ): vocoder.hop_batch(items),
        "flow_hop_runaway_rows16": lambda items=rows(15, 78) + rows(
            1, RUNAWAY_WINDOW
        ): vocoder.hop_batch(items),
        "flow_final_rows1": lambda items=rows(1, 125): vocoder.leftover_batch(items),
        "flow_final_rows16": lambda items=rows(16, 125): vocoder.leftover_batch(items),
        "flow_final_long_rows16": lambda items=rows(16, 500): vocoder.leftover_batch(
            items
        ),
        f"flow_buffered_rows16_graph_{bucket}": buffered(True),
        "flow_buffered_rows16_eager": buffered(False),
    }
    ledgers = []
    with torch.inference_mode():
        for label, call in points.items():
            print(f"profiling {label}", flush=True)
            ledgers.append(
                measure(
                    call,
                    label=label,
                    out_dir=out_dir,
                    roots={"flow": vocoder.flow.flow},
                    functions=flow_functions(),
                    repeats=repeats,
                )
            )
    vocoder.flow.cuda_graph_runner = None
    return ledgers


def profile_hift(vocoder, stream, out_dir, repeats):
    with torch.inference_mode():
        item = flow_input(
            replace(
                stream,
                tokens=torch.randint(
                    0, VOCAB_SIZE, (1, FIRST_HOP_WINDOW), dtype=torch.int32
                ),
            ),
            FIRST_HOP_WINDOW,
        )
        hop_mel = vocoder.hop_batch([item])[0]
    samples_per_frame = int(vocoder.hift.istft_params["hop_len"])
    for rate in vocoder.hift.upsample_rates:
        samples_per_frame *= int(rate)
    new_frames = int(hop_mel.shape[2])

    def history(frames):
        return hop_mel.repeat(1, 1, frames // new_frames) if frames else None

    points = {}
    for frames in (0, 1000, 4000):
        points[f"hift_hop_history{frames}"] = (
            lambda held=history(frames), frames=frames: vocoder.hift_delta(
                hop_mel,
                hift_mel=held,
                speech_offset=frames * samples_per_frame,
                finalize=False,
            )
        )
    points["hift_final_history1000"] = lambda held=history(1000): vocoder.hift_delta(
        hop_mel, hift_mel=held, speech_offset=1000 * samples_per_frame, finalize=True
    )
    batch_mel = hop_mel.repeat(1, 1, 250 // new_frames)
    points["hift_batch_rows16_250"] = lambda: vocoder.mel2wav_batch([batch_mel] * 16)
    ledgers = []
    with torch.inference_mode():
        for label, call in points.items():
            print(f"profiling {label}", flush=True)
            ledgers.append(
                measure(
                    call,
                    label=label,
                    out_dir=out_dir,
                    roots={"hift": vocoder.hift},
                    functions=hift_functions(vocoder.hift),
                    repeats=repeats,
                )
            )
    return ledgers


def profile_preprocess(checkpoint, device, repeats):
    from benchmarks.dataset.prepare import SEEDTTS_DATASET_ID, SEEDTTS_DATASET_REVISION
    from benchmarks.dataset.seedtts import load_seedtts_samples
    from sglang_omni.models.fun_cosyvoice3.request_builders import (
        _load_prompt_audio,
        _load_prompt_audio_24k,
    )
    from sglang_omni.models.fun_cosyvoice3.utils import (
        SpeakerEncoder,
        SpeechTokenizerV3,
        extract_prompt_speech_feat,
    )

    threads = max(1, min(16, os.cpu_count() or 1))
    tokenizer = SpeechTokenizerV3(
        os.path.join(checkpoint, "speech_tokenizer_v3.onnx"),
        device=device,
        intra_op_threads=threads,
    )
    speaker = SpeakerEncoder(
        os.path.join(checkpoint, "campplus.onnx"),
        device=device,
        intra_op_threads=threads,
    )
    references = []
    seen = set()
    for sample in load_seedtts_samples(
        SEEDTTS_DATASET_ID, 400, split="en", revision=SEEDTTS_DATASET_REVISION
    ):
        if sample.ref_audio not in seen:
            seen.add(sample.ref_audio)
            references.append(
                (
                    len(_load_prompt_audio(sample.ref_audio)) / PROMPT_AUDIO_SR,
                    sample.ref_audio,
                )
            )
    references.sort()
    chosen = {
        "shortest": references[0],
        "median": references[len(references) // 2],
        "longest": references[-1],
    }

    def timed(call):
        started = time.perf_counter()
        result = call()
        return (time.perf_counter() - started) * 1e3, result

    rows = []
    for name, (seconds, path) in chosen.items():
        steps = {
            "load 16k": lambda: _load_prompt_audio(path),
            "load 24k": lambda: _load_prompt_audio_24k(path),
        }
        audio_16k = _load_prompt_audio(path)
        audio_24k = _load_prompt_audio_24k(path)
        steps["campplus embedding"] = lambda: speaker.extract_embedding(
            audio_16k, PROMPT_AUDIO_SR
        )
        steps["s3 tokenizer"] = lambda: tokenizer.extract_speech_token(
            audio_16k, PROMPT_AUDIO_SR
        )
        steps["prompt mel"] = lambda: extract_prompt_speech_feat(
            audio_24k, FLOW_AUDIO_SR
        )
        for step, call in steps.items():
            first_ms, _ = timed(call)
            repeat_ms = statistics.median(timed(call)[0] for _ in range(repeats))
            rows.append(
                {
                    "reference": name,
                    "seconds": seconds,
                    "step": step,
                    "first_call_ms": first_ms,
                    "repeat_median_ms": repeat_ms,
                }
            )
            print(
                f"{name} {seconds:.1f} s {step}: first {first_ms:.1f} ms, repeat {repeat_ms:.1f} ms",
                flush=True,
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--components", default="flow,hift,preprocess")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    components = set(args.components.split(","))
    os.makedirs(args.out, exist_ok=True)
    info = provenance(args.device)
    info["sglang_omni"] = stages.__file__
    checkpoint, vocoder = load_vocoder(args.model, args.device, torch.bfloat16)
    stream = build_streams(
        checkpoint, args.device, count=1, prompt_tokens=PROMPT_TOKENS, min_generated=1
    )[0]
    print(f"prompt {stream.sample_id}, {PROMPT_TOKENS} tokens", flush=True)
    ledgers = []
    if "flow" in components:
        ledgers += profile_flow(
            vocoder, stream, args.device, os.path.join(args.out, "traces"), args.repeats
        )
    if "hift" in components:
        ledgers += profile_hift(
            vocoder, stream, os.path.join(args.out, "traces"), args.repeats
        )
    preprocess = (
        profile_preprocess(checkpoint, args.device, args.repeats)
        if "preprocess" in components
        else []
    )

    with open(os.path.join(args.out, "components.json"), "w") as handle:
        json.dump(
            {"provenance": info, "ledgers": ledgers, "preprocess": preprocess},
            handle,
            indent=1,
        )
    markdown = render_markdown(ledgers)
    if preprocess:
        markdown += "\n### preprocessing, one reference\n\n| reference | seconds | step | first call ms | repeat median ms |\n|---|---|---|---|---|\n"
        for row in preprocess:
            markdown += (
                f"| {row['reference']} | {row['seconds']:.1f} | {row['step']} | "
                f"{row['first_call_ms']:.1f} | {row['repeat_median_ms']:.1f} |\n"
            )
    with open(os.path.join(args.out, "components.md"), "w") as handle:
        handle.write(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
