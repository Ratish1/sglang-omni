# SPDX-License-Identifier: Apache-2.0
"""Step 2's HiFT step with and without its CUDA graphs, on the tree it runs from and the real
checkpoint.

Timing (default): per case (R rows, each a history of H mel frames of which the last N are new, hops
or finals), the step run eagerly (no runner) and through the graph runner built as the factory builds
it: median wall with the device synchronized, kernels, host launch calls (cudaLaunchKernel family,
cudaGraphLaunch, memcpy and memset calls) and device time of one step. The runner's capture log line
(tiers, time, memory) is printed.

Accuracy (--accuracy): a 60 s mel of real speech streamed through hift_step with the graphs, hop
schedule 50, 100, 200, 200, ... frames and a final, against HiFT over each step's whole history with
its sine source in float64.

    python hift_graph_bench_c20261010.py --out /data/hg/graphs.json
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import statistics
import time

import torch
import torchaudio
from matcha.utils.audio import mel_spectrogram
from sglang.srt.arg_groups.cuda_graph_hook import generate_prefill_cuda_graph_batch_sizes
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.hift_cuda_graph import HiftCudaGraphRunner
from sglang_omni.models.fun_cosyvoice3.vocoder_graph_capture import (
    VocoderGraphCapture,
    graph_row_tiers,
)
from sglang_omni.platforms import current_platform

SAMPLES_PER_FRAME = 480
CASES = (
    (1, 50, 50, False),
    (1, 100, 150, False),
    (1, 200, 350, False),
    (1, 200, 3000, False),
    (8, 100, 250, False),
    (16, 50, 50, False),
    (16, 100, 150, False),
    (16, 200, 350, False),
    (16, 200, 600, False),
    (16, 100, 350, True),
    (16, 200, 1550, False),
)
REPEATS = 20
LAUNCH_CALLS = (
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
    "cudaGraphLaunch",
    "cudaMemcpyAsync",
    "cudaMemsetAsync",
)


def build_runner(vocoder: stages.CosyVoice3Vocoder, max_rows: int) -> HiftCudaGraphRunner:
    """The runner as create_vocoder_executor builds it at the default hop schedule."""
    device = torch.device("cuda", torch.cuda.current_device())
    context = stages.HIFT_DECODE_CONTEXT_FRAMES + vocoder.hift_hold_frames
    ratio = vocoder.flow.token_mel_ratio
    shortest = ratio + context
    longest = ratio * (100 + 3 - 1) + context
    ladder = generate_prefill_cuda_graph_batch_sizes(2 * longest)
    smallest = next(frames for frames in ladder if frames >= shortest)
    largest = next(frames for frames in ladder if frames >= longest)
    runner = HiftCudaGraphRunner(
        vocoder.hift,
        graphs=VocoderGraphCapture(
            current_platform.get_device_graph_backend(device),
            device=device,
            autocast_dtype=torch.bfloat16,
        ),
        mel_channels=vocoder.flow.output_size,
        f0_left_frames=vocoder.hift_f0_left_frames,
        samples_per_frame=vocoder.hift_samples_per_mel_frame,
        row_tiers=graph_row_tiers(max_rows),
        frame_tiers=[frames for frames in ladder if smallest <= frames <= largest],
    )
    runner.capture()
    return runner


def step_rows(histories: list[torch.Tensor], new_frames: int, is_final: bool) -> list:
    rows = []
    for history in histories:
        emitted_frame = history.shape[2] - new_frames
        start_frame = max(emitted_frame - stages.HIFT_DECODE_CONTEXT_FRAMES, 0)
        mel_start_frame = max(start_frame - 8, 0)
        rows.append(
            stages.HiftStepRow(
                mel=history[:, :, mel_start_frame:],
                mel_start_frame=mel_start_frame,
                emitted_samples=emitted_frame * SAMPLES_PER_FRAME,
                source_phase_cycles=(
                    torch.rand(9, device="cuda", dtype=torch.float64)
                    if start_frame > 0
                    else None
                ),
                is_final=is_final,
            )
        )
    return rows


def measure(vocoder: stages.CosyVoice3Vocoder, step: list) -> dict[str, float]:
    for _ in range(3):
        vocoder.hift_step(step)
    torch.cuda.synchronize()
    walls = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        vocoder.hift_step(step)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - start) * 1e3)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        vocoder.hift_step(step)
        torch.cuda.synchronize()
    events = prof.events()
    kernels = [e for e in events if e.device_type.name == "CUDA"]
    launches = [e for e in events if e.device_type.name == "CPU" and e.name in LAUNCH_CALLS]
    return {
        "wall_ms_median": round(statistics.median(walls), 2),
        "kernels": len(kernels),
        "launch_calls": len(launches),
        "device_ms": round(sum(e.device_time for e in kernels) / 1e3, 2),
    }


def timing(vocoder: stages.CosyVoice3Vocoder, runner: HiftCudaGraphRunner) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(0)
    report: dict[str, dict] = {}
    for rows, new_frames, history_frames, is_final in CASES:
        histories = [
            torch.randn(1, 80, history_frames, device="cuda", generator=generator) * 3
            for _ in range(rows)
        ]
        step = step_rows(histories, new_frames, is_final)
        vocoder.hift_cuda_graph_runner = None
        eager = measure(vocoder, step)
        vocoder.hift_cuda_graph_runner = runner
        graphs = measure(vocoder, step)
        name = f"rows={rows} new={new_frames} history={history_frames} final={is_final}"
        report[name] = {"eager": eager, "graphs": graphs}
        print(name, report[name], flush=True)
    return report


def real_speech_mel(cosyvoice_dir: str, frames: int) -> torch.Tensor:
    pieces = []
    for name in ("zero_shot_prompt.wav", "cross_lingual_prompt.wav"):
        audio, rate = torchaudio.load(f"{cosyvoice_dir}/asset/{name}")
        pieces.append(torchaudio.functional.resample(audio[:1], rate, 24000))
    audio = torch.cat(pieces, dim=1)
    audio = audio.repeat(1, math.ceil(frames * SAMPLES_PER_FRAME / audio.shape[1]) + 1)
    mel = mel_spectrogram(
        audio, n_fft=1920, num_mels=80, sampling_rate=24000, hop_size=480,
        win_size=1920, fmin=0, fmax=None, center=False,
    )
    return mel[:, :, :frames].cuda()


def float64_source_hift(hift, source_module, mel: torch.Tensor, finalize: bool) -> torch.Tensor:
    f0 = hift.f0_predictor(mel.double(), finalize=finalize).float()
    source, _, _ = source_module(hift.f0_upsamp(f0.double()[:, None]).transpose(1, 2))
    source = source.transpose(1, 2).float()
    if finalize:
        return hift.decode(x=mel, s=source, finalize=True)
    else:
        return hift.decode(
            x=mel[:, :, : -hift.f0_predictor.condnet[0].causal_padding],
            s=source,
            finalize=False,
        )


def accuracy(vocoder: stages.CosyVoice3Vocoder, cosyvoice_dir: str) -> dict:
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    hift = vocoder.hift
    source_module = copy.deepcopy(hift.m_source).double()
    mel = real_speech_mel(cosyvoice_dir, 3000)
    history_ends = [50, 150, 350]
    while history_ends[-1] + 200 < 3000:
        history_ends.append(history_ends[-1] + 200)
    history_ends.append(3000)
    tail, tail_start, phase, emitted, previous = None, 0, None, 0, 0
    report: dict[str, dict] = {}
    for step, frames in enumerate(history_ends):
        is_final = step == len(history_ends) - 1
        new_frames = mel[:, :, previous:frames]
        previous = frames
        tail = new_frames if tail is None else torch.cat((tail, new_frames), dim=2)
        (result,) = vocoder.hift_step(
            [
                stages.HiftStepRow(
                    mel=tail,
                    mel_start_frame=tail_start,
                    emitted_samples=emitted,
                    source_phase_cycles=phase,
                    is_final=is_final,
                )
            ]
        )
        tail = tail[:, :, result.mel_start_frame - tail_start :]
        tail_start, phase = result.mel_start_frame, result.source_phase_cycles
        reference = float64_source_hift(hift, source_module, mel[:, :, :frames], is_final)
        expected = reference[:, emitted:].cpu()
        assert result.emitted_samples == reference.shape[1]
        error = result.samples - expected
        snr_db = 10 * math.log10(
            float(expected.pow(2).sum()) / max(float(error.pow(2).sum()), 1e-30)
        )
        name = f"step={step} history={frames}"
        report[name] = {"max_abs_error": float(error.abs().max()), "snr_db": round(snr_db, 1)}
        print(name, report[name], flush=True)
        emitted = result.emitted_samples
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--cosyvoice-dir", default="/workspace/CosyVoice")
    parser.add_argument("--out", required=True)
    parser.add_argument("--accuracy", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    checkpoint_dir = stages.resolve_checkpoint(args.model)
    flow, hift = stages.load_cosyvoice3_flow_hift(
        checkpoint_dir, device="cuda", autocast_dtype=torch.bfloat16
    )
    vocoder = stages.CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    print(f"tree={stages.__file__}", flush=True)
    with torch.inference_mode(), vocoder.stream_context:
        runner = build_runner(vocoder, max_rows=16)
        if args.accuracy:
            vocoder.hift_cuda_graph_runner = runner
            report = accuracy(vocoder, args.cosyvoice_dir)
        else:
            report = timing(vocoder, runner)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=1)


if __name__ == "__main__":
    main()
