# SPDX-License-Identifier: Apache-2.0
"""Streaming HiFT step on the tree it runs from, real checkpoint, main's whole history rows
or step 1's windowed rows (HiftStepResult present).

Timing (default): one step of R rows, each a history of H mel frames of which the last N
are new, hops or finals. Median wall with the device synchronized, kernels and device time
of one step, peak memory above what was allocated before the step.

Accuracy (--accuracy): a 60 s mel of real speech (CosyVoice's two prompt wavs, repeated)
streamed through hift_step in the hop schedule 50, 100, 200, 200, ... frames and a final;
every step's samples against HiFT over that step's whole history with its sine source in
float64, as max absolute error and SNR in dB.

    python hift_window_bench_c20261010.py --out /data/hw/main.json
    python hift_window_bench_c20261010.py --accuracy --out /data/hw/main_acc.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time

import torch
import torchaudio
from matcha.utils.audio import mel_spectrogram
from torch.profiler import ProfilerActivity, profile

from sglang_omni.models.fun_cosyvoice3 import stages

IS_WINDOWED = hasattr(stages, "HiftStepResult")
SAMPLES_PER_FRAME = 480
# rows, new frames, history frames, final: c1 hops and a 60 s history, c16 at SeedTTS
# lengths and at long form (31 s), 32 rows at 39 s
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
    (32, 100, 1950, False),
)
REPEATS = 20


def step_rows(histories: list[torch.Tensor], new_frames: int, is_final: bool) -> list:
    rows = []
    for history in histories:
        emitted_frame = history.shape[2] - new_frames
        if IS_WINDOWED:
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
        else:
            rows.append(
                stages.HiftStepRow(
                    history=history,
                    emitted_samples=emitted_frame * SAMPLES_PER_FRAME,
                    is_final=is_final,
                )
            )
    return rows


def timing(vocoder: stages.CosyVoice3Vocoder) -> dict[str, dict[str, float]]:
    generator = torch.Generator(device="cuda").manual_seed(0)
    report: dict[str, dict[str, float]] = {}
    for rows, new_frames, history_frames, is_final in CASES:
        histories = [
            torch.randn(1, 80, history_frames, device="cuda", generator=generator) * 3
            for _ in range(rows)
        ]
        step = step_rows(histories, new_frames, is_final)
        for _ in range(3):
            vocoder.hift_step(step)
        torch.cuda.synchronize()
        walls = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            vocoder.hift_step(step)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - start) * 1e3)
        resident = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        vocoder.hift_step(step)
        torch.cuda.synchronize()
        peak_mib = (torch.cuda.max_memory_allocated() - resident) / 2**20
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            vocoder.hift_step(step)
            torch.cuda.synchronize()
        kernels = [e for e in prof.events() if e.device_type.name == "CUDA"]
        name = f"rows={rows} new={new_frames} history={history_frames} final={is_final}"
        report[name] = {
            "wall_ms_median": round(statistics.median(walls), 2),
            "kernels": len(kernels),
            "device_ms": round(sum(e.device_time for e in kernels) / 1e3, 2),
            "peak_mib": round(peak_mib, 1),
        }
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
        audio,
        n_fft=1920,
        num_mels=80,
        sampling_rate=24000,
        hop_size=480,
        win_size=1920,
        fmin=0,
        fmax=None,
        center=False,
    )
    return mel[:, :, :frames].cuda()


def float64_source_hift(
    hift: torch.nn.Module, source_module: torch.nn.Module, mel: torch.Tensor, finalize: bool
) -> torch.Tensor:
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


def accuracy(vocoder: stages.CosyVoice3Vocoder, cosyvoice_dir: str) -> dict[str, dict]:
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
        if IS_WINDOWED:
            tail = new_frames if tail is None else torch.cat((tail, new_frames), dim=2)
            row = stages.HiftStepRow(
                mel=tail,
                mel_start_frame=tail_start,
                emitted_samples=emitted,
                source_phase_cycles=phase,
                is_final=is_final,
            )
            (result,) = vocoder.hift_step([row])
            samples, emitted_after = result.samples, result.emitted_samples
            tail = tail[:, :, result.mel_start_frame - tail_start :]
            tail_start, phase = result.mel_start_frame, result.source_phase_cycles
        else:
            row = stages.HiftStepRow(
                history=mel[:, :, :frames], emitted_samples=emitted, is_final=is_final
            )
            ((samples, emitted_after),) = vocoder.hift_step([row])
        reference = float64_source_hift(hift, source_module, mel[:, :, :frames], is_final)
        expected = reference[:, emitted:].cpu()
        assert emitted_after == reference.shape[1], (emitted_after, reference.shape)
        error = (samples - expected).abs()
        snr_db = 10 * math.log10(
            float(expected.pow(2).sum()) / max(float((samples - expected).pow(2).sum()), 1e-30)
        )
        report[f"step={step} history={frames}"] = {
            "max_abs_error": float(error.max()),
            "snr_db": round(snr_db, 1),
        }
        print(f"step={step} history={frames}", report[f"step={step} history={frames}"], flush=True)
        emitted = emitted_after
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="FunAudioLLM/Fun-CosyVoice3-0.5B-2512")
    parser.add_argument("--cosyvoice-dir", default="/workspace/CosyVoice")
    parser.add_argument("--out", required=True)
    parser.add_argument("--accuracy", action="store_true")
    args = parser.parse_args()
    checkpoint_dir = stages.resolve_checkpoint(args.model)
    flow, hift = stages.load_cosyvoice3_flow_hift(
        checkpoint_dir, device="cuda", autocast_dtype=torch.bfloat16
    )
    vocoder = stages.CosyVoice3Vocoder(flow, hift, autocast_dtype=torch.bfloat16)
    print(f"windowed={IS_WINDOWED} tree={stages.__file__}", flush=True)
    with torch.inference_mode(), vocoder.stream_context:
        if args.accuracy:
            report = accuracy(vocoder, args.cosyvoice_dir)
        else:
            report = timing(vocoder)
    with open(args.out, "w") as handle:
        json.dump({"windowed": IS_WINDOWED, "report": report}, handle, indent=1)


if __name__ == "__main__":
    main()
