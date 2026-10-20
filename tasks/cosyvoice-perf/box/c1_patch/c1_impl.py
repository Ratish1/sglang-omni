"""C1 for measurement without a repo edit: when sglang_omni.models.fun_cosyvoice3.stages is imported, wrap
CosyVoice3Vocoder.__init__ so a CUDA vocoder's HiFT gets the proposed patch_sine_harmonics
(DESIGN_NEXT_20261012.md 13, the same text) unless HIFT_SINE=0, and with HIFT_SPECTRA=1 D's patch_hift_spectra
(section 14).
Put this directory first on PYTHONPATH."""

import importlib.abc
import importlib.util
import math
import os
import sys

import torch
import torch.nn.functional as F

TARGET = "sglang_omni.models.fun_cosyvoice3.stages"


def patch_sine_harmonics(hift, device: torch.device) -> None:
    sine_generator = hift.m_source.l_sin_gen
    harmonics = torch.arange(
        1, sine_generator.harmonic_num + 2, dtype=torch.float32, device=device
    ).view(1, 1, -1)

    def forward(f0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sine_waves = (
            sine_generator._f02sine(torch.multiply(f0, harmonics))
            * sine_generator.sine_amp
        )
        uv = sine_generator._f02uv(f0)
        noise_amp = uv * sine_generator.noise_std + (1 - uv) * sine_generator.sine_amp / 3
        noise = noise_amp * sine_generator.sine_waves[:, : sine_waves.shape[1]]
        return sine_waves * uv + noise, uv, noise

    sine_generator.forward = forward


def patch_hift_spectra(hift, device: torch.device) -> None:
    n_fft = hift.istft_params["n_fft"]
    hop_length = hift.istft_params["hop_len"]
    bins = n_fft // 2 + 1
    window = hift.stft_window.to(device=device, dtype=torch.float64)
    angle = (2 * math.pi / n_fft) * torch.outer(
        torch.arange(bins, device=device, dtype=torch.float64),
        torch.arange(n_fft, device=device, dtype=torch.float64),
    )
    forward_basis = (torch.cat([torch.cos(angle), -torch.sin(angle)]) * window).T.float()
    one_sided = torch.full((bins, 1), 2.0, device=device, dtype=torch.float64)
    one_sided[0] = 1.0
    one_sided[-1] = 1.0
    inverse_basis = (
        torch.cat([torch.cos(angle) * one_sided, -torch.sin(angle) * one_sided]) * window / n_fft
    ).float()
    squared_window = hift.stft_window.to(device).pow(2)

    def stft(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        frames = F.pad(x[:, None], (n_fft // 2, n_fft // 2), mode="reflect")[:, 0].unfold(-1, n_fft, hop_length)
        spectrum = (frames @ forward_basis).transpose(1, 2)
        return spectrum[:, :bins], spectrum[:, bins:]

    def istft(magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = torch.clip(magnitude, max=1e2)
        spectrum = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1)
        frames = spectrum.transpose(1, 2) @ inverse_basis
        frame_count = frames.shape[1]
        length = n_fft + hop_length * (frame_count - 1)
        signal = torch.ops.aten.unfold_backward(frames, [frames.shape[0], length], 1, n_fft, hop_length)
        envelope = torch.ops.aten.unfold_backward(
            squared_window.expand(1, frame_count, n_fft), [1, length], 1, n_fft, hop_length
        )
        trim = n_fft // 2
        return signal[:, trim:-trim] / envelope[:, trim:-trim]

    hift._stft = stft
    hift._istft = istft


def patch_stages(module):
    vocoder = module.CosyVoice3Vocoder
    init = vocoder.__init__

    def init_patched(self, flow, hift, *args, **kwargs):
        init(self, flow, hift, *args, **kwargs)
        device = next(self.flow.parameters()).device
        if device.type == "cuda":
            if os.environ.get("HIFT_SINE", "1") == "1":
                patch_sine_harmonics(hift, device)
                print(f"c1_patch applied on {device}", file=sys.stderr, flush=True)
            if os.environ.get("HIFT_SPECTRA") == "1":
                patch_hift_spectra(hift, device)
                print(f"hift_spectra applied on {device}", file=sys.stderr, flush=True)

    vocoder.__init__ = init_patched


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen = False

    def find_spec(self, name, path, target=None):
        if name != TARGET or self.seen:
            return None
        self.seen = True
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch_stages(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, PatchOnImport())
print("c1_patch on", file=sys.stderr, flush=True)
