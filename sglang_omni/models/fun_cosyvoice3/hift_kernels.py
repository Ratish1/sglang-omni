# SPDX-License-Identifier: Apache-2.0
"""HiFT's Snake activation, short time Fourier transform and its inverse as Triton kernels.
Snake rounds each step as CosyVoice's eager ops do; every spectral output element sums its
own inputs in one fixed order, so a row's result does not follow what shares its batch."""

from __future__ import annotations

import torch

from sglang_omni.models.fun_cosyvoice3.causal_conv import HAS_TRITON

SAMPLES_PER_PROGRAM = 1024
FRAMES_PER_PROGRAM = 128
OUTPUT_SAMPLES_PER_PROGRAM = 256

if HAS_TRITON:
    import triton
    import triton.language as tl
    from sglang.kernels.ops.diffusion.common.numerics import mul_rn_f32
    from triton.language.extra import libdevice

    # note (ratish): sample counts change every step, one binary serves them all.
    @triton.jit(do_not_specialize=["samples", "channels"])
    def snake_kernel(
        x_ptr,
        alpha_ptr,
        inverse_alpha_ptr,
        out_ptr,
        samples,
        channels,
        BLOCK_SAMPLES: tl.constexpr,
    ):
        row = tl.program_id(1).to(tl.int64)
        offsets = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
        is_sample = offsets < samples
        alpha = tl.load(alpha_ptr + row % channels)
        inverse_alpha = tl.load(inverse_alpha_ptr + row % channels)
        x = tl.load(x_ptr + row * samples + offsets, mask=is_sample)
        sine = libdevice.sin(mul_rn_f32(x, alpha))
        # note (ratish): each product rounds alone, as in the eager ops; an FMA would not.
        out = x + mul_rn_f32(inverse_alpha, mul_rn_f32(sine, sine))
        tl.store(out_ptr + row * samples + offsets, out, mask=is_sample)

    @triton.jit(do_not_specialize=["frames", "padded_samples"])
    def stft_kernel(
        padded_signal_ptr,
        basis_ptr,
        spectrum_ptr,
        frames,
        padded_samples,
        N_FFT: tl.constexpr,
        HOP: tl.constexpr,
        BINS: tl.constexpr,
        BLOCK_FRAMES: tl.constexpr,
    ):
        row = tl.program_id(1).to(tl.int64)
        frame = tl.program_id(0) * BLOCK_FRAMES + tl.arange(0, BLOCK_FRAMES)
        is_frame = frame < frames
        taps = tl.arange(0, N_FFT)
        frame_samples = tl.load(
            padded_signal_ptr
            + row * padded_samples
            + frame[:, None] * HOP
            + taps[None, :],
            mask=is_frame[:, None],
            other=0.0,
        )
        for component in tl.static_range(2 * BINS):
            basis = tl.load(basis_ptr + component * N_FFT + taps)
            tl.store(
                spectrum_ptr + (row * 2 * BINS + component) * frames + frame,
                tl.sum(frame_samples * basis[None, :], axis=1),
                mask=is_frame,
            )

    @triton.jit(do_not_specialize=["frames", "samples"])
    def istft_kernel(
        real_ptr,
        imag_ptr,
        basis_ptr,
        squared_window_ptr,
        out_ptr,
        frames,
        samples,
        N_FFT: tl.constexpr,
        HOP: tl.constexpr,
        BINS: tl.constexpr,
        BINS_BLOCK: tl.constexpr,
        BLOCK_SAMPLES: tl.constexpr,
    ):
        row = tl.program_id(1).to(tl.int64)
        output = tl.program_id(0) * BLOCK_SAMPLES + tl.arange(0, BLOCK_SAMPLES)
        is_output = output < samples
        position = output + N_FFT // 2
        bins = tl.arange(0, BINS_BLOCK)
        is_bin = bins < BINS
        signal = tl.zeros((BLOCK_SAMPLES,), dtype=tl.float32)
        envelope = tl.zeros((BLOCK_SAMPLES,), dtype=tl.float32)
        for overlap in tl.static_range(N_FFT // HOP):
            frame = position // HOP - overlap
            tap = position - frame * HOP
            is_frame = is_output & (frame >= 0) & (frame < frames)
            mask = is_frame[:, None] & is_bin[None, :]
            offsets = (row * BINS + bins[None, :]) * frames + frame[:, None]
            real = tl.load(real_ptr + offsets, mask=mask, other=0.0)
            imag = tl.load(imag_ptr + offsets, mask=mask, other=0.0)
            cosine = tl.load(
                basis_ptr + bins[None, :] * N_FFT + tap[:, None], mask=mask, other=0.0
            )
            sine = tl.load(
                basis_ptr + (BINS + bins[None, :]) * N_FFT + tap[:, None],
                mask=mask,
                other=0.0,
            )
            value = tl.sum(real * cosine + imag * sine, axis=1)
            signal += tl.where(is_frame, value, 0.0)
            envelope += tl.load(squared_window_ptr + tap, mask=is_frame, other=0.0)
        tl.store(out_ptr + row * samples + output, signal / envelope, mask=is_output)

else:
    pass


def snake(
    x: torch.Tensor, alpha: torch.Tensor, inverse_alpha: torch.Tensor
) -> torch.Tensor:
    """x: (batch, channels, samples) float32; alpha and inverse_alpha: (channels,), the
    latter 1 / (alpha + 1e-9) as the module computes it."""
    batch, channels, samples = x.shape
    assert x.is_contiguous() and x.dtype == torch.float32
    out = torch.empty_like(x)
    grid = (triton.cdiv(samples, SAMPLES_PER_PROGRAM), batch * channels)
    snake_kernel[grid](
        x,
        alpha,
        inverse_alpha,
        out,
        samples,
        channels,
        BLOCK_SAMPLES=SAMPLES_PER_PROGRAM,
    )
    return out


def short_time_spectrum(
    padded_signal: torch.Tensor, basis: torch.Tensor, hop_length: int
) -> torch.Tensor:
    """padded_signal: (rows, samples) float32, reflect padded; basis: (2 * bins, n_fft),
    the windowed cosines then the windowed negative sines. Returns (rows, 2 * bins,
    frames), the real parts then the imaginary parts."""
    rows, padded_samples = padded_signal.shape
    components, n_fft = basis.shape
    assert padded_signal.is_contiguous() and padded_signal.dtype == torch.float32
    frames = (padded_samples - n_fft) // hop_length + 1
    spectrum = padded_signal.new_empty(rows, components, frames)
    grid = (triton.cdiv(frames, FRAMES_PER_PROGRAM), rows)
    stft_kernel[grid](
        padded_signal,
        basis,
        spectrum,
        frames,
        padded_samples,
        N_FFT=n_fft,
        HOP=hop_length,
        BINS=components // 2,
        BLOCK_FRAMES=FRAMES_PER_PROGRAM,
    )
    return spectrum


def overlap_add_inverse(
    real: torch.Tensor,
    imag: torch.Tensor,
    basis: torch.Tensor,
    squared_window: torch.Tensor,
    hop_length: int,
) -> torch.Tensor:
    """real, imag: (rows, bins, frames) float32; basis: (2 * bins, n_fft), the one sided
    inverse's windowed cosines then negative sines over n_fft. Each frame's inverse,
    overlap added, divided by the squared window's overlap and trimmed by n_fft / 2 on
    each side, as torch.istft with center=True. Returns (rows, hop_length * (frames - 1)).
    """
    rows, bins, frames = real.shape
    components, n_fft = basis.shape
    assert real.is_contiguous() and imag.is_contiguous() and components == 2 * bins
    samples = hop_length * (frames - 1)
    out = real.new_empty(rows, samples)
    grid = (triton.cdiv(samples, OUTPUT_SAMPLES_PER_PROGRAM), rows)
    istft_kernel[grid](
        real,
        imag,
        basis,
        squared_window,
        out,
        frames,
        samples,
        N_FFT=n_fft,
        HOP=hop_length,
        BINS=bins,
        BINS_BLOCK=triton.next_power_of_2(bins),
        BLOCK_SAMPLES=OUTPUT_SAMPLES_PER_PROGRAM,
    )
    return out
