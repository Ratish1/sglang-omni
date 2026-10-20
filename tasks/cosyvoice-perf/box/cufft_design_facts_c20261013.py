# Facts for HiFT's FFTs without a cuFFT plan per step (n_fft 16, hop 4, a Hann window, float32, centred):
# 1. bucketing: is a row's torch.stft and torch.istft bit identical when its batch has more rows and more frames
#    (zero padded), against the row in its own exact shape?
# 2. the device memory the plan cache holds per plan (cudaMemGetInfo around 64 new plans of each kind),
# 3. a 16 point DFT as a float32 matmul over unfolded frames: its largest difference from torch.stft and torch.istft,
#    and whether a new shape waits for a busy device (host time per call with 50 ms of queued work).
# usage: python3 cufft_design_facts_c20261013.py
import math
import time

import torch
import torch.nn.functional as F

N_FFT = 16
HOP = 4
device = torch.device("cuda:0")
window = torch.hann_window(N_FFT, device=device)
torch.manual_seed(0)


def stft(signal):
    return torch.stft(signal, N_FFT, HOP, N_FFT, window=window, return_complex=True)


def istft(spectrum):
    return torch.istft(spectrum, N_FFT, HOP, N_FFT, window=window)


print(f"torch {torch.__version__}, matmul tf32 {torch.backends.cuda.matmul.allow_tf32}, cudnn tf32 {torch.backends.cudnn.allow_tf32}")

# 1. bucketing
rows, samples = 3, 24_000
signal = torch.randn(rows, samples, device=device) * 0.1
exact = stft(signal)
padded_signal = torch.zeros(8, samples + 4_000, device=device)
padded_signal[:rows, :samples] = signal
padded = stft(padded_signal)
frames = exact.shape[-1]
interior = frames - 4
same_interior = torch.equal(exact[..., :interior], padded[:rows, :, :interior])
max_diff = (exact[..., :interior] - padded[:rows, :, :interior]).abs().max().item()
print(f"1. stft row in a padded batch: interior frames identical {same_interior}, max diff {max_diff:.3e}")
back_exact = istft(exact)
padded_spectrum = torch.zeros(8, exact.shape[1], frames + 1000, dtype=exact.dtype, device=device)
padded_spectrum[:rows, :, :frames] = exact
back_padded = istft(padded_spectrum)
keep = back_exact.shape[-1] - 64
same_back = torch.equal(back_exact[:, :keep], back_padded[:rows, :keep])
print(f"1. istft row in a padded batch: interior samples identical {same_back}, max diff "
      f"{(back_exact[:, :keep] - back_padded[:rows, :keep]).abs().max().item():.3e}")

# 2. memory per plan
torch.cuda.synchronize()
torch.backends.cuda.cufft_plan_cache[0].clear()
torch.cuda.empty_cache()
free_before, _ = torch.cuda.mem_get_info(device)
allocated_before = torch.cuda.memory_allocated(device)
for index in range(64):
    spectrum = stft(torch.zeros(1 + index % 16, 4_800 + 480 * index, device=device))
    istft(spectrum)
torch.cuda.synchronize()
torch.cuda.empty_cache()
free_after, _ = torch.cuda.mem_get_info(device)
plans = torch.backends.cuda.cufft_plan_cache[0].size
print(f"2. {plans} plans cached, device memory outside the caching allocator "
      f"{((free_before - free_after) - (torch.cuda.memory_allocated(device) - allocated_before)) / 2**20:.1f} MiB, "
      f"{((free_before - free_after) / plans) / 2**10:.0f} KiB per plan")

# 3. DFT as a matmul
bins = N_FFT // 2 + 1
n = torch.arange(N_FFT, device=device, dtype=torch.float64)
k = torch.arange(bins, device=device, dtype=torch.float64)
angle = 2 * math.pi * k[:, None] * n[None, :] / N_FFT
forward_basis = torch.cat([torch.cos(angle), -torch.sin(angle)]) * window.double()[None, :]
scale = torch.full((bins,), 2.0, device=device, dtype=torch.float64)
scale[0] = 1.0
scale[-1] = 1.0
inverse_basis = torch.cat([torch.cos(angle) * scale[:, None], -torch.sin(angle) * scale[:, None]]) / N_FFT
inverse_basis = inverse_basis * window.double()[None, :]
forward_basis = forward_basis.float()
inverse_basis = inverse_basis.float()


def stft_matmul(signal):
    padded = F.pad(signal[:, None], (N_FFT // 2, N_FFT // 2), mode="reflect")[:, 0]
    frames = padded.unfold(-1, N_FFT, HOP)
    spectrum = frames @ forward_basis.T
    return torch.complex(spectrum[..., :bins], spectrum[..., bins:]).transpose(1, 2)


def istft_matmul(spectrum):
    parts = torch.cat([spectrum.real, spectrum.imag], dim=1).transpose(1, 2)
    frames = parts @ inverse_basis
    frame_count = frames.shape[1]
    length = N_FFT + HOP * (frame_count - 1)
    signal = torch.ops.aten.unfold_backward(frames, [frames.shape[0], length], 1, N_FFT, HOP)
    envelope = torch.ops.aten.unfold_backward(
        window.pow(2).expand(1, frame_count, N_FFT), [1, length], 1, N_FFT, HOP
    )
    trim = N_FFT // 2
    return signal[:, trim:-trim] / envelope[:, trim:-trim]


signal = torch.randn(5, 30_000, device=device) * 0.1
reference = stft(signal)
candidate = stft_matmul(signal)
print(f"3. stft matmul against torch.stft: max abs diff {(reference - candidate).abs().max().item():.3e}, "
      f"max abs {reference.abs().max().item():.3e}")
back_reference = istft(reference)
back_candidate = istft_matmul(reference)
print(f"3. istft matmul against torch.istft: max abs diff {(back_reference - back_candidate).abs().max().item():.3e}, "
      f"max abs {back_reference.abs().max().item():.3e}")

torch.cuda._sleep(1_000_000)
torch.cuda.synchronize()
start = time.perf_counter()
torch.cuda._sleep(100_000_000)
torch.cuda.synchronize()
cycles_per_ms = 100_000_000 / ((time.perf_counter() - start) * 1e3)
other = torch.cuda.Stream(device=device)
waits = []
for index in range(24):
    signal = torch.randn(1 + index % 16, 9_000 + 1_234 * index, device=device)
    torch.cuda.synchronize()
    with torch.cuda.stream(other):
        torch.cuda._sleep(int(50 * cycles_per_ms))
    start = time.perf_counter()
    istft_matmul(stft_matmul(signal))
    waits.append((time.perf_counter() - start) * 1e3)
    torch.cuda.synchronize()
print(f"3. matmul stft and istft, 24 new shapes beside 50 ms of queued work: host ms per call "
      f"{', '.join(f'{w:.1f}' for w in waits)}")
