# Does a cuFFT plan creation wait for queued device work? HiFT's stft and istft shapes (n_fft 16, hop 4, a Hann
# window, float32), timed on the host per call with no synchronize: the device idle, the device busy on the same
# stream, the device busy on another stream, each with a new shape (a plan cache miss) and with the shape just used
# (a hit). Busy is torch.cuda._sleep for about BUSY_MS. Run under nsys to count cuModuleLoadData per case.
# usage: python3 cufft_plan_wait_c20261013.py
import time

import torch

BUSY_MS = 50.0
N_FFT = 16
HOP = 4

device = torch.device("cuda:0")
window = torch.hann_window(N_FFT, device=device)
other = torch.cuda.Stream(device=device)
cycles_per_ms = None


def calibrate():
    global cycles_per_ms
    torch.cuda._sleep(1_000_000)
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda._sleep(100_000_000)
    torch.cuda.synchronize()
    cycles_per_ms = 100_000_000 / ((time.perf_counter() - start) * 1e3)


def busy(stream):
    with torch.cuda.stream(stream):
        torch.cuda._sleep(int(BUSY_MS * cycles_per_ms))


def stft(rows, samples):
    signal = torch.randn(rows, samples, device=device)
    torch.cuda.synchronize()
    return signal


def timed(label, operation):
    start = time.perf_counter()
    operation()
    host_ms = (time.perf_counter() - start) * 1e3
    torch.cuda.synchronize()
    print(f"{label:60s} host {host_ms:8.2f} ms, plans cached {torch.backends.cuda.cufft_plan_cache[0].size}", flush=True)


def forward(signal):
    return lambda: torch.stft(signal, N_FFT, HOP, N_FFT, window=window, return_complex=True)


def inverse(spectrum):
    return lambda: torch.istft(spectrum, N_FFT, HOP, N_FFT, window=window)


calibrate()
print(f"cycles per ms {cycles_per_ms:.0f}, torch {torch.__version__}, cuda {torch.version.cuda}")
torch.backends.cuda.cufft_plan_cache[0].clear()
samples = 48_000
for rows, case in ((3, "idle"), (4, "same stream busy"), (5, "other stream busy")):
    signal = stft(rows, samples + 480 * rows)
    if case == "same stream busy":
        busy(torch.cuda.current_stream(device))
    elif case == "other stream busy":
        busy(other)
    else:
        pass
    timed(f"stft rows {rows}, new shape, {case}", forward(signal))
    if case == "same stream busy":
        busy(torch.cuda.current_stream(device))
    elif case == "other stream busy":
        busy(other)
    else:
        pass
    timed(f"stft rows {rows}, same shape, {case}", forward(signal))
    spectrum = torch.stft(signal, N_FFT, HOP, N_FFT, window=window, return_complex=True)
    torch.cuda.synchronize()
    if case == "same stream busy":
        busy(torch.cuda.current_stream(device))
    elif case == "other stream busy":
        busy(other)
    else:
        pass
    timed(f"istft rows {rows}, new shape, {case}", inverse(spectrum))
    if case == "same stream busy":
        busy(torch.cuda.current_stream(device))
    elif case == "other stream busy":
        busy(other)
    else:
        pass
    timed(f"istft rows {rows}, same shape, {case}", inverse(spectrum))
with torch.cuda.stream(other):
    torch.cuda._sleep(1000)
busy(torch.cuda.current_stream(device))
timed("a plain new kernel shape (matmul 37x41), same stream busy", lambda: torch.randn(37, 41, device=device) @ torch.randn(41, 43, device=device))
