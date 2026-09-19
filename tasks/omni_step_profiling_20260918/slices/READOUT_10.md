# Run 10 readout: live windows and whole-serve Nsight on main after PF

moss card 1 alone (another user's jobs on cards 0 and 2 from 07:55), RTX 4090 D,
Qwen3-TTS-12Hz-1.7B-Base, main `9e482b027` (profiler tree `4b1bfe7f5` = main + profiler).
Scripts: `run_live_boot.sh`, `live_window.py`, `window_budget.py`, `stall_anatomy.py`,
`host_step_stacks.py`, `thread_stacks.py`, `run_nsys_boot.sh`, `nsys_metrics.py`. Text
outputs: `artifacts/moss_omni_step_profiling/omni_step_profiling/run10/`; traces (1.2 GB),
nsys-rep and sqlite stay on the box.

## 1. Method and validity

- Degenerate captures (prefill-only b1, b16; decode-only b1, b16), then live cells: the
  seed-tts benchmark itself is the load; a torch window of scheduler forwards is armed
  15 s after the timed requests start. Three passes per cell: request events only (the
  unprofiled client numbers), formal window (timing), with-stack window (names only).
- Nsight wraps the whole serve; the window is the benchmark's timed requests cut from
  bench.log; GPU metrics ad10x at 2 kHz.
- Cross-check, seed-tts c16: device empty 16.1% (torch window, 11.1 s) vs 16.3% (Nsight,
  31.8 s); device time by thread 65.1 / 21.9 / 12.2 / 11.9 / 2.6% vs 65.6 / 21.8 / 12.3 /
  11.9 / 2.7%. Two tools, two boots, same shares.
- Overhead: unprofiled 14.35 req/s; Nsight 12.65 (-12%); the torch window runs about 18.5
  ms per step against about 16 unprofiled. Shares are quoted, never absolute times.
- The long-form bench lasted 20 s, so its 400-step window was closed by hand and holds
  the last 5 s (batch 16 draining to 1). Its Nsight window covers the whole bench.
- Cards differ: run 09 dmon shows card 4 at a 1.7% higher core clock than card 3, no
  throttle flags. A slice under about 3% needs a same-card or swapped-card A/B.

## 2. Device time by owner (share of the window; streams overlap, so shares exceed 100%)

| owner | seed-tts c16 | long-form c16 (Nsight, whole bench) | seed-tts c1 |
| --- | ---: | ---: | ---: |
| scheduler (talker, predictor, prefill) | 65.1% | 79.3% | 85.5% |
| vocoder initial worker (reference-prefixed bootstraps) | 21.9% | 3.5% | 3.5% |
| vocoder follow-up workers (2) | 24.1% | 22.9% | 6.0% |
| reference encoder | 2.6% | 0.3% | about 0 |
| device empty | 16.1% | 10.7% | 11.9% (gaps between requests) |
| SM Issue / SMs Active / DRAM read (Nsight) | 12.5 / 62.6 / 34.0% | 9.8 / 65.5 / 42.2% | |

Scheduler loop wall, seed-tts c16: decode 78.8%, prefill 21.2%. A decode step at batch
14 to 16 costs 16 to 18 ms mean (11.1 alone) with 8.5 ms of other-thread device time
inside it; a one-sequence prefill step 21.6 ms mean with 7.5 ms of scheduler device time
and 8.8 ms device empty.

## 3. Kernels (Nsight, seed-tts c16, 26.6 s of kernel time)

- One GEMM kernel, the cuBLAS pick for batch above 1
  (`cutlass_80_wmma_tensorop_bf16_s161616gemm`), 14.57 s = 55%, SM Issue 10.9, DRAM read
  46.6: bandwidth-bound, few SMs issuing. It is why the window's SM Issue is low.
- Vocoder convs run at SM Issue 38 to 43 (`implicit_convolve_sgemm` 2.30 s), the cuDNN
  layout transposes 1.38 s, eager elementwise kernels about 1.8 s.
- Seeded sampler 1.14 s = 4.3% (5.4 to 5.5% of scheduler device time in both live
  windows).

## 4. Device-empty time, seed-tts c16

- Gaps of at least 1 ms are 2.6% of the window; gaps of at least 50 us are 11.6%: the
  empty time is thousands of short gaps, half in prefill steps, half in decode steps.
- Scheduler thread during those gaps: nothing on it 32%, inside `cudaGraphLaunch` 21%,
  `cudaLaunchKernel` 8.5%, the eager attention op 8%, `cudaMemcpyAsync` 7%.
- In the same gaps the 8 preprocessing threads (`pipeline_state.py`) ran 4,500 to 8,600
  host events each. #1923 measured this contention on H200 and moved preprocessing to its
  own process as an opt-in; it costs 2.2 GB, so it is not the default, and not available
  on 24 GB.
- Prefill alone (formal b1): span 13.6 ms for 10.3 ms of device work: 28 eager attention
  calls between 30 graph segments bound the step on the host.

## 5. Request timeline, seed-tts c16 unprofiled (1,089 requests)

preprocessing 38.6 ms avg (p95 67.2); request build to queue 13.6 ms; queue to prefill
6.5 ms; prefill 14.5 ms; TTFC mean 125 ms.

## 6. What this means for the next slice

1. Under real load the vocoder takes 46% of device time on seed-tts and 26% on long-form.
   The Base bootstrap alone is 21.9% on seed-tts: cohorts key on total width, reference
   lengths differ, so every Base bootstrap decodes alone, as a binary sequence of windows
   (a 63-frame prefix is 32+16+8+4+2+1, about 22 ms, against 10.6 ms for one 64-wide
   decode). Bootstrap cost is a short-output effect (seed-tts mean 52 frames); follow-up
   chunk cost is the same share in both workloads.
2. The decode GEMMs are bandwidth-bound and 55% of kernel time; nothing bit-identical
   shortens them. Batch is what raises SM Issue there, and the vocoder's share is what
   limits batch.
3. Bit-identical candidates: host time in prefill steps and the interpreter contention
   from preprocessing (device empty 16%); the seeded sampler noise (P2, 4.3% of kernel
   time on this card, fp64-rate dependent).
4. Vocoder changes never alter codes (the vocoder does not feed the talker), so they
   cannot flip a token; they move waveform samples at rounding level. Whether that
   counts as "same behaviour" is the user's call before V2 or a bootstrap redesign.
