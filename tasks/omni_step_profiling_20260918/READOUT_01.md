# Readout 01: first live step captures, Qwen3-TTS 1.7B Base, moss GPU 6

Run: 2026-09-18, one boot of upstream 144bd6399 + `patches/step_profiler.patch`, RTX 4090 D
(sm89), streaming, seed-tts en. Traces and ledgers stay on the box under
`/workspace/sglang-omni/.tmp/omni_step_profiling/run01/`.

## Checks

| check | result |
| --- | --- |
| C1 window exact | decode b16 40 spans fwd 566..605 contiguous (38 at bs 16, the last 2 at bs 15 as one request ended); decode b1 40; prefill b1 10; prefill b16 4 coalesced extends (bs 1, 8, 4, 3) |
| C3 owners | 0 unknown, 0 cpu-op vs launch-thread mismatches in every trace; the two `streaming_vocoder.py` threads own their work |
| C4 overhead | client ms per frame, decode b16: unprofiled 12.70 and 12.95, formal capture 12.53, inside the unprofiled spread. The with_stack mapping run: 59.1, so mapping traces name kernels and never time them |
| C5 ledger vs BBuf | not run: cloning the BBuf repo in the container was blocked by the permission classifier |

Defects seen:

- `Failed to export trace: Trace is already saved` after every stop. On torch 2.13 `profile.stop()`
  fires `on_trace_ready`, which exports and gzips; `TorchProfiler.stop` then exports again
  and fails. The trace is correct. Present on main.
- kineto logs `External init callback must run in same thread as registerClient` once, when the
  window starts on the scheduler thread. Coverage is complete (C3), so it is noise here.

## Decode, bs 16 (38 steady steps)

| read | p50 | p95 | mean |
| --- | ---: | ---: | ---: |
| step wall ms | 11.23 | 25.37 | 13.29 |
| device busy ms, all threads | 11.11 | 24.80 | 12.92 |
| device idle ms | 0.12 | 1.21 | 0.37 |
| scheduler-owned device busy ms | 11.11 | 15.67 | 11.55 |
| scheduler blocked in sync calls ms | 3.16 | 12.65 | 4.16 |
| kernels per step | 1314 | 1314 | 1314.3 |

- The step is device bound on this card: idle is 0.12 ms of 11.23 at p50, and the scheduler
  spends 3.2 ms blocked in `cudaEventSynchronize` waiting for the device.
- Per step: 2 graph replays (talker 367 kernels, predictor 902), 38 `cudaLaunchKernel`,
  6 driver launches, 14 memcpy (11 DtoD, 2 HtoD, 1 DtoH), 1 event sync.
- p95 steps (25 ms) are the steps a vocoder decode lands in: the two vocoder threads add
  9 to 10 ms of device time each on those steps, serialized with the talker.

GPU time in the window (553 ms, 57,623 events):

| kernel family | ms | share | per step |
| --- | ---: | ---: | ---: |
| cutlass_80_wmma_tensorop_bf16_s161616gemm (one kernel) | 341.4 | 61.7% | 472 launches, 8.98 ms, 19.0 us each |
| vocoder convs: implicit_convolve_sgemm, precomputed_convolve_sgemm, cudnn fprop/dgrad, xmma | about 55 | about 10% | on vocoder steps |
| cudnn nchwToNhwc / nhwcToNchw transposes | 11.6 | 2.1% | on vocoder steps |
| _seeded_top_k_top_p_sample_kernel | 23.0 | 4.2% | 15 launches, 40 us each |
| predictor SDPA flash_fwd | 18.7 | 3.4% | 80 launches, 6.2 us |
| talker flashinfer BatchDecode | 11.9 | 2.2% | 28 launches |

## Decode, bs 1 (39 steady steps)

Step wall p50 8.84 ms, device busy 8.71, idle 0.12. One kernel,
`internal::gemvx::kernel`, takes 72.6% (17,855 launches, 6.72 ms per step, 14.7 us each).
Same weights as bs 16, read once per step either way.

GEMM time per step: bs 1 6.72 ms (gemv), bs 16 8.98 ms (wmma). For decode GEMMs that stream
the weights once per step, bs 16 should cost about what bs 1 costs; the extra 2.26 ms per
step (20 percent of the bs 16 step) is the kernel cuBLAS picks for M = 16 on sm89.

## Prefill, bs 1 (9 steady extends)

Span 25.9 ms per extend, scheduler device busy 10.4 ms: the extend is host bound. Per extend:
197 `cudaLaunchKernelExC`, 156 `cudaLaunchKernel`, 88 + 32 driver launches, 23 HtoD, 4 DtoH,
4 `cudaStreamSynchronize`, 56 memsets, 1 graph replay (902 kernels, the predictor).
"Wall" for serial prefill includes the gap to the next request and is not a step cost.
Outside the scheduler, thread 587083 owns 4,175 kernels per request (22.8 ms of device time),
plus 8 threads with about 290 kernels each and thread 587063. These are the preprocessing
side; a with_stack prefill capture names them.

## What transfers to the H100 and what does not

Transfers: kernel and launch counts per step, 2 replays with 367 / 902 kernels, the syncs,
the prefill launch counts, the vocoder transposes (a layout property). Does not transfer:
the device-bound verdict (H100 bandwidth is about 3.3x), the GEMM kernel cuBLAS picks for
M = 16 (sm89 wmma here, other kernels on sm90), and the conv algorithms cuDNN picks.

## Split by graph replay and the bandwidth floor (ledger2, config.json)

Dims: talker hidden 2048, 28 layers, intermediate 6144, 16 q / 8 kv heads x 128; predictor
hidden 1024, 5 layers, intermediate 3072, 16 q / 8 kv heads x 128, 16 code groups (15
predicted per step), per-group vocab 2048. Weights read per step, bf16:

- talker: 28 x (qkv 2048x4096 + o 2048x2048 + gate_up 2048x12288 + down 6144x2048)
  = 1.409 G params = 2.82 GB, 113 GEMMs (28 x 4 + codec head);
- predictor: 15 sub-steps x 5 layers x (qkv 1024x4096 + o 2048x1024 + gate_up 1024x6144
  + down 3072x1024) = 1.18 G param reads = 2.36 GB (157 MB per sub-step, over the 72 MB
  L2, so re-read from DRAM each sub-step), 352 GEMMs.

Floor at the 4090 D's 1008 GB/s: talker 2.80 ms, predictor 2.34 ms (heads excluded).

| replay | kernels | GEMM ms bs 1 | GEMM ms bs 16 | floor ms | bs 16 over floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| #0 talker | 339 / 367 | 3.26 (113 gemv, 28.8 us) | 4.00 (113 wmma, 35.4 us) | 2.80 | 1.20 |
| #1 predictor | 902 | 3.53 (352 gemv, 10.0 us) | 4.92 (352 wmma, 14.0 us) | 2.34 | 2.58 |

Predictor replay extras at bs 16: 15 `_seeded_top_k_top_p_sample_kernel` at 40.4 us (0.61 ms,
the same at bs 1), 80 SDPA flash (0.49 ms). Replay sums: talker 4.29 ms, predictor 6.58 ms.

The predictor is the larger lever and it is omni code (`models/qwen3_tts/sglang_model.py`,
`predictor_kernels.py`); the talker's linears are SGLang's.

## Prefill threads named (with_stack prefill capture, 4 steady extends)

| thread | frames | device ms per request |
| --- | --- | ---: |
| scheduler | talker prefill + predictor replay | 10.45 |
| vocoder decode | `incremental_codec_cuda_graph.py`, `codec_state_arena.py` | 24.5 |
| reference encode | `request_builders.py:_encode_waveform`, `reference_encoder_cuda_graph.py` | 3.8 |
| preprocessing | `modeling_qwen3_tts.py:266 forward`, `pipeline_state.py` | 1.0 |

The largest device cost on the first-audio path is the vocoder's first decode, not the
talker prefill: 24.5 ms per request, led by `implicit_convolve_sgemm` (198 calls, 147 to
229 us) and about 90 NCHW/NHWC transposes per request. The earlier "thread 587083" is this
vocoder thread.

## Next reads before any slice

1. Model dims from the checkpoint config, to put a bytes-per-step floor under the GEMM time
   (the bs 1 gemv gives the measured floor today).
2. Ledger: device time and top kernels per graph replay (talker vs predictor), so the
   8.98 ms of GEMM splits by owner.
3. A with_stack prefill capture to name thread 587083 and the 8-thread pool.
4. Vocoder: which conv modules pick `implicit_convolve_sgemm` and why the NCHW/NHWC
   transposes appear, read from the code path and a with_stack trace.
5. C2 and C6 in an Nsight runbook (graph node counts, ad10x metrics in the container).
