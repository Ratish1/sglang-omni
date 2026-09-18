# Run 03 readout: V1-e0 to V1-e4 and P2-e1

moss card 0, RTX 4090 D, torch 2.13.0+cu130, cuDNN 9.20, triton 3.7.1. Tree
144bd6399 (`.tmp/wt/step-prof`). Scripts md5 `vocoder_resident_bench.py`
b01081bd2b2c134dcdbee2c1fd218ddf, `sampler_stage_bench.py` cbbe4d4e7351520cb095ea197d3f6e9c.
Full outputs: `artifacts/moss_omni_step_profiling/omni_step_profiling/run03/` (tar md5
cbd3fb9aa9dbf5228e6f517226c5c5a9). Every number below is read from those files.

## 1. P2-e1: the sampler's 43 us is the fp64 Gumbel noise

us per call, 15 calls per CUDA graph, median of 50 replays. Stage n includes stages 1..n.

| stage | bs 1 w1 | w2 | w4 | w8 | bs 16 w1 | w2 | w4 | w8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 load, scale | 1.43 | 1.43 | 1.37 | 1.37 | 1.43 | 1.37 | 1.37 | 1.37 |
| 2 + pack, topk | 32.09 | 17.13 | 7.78 | 6.76 | 31.06 | 16.38 | 7.71 | 6.76 |
| 3 + unpack, softmax, log | 32.63 | 18.09 | 8.26 | 7.37 | 32.09 | 17.34 | 8.19 | 7.44 |
| 4 + hash | 33.11 | 18.57 | 8.47 | 7.78 | 32.43 | 17.75 | 8.46 | 7.78 |
| 5 + fp64 Gumbel, argmax (full) | 42.12 | 29.29 | 26.62 | 42.94 | 41.37 | 28.60 | 24.10 | 39.46 |
| 5 with fp32 Gumbel (attribution) | 34.47 | 19.93 | 9.28 | 8.53 | 31.13 | 17.61 | 8.40 | 7.92 |

Shipped kernel: 42.94 us (bs 1), 43.01 us (bs 16). The stage 5 copy reproduced the
shipped tokens at every width (assert passed).

- At the shipped `num_warps=8`, the fp64 Gumbel adds 35.2 us (bs 1) and 31.7 us (bs 16)
  on top of stage 4's 7.8 us. The fp32 copy adds 0.8 us.
- The fp64 cost grows with the warp count (w1 9.0 us, w8 35.2 us at bs 1): every warp
  evaluates the same 64 fp64 logs, and they queue on one SM's fp64 units (sm89 runs fp64
  at 1/64 of fp32). Selection alone at w8 is 6.8 us.
- The hypothesis in P2_SAMPLER_LATENCY.md section 3 holds. Design: P2 file section 5
  (updated).

## 2. V1-e0: dispatch of the resident chain

state_spec: 29 conv histories, 6 transconv overlaps, 8 layers x 2 K/V, retained 71.
Channels-fastest weight copies: 39 modules, 111.8 MiB (37 with the depthwise conv NCL,
same MiB to one decimal).

| bs | kernels current (transposes) | kernels resident (transposes) | conv outputs not channels-last | max abs vs current |
| ---: | --- | --- | ---: | ---: |
| 1 | 1054 (42) | 1066 (1) | 16 | 1.465e-02 |
| 8 | 1109 (52) | 1110 (5) | 16 | 4.883e-03 |

Depthwise NCL arm: same kernel and transpose counts, 14 conv outputs not channels-last,
max abs 8.765e-02 (bs 1) and 6.836e-03 (bs 8). The resident chain removes 41 to 47
cuDNN transposes per decode but not the other 1,000; 16 conv calls still return
NCL outputs (which calls: not recorded; next run names them).

## 3. V1-e1: full decode per captured key (CUDA graph, ms, median of 30)

| width | batch | current | resident | delta | compiled current | compiled resident | delta | eager to compiled |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 2.601 | 2.657 | +2.2% | | | | |
| 1 | 2 | 3.387 | 2.958 | -12.7% | | | | |
| 1 | 4 | 3.359 | 3.357 | -0.1% | | | | |
| 1 | 8 | 3.546 | 3.882 | +9.5% | | | | |
| 2 | 1 | 3.435 | 2.932 | -14.6% | | | | |
| 2 | 2 | 3.550 | 3.292 | -7.3% | | | | |
| 2 | 4 | 3.807 | 3.780 | -0.7% | | | | |
| 2 | 8 | 4.354 | 5.291 | +21.5% | | | | |
| 3 | 1 | 3.554 | 2.991 | -15.8% | | | | |
| 3 | 2 | 3.582 | 3.487 | -2.7% | | | | |
| 3 | 4 | 4.087 | 4.432 | +8.4% | | | | |
| 3 | 8 | 5.230 | 5.985 | +14.4% | | | | |
| 4 | 1 | 3.656 | 3.182 | -13.0% | | | | |
| 4 | 2 | 3.811 | 3.773 | -1.0% | | | | |
| 4 | 4 | 4.495 | 5.093 | +13.3% | | | | |
| 4 | 8 | 6.015 | 6.787 | +12.8% | | | | |
| 5 | 1 | 3.757 | 3.289 | -12.5% | | | | |
| 5 | 2 | 3.903 | 3.979 | +1.9% | | | | |
| 5 | 4 | 4.762 | 5.194 | +9.1% | | | | |
| 5 | 8 | 6.850 | 7.804 | +13.9% | | | | |
| 6 | 1 | 3.870 | 3.411 | -11.9% | | | | |
| 6 | 2 | 4.064 | 4.344 | +6.9% | | | | |
| 6 | 4 | 5.197 | 5.926 | +14.0% | | | | |
| 6 | 8 | 7.625 | 8.701 | +14.1% | | | | |
| 7 | 1 | 4.011 | 3.486 | -13.1% | | | | |
| 7 | 2 | 4.119 | 4.531 | +10.0% | | | | |
| 7 | 4 | 5.608 | 6.429 | +14.6% | | | | |
| 7 | 8 | 8.598 | 9.632 | +12.0% | | | | |
| 8 | 1 | 4.182 | 3.698 | -11.6% | 2.725 | 2.028 | -25.6% | -34.8% |
| 8 | 2 | 4.666 | 4.974 | +6.6% | 2.676 | 2.991 | +11.8% | -42.6% |
| 8 | 4 | 6.210 | 6.809 | +9.6% | 4.436 | 4.466 | +0.7% | -28.6% |
| 8 | 8 | 10.084 | 10.940 | +8.5% | 7.628 | 6.827 | -10.5% | -24.4% |
| 16 | 1 | 5.126 | 4.793 | -6.5% | | | | |
| 16 | 2 | 5.302 | 6.803 | +28.3% | | | | |
| 16 | 4 | 9.313 | 10.866 | +16.7% | | | | |
| 16 | 8 | 23.320 | 22.926 | -1.7% | | | | |
| 32 | 1 | 6.685 | 6.693 | +0.1% | | | | |
| 32 | 2 | 8.029 | 11.020 | +37.3% | | | | |
| 32 | 4 | 23.164 | 23.204 | +0.2% | | | | |
| 32 | 8 | 50.628 | 49.972 | -1.3% | | | | |
| 64 | 1 | 10.992 | 10.913 | -0.7% | | | | |
| 64 | 2 | 17.244 | 23.337 | +35.3% | | | | |
| 64 | 4 | 50.461 | 50.369 | -0.2% | | | | |
| 64 | 8 | 105.796 | 104.060 | -1.6% | | | | |

The depthwise-NCL arm is within 0.6 percent of the resident arm at every key (raw file).
Kernel counts: eager 1,043 to 1,154 per decode; compiled width 8: 328 to 358 current,
311 to 350 resident. Compiled current still runs 42 to 60 cuDNN transposes.

Reads:

- The resident layout alone is not a win across workloads: 12 to 16 percent faster at
  batch 1 for widths 2 to 8, 7 to 37 percent slower at batch 2 to 8 for most widths,
  flat at the largest keys. Bench 02's per-call conv savings (conv calls timed alone,
  10 repeats in one graph) do not carry over to the chain.
- Compiling width 8 cuts 24 to 43 percent of the eager decode and 70 percent of the
  kernels, at every batch. Every other captured width replays eager (audit A2).
- The current path has a super-linear step at 128 frame-rows (batch x width): w32 b2
  8.0 ms to w32 b4 23.2 ms (2.9x for 2x rows), w16 b4 9.3 to w16 b8 23.3 (2.5x),
  w64 b1 11.0 to w64 b2 17.2, and w64 b8 105.8 ms. Arithmetic on the shapes: one
  96-channel activation at 128 frame-rows is 128 x 1920 samples x 96 x 2 B = 47 MB,
  against a 72 MB L2, and the eager SnakeBeta writes a full-size temporary per op.
  Hypothesis, unmeasured: past that size the eager elementwise chain runs from DRAM.
  This hits the bootstrap windows (widths 16 to 64) at batch 2 and up.

## 4. V1-e2: numerics on real codes (SNR to the fp32 decoder, dB, minimum over rows)

Codes: `tokenizer.encode` of 8 seed-tts reference clips; the list holds 4 distinct clips
twice (frames 49, 49, 82, 82, 96, 96, 75, 75: consecutive seed-tts samples share a
reference), so there are 4 distinct utterances.

| cohort | current | current compiled | resident | resident compiled |
| --- | ---: | ---: | ---: | ---: |
| utt 0/1 (49 frames) | 34.41 | 36.51 | 35.23 | 38.33 |
| utt 2/3 (82) | 35.37 | 39.60 | 35.56 | 38.53 |
| utt 4/5 (96) | 32.86 | 36.93 | 33.22 | 37.58 |
| utt 6/7 (75) | 36.07 | 37.99 | 36.59 | 39.57 |
| bs 4 cohort | 34.35 | 36.41 | 34.84 | 37.46 |

Max abs to the truth: 5.9e-03 to 6.4e-02 across arms (raw file). Resident eager is
closer to fp32 than current eager in every cohort; resident compiled is closer than
current compiled in 4 of 5. Compiled is closer to fp32 than eager in every cohort (the
eager chain rounds each of its many intermediate tensors to bf16).

## 5. V1-e4: non-streaming tokenizer.decode, weights re-laid in place

| audio | batch | NCL weights ms | channels-fastest ms | delta | waveform max abs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 s | 1 | 28.62 | 26.49 | -7.4% | 3.906e-03 |
| 10 s | 8 | 218.44 | 196.88 | -9.9% | 1.172e-02 |
| 30 s | 1 | 88.73 | 85.38 | -3.8% | 3.906e-03 |
| 30 s | 8 | 699.44 | 663.67 | -5.1% | 7.812e-03 |

Device timeline of the whole `tokenizer.decode` call (median of 30). Re-laying the conv
weights in place makes the non-streaming path faster at every size measured.

## 6. What changes in the plan

1. P2: confirmed; the noise does not depend on the logits, only on (seed, position,
   rank), so all 15 sub-steps' noise can be computed once per step by a grid of many
   programs, then read by the sampler. Design in P2_SAMPLER_LATENCY.md.
2. V1 is not designed from these numbers. Next: V1-e5, the kernel time of one replay
   split by category (conv compute, cuDNN transposes, elementwise, copies and cat,
   transformer, arena), current vs resident, at w1 b1, w8 b1, w8 b8, w16 b2, w32 b4,
   w64 b1, and the names of the 16 convs that return NCL.
3. V2 moves ahead of V1: compiling one width removes 70 percent of the kernels and 24 to
   43 percent of the time, and the 128 frame-row step points at elementwise traffic.
   V2-e1: the same category split, plus every captured width compiled (startup seconds,
   graph memory, ms per key), in the same run as V1-e5.
4. D1 (weight layout) is settled in favor of in place if V1 ships: non-streaming gets
   faster, no extra 111.8 MiB.
