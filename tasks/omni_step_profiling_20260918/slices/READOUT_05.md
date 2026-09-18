# Run 05 readout: P1-e1b, V1-e5 (vocoder time by kernel category), V2-e1 (compile coverage)

moss card 0, RTX 4090 D, torch 2.13.0+cu130, cuDNN 9.20. Scripts md5 in
`run05/md5.txt`. Outputs: `artifacts/moss_omni_step_profiling/omni_step_profiling/run05/`
(tgz md5 d1451fa6b9fdf09f8a5fce468e09480f). In `v1e5_split.txt` the profiler's USDT lines
interleave with stdout; one category line (current conv at w32 b4) was overwritten and
is derived below by subtraction.

## 1. P1-e1b: the base chain at bs 32 (M = 32 in the first pass)

Sub-step 0 logits against fp32: base bs 32 mean abs 3.0837e-02, max 0.2307 (320 rows).
The pair pass at bs 16 (also M = 32) measured 3.0299e-02 (READOUT_04). The M = 32 error
belongs to the M = 32 GEMM kernel, which the base chain already runs at 32 concurrent
requests. With READOUT_04 section 3, P1-e1 passes at every batch size: the pair pass at
batch B carries the base chain's numerics at batch 2B.

## 2. V1-e5: one decode's device time by kernel category (us, one profiled replay)

| key | arm | replay ms | conv | transpose | elementwise | copy/cat | gemm | triton | kernels |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| w1 b1 | current | 2.600 | 792 | 382 | 899 | 241 | 295 | | 1043 |
| w1 b1 | resident | 2.681 | 466 | 10 | 897 | 319 | 290 | | 1057 |
| w1 b1 | compiled | 1.840 | 838 | 381 | | 11 | 293 | 136 | 327 |
| w8 b1 | current | 4.422 | 1350 | 435 | 1112 | 318 | 500 | | 1054 |
| w8 b1 | resident | 3.668 | 696 | 2 | 1056 | 693 | 470 | | 1066 |
| w8 b1 | compiled | 2.723 | 1424 | 429 | | 14 | 503 | 176 | 328 |
| w8 b8 | current | 10.686 | 4887 | 567 | 2684 | 702 | 789 | | 1109 |
| w8 b8 | resident | 10.917 | 2899 | 11 | 3026 | 3810 | 776 | | 1110 |
| w8 b8 | compiled | 7.639 | 5299 | 571 | | | 795 | 402 | 358 |
| w16 b2 | current | 5.291 | 1600 | 558 | 1691 | 462 | 595 | | 1114 |
| w16 b2 | resident | 6.782 | 1926 | 3 | 1797 | 2043 | 596 | | 1106 |
| w16 b2 | compiled | 3.555 | 1935 | 595 | | | 638 | 280 | 363 |
| w32 b4 | current | 23.104 | about 11700 (by subtraction) | 671 | 7863 | 1282 | 1206 | | 1138 |
| w32 b4 | resident | 23.163 | 5653 | 10 | 8148 | 7794 | 1198 | | 1146 |
| w32 b4 | compiled | 15.952 | 13132 | 719 | | | 1219 | 646 | 387 |
| w64 b1 | current | 10.953 | 6120 | 513 | 2578 | 621 | (not listed) | | 1073 |
| w64 b1 | resident | 10.906 | 2927 | 6 | 2984 | 3842 | 751 | | 1085 |
| w64 b1 | compiled | 8.723 | 6518 | 514 | | 15 | 765 | 384 | 355 |

Reads:

- The largest single cost at every size is cuDNN's legacy non-tensor-core direct
  convolution on NCL input: `implicit_convolve_sgemm` 0.48 ms (w8 b1), 2.07 ms (w8 b8),
  5.77 ms (w32 b4); plus `precomputed_convolve_sgemm` 0.75 ms (w8 b8), 3.07 ms (w32 b4).
  Compiling does not change them (inductor calls the same cuDNN conv).
- In channels-last the same convs run on tensor-core kernels (`sm80/sm86_xmma_fprop`,
  cutlass fprop): resident conv time is 40 to 52 percent below current at every key but
  w16 b2.
- The resident prototype loses that gain to copies: 14 to 16 conv calls return NCL
  outputs although their weights are channels-fastest: every 1x1 conv (`conv2` of each
  residual unit, the quantizer `output_proj`) and both depthwise convs. Their strided
  outputs feed the next cat or elementwise op and turn into the generic strided
  `direct_copy` kernel (3.03 ms at w8 b8, 3.8 ms copy/cat at w64 b1).
- Eager elementwise work (SnakeBeta and adds) is 24 to 34 percent of a current decode;
  compiled, it becomes 0.14 to 0.65 ms of fused Triton kernels.
- Compiled current still spends 0.38 to 0.72 ms per decode in cuDNN transposes.

## 3. V2-e1: every captured key compiled (ms per replay, median of 30)

Eager against a static compile per key (`dynamic=False`, the runner's width-8 setting)
and one dynamic compile shared by all keys (`dynamic=True`); each arm in its own
process. Full table: `v2e1_{eager,static,dynamic}.txt`.

| key | eager | static | dynamic | key | eager | static | dynamic |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| w1 b1 | 2.602 | 1.633 | 1.641 | w8 b8 | 9.597 | 7.326 | 7.706 |
| w1 b8 | 3.071 | 2.099 | 2.108 | w16 b1 | 4.452 | 3.347 | 3.274 |
| w2 b2 | 2.840 | 1.878 | 1.972 | w16 b8 | 22.825 | 15.725 | 16.921 |
| w4 b4 | 3.775 | 2.555 | 2.569 | w32 b2 | 7.657 | 5.528 | 5.756 |
| w5 b8 | 6.420 | 4.818 | 4.920 | w32 b8 | 50.059 | 32.102 | 33.741 |
| w8 b1 | 3.480 | 2.510 | 2.428 | w64 b1 | 10.575 | 8.257 | 8.845 |
| w8 b4 | 5.726 | 4.286 | 4.357 | w64 b8 | 105.304 | 65.962 | 69.854 |

- Compiling cuts 20 to 40 percent at every key; dynamic is within 0 to 8 percent of
  static (slower at the largest keys).
- Startup: static costs about 20 s of compile and 17 s of capture per key (44 keys, about
  28 minutes); dynamic compiled 4 times (the first two widths at batch 1 and 2, 31 to
  76 s each) and captured in 20 to 55 s for the first 10 keys, then under a second:
  about 3.5 minutes of compile plus capture in all.
- The super-linear step at 128 frame-rows remains compiled (w16 b4 6.78 to w16 b8
  15.73 ms), so it is not only the eager elementwise temporaries; the conv time carries
  it (w32 b4 compiled: conv 13.1 of 15.7 ms).

## 4. What this changes

The vocoder slice becomes one design with three measured parts, to be tested together:
dynamic compile over every captured width (V2), channels-last for the tensor-core convs
with the 1x1 convs as matrix multiplies on (B, T, C) and the depthwise conv's layout
chosen by measurement (V1), under that compile. Next experiment V-e6 (script
`vocoder_resident_bench.py` extended): that chain, eager and dynamic-compiled, every
captured key, timing, kernel split and V1-e2 numerics.
