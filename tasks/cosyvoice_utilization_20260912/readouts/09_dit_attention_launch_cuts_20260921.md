# Readout 09: launch cuts in the packed DiT's attention (2026-09-21)

Tree under test: `slice/cosyvoice-7-1-dit-attention-launch-cuts` dba376c51 on upstream main
fbb486a2e (the DiT weight precast is in main). Raw runs: `artifacts/cosyvoice-4090-20260918/s7-r2`
(main baseline with the call ledger, GIL capture), `s8` inside it (probe), `s9-r1` (serving A/B).
RTX 4090 D, default launch plus `--tts_engine.engine.mem_fraction_static 0.28`.

## 1. Where main stands at c16 streaming (call ledger, 1,088 requests, seed 1234)

3.663 req/s with the ledger attached. 211 hop calls and 126 finals. A hop call is 13 rows and
4,300 frames at the median (16 rows and 6,350 at p95) and takes 611 ms of host time for 619 ms of
GPU time: at c16 a Flow call is device bound, the launch floor is a c1 and small step matter. A
scheduler step is 870 ms at the median. HiFT runs 3,204 times, one row at a time, 16 ms at the
median over a row's whole mel history.

So at c16 the device time of a call decides, and readout 08 has its split: matmul 43 %, pointwise
38 %, FA3 13 %.

## 2. What the attention paid for (source, x_transformers 2.28.4)

```
PackedDiT.attend, per block, per Euler step            packed_dit.py:268
  to_q(x), to_k(x), to_v(x)     x is the float32 norm output: three casts of the same tensor
  apply_rotary_pos_emb(q)       cos(freqs), sin(freqs) again, * scale (1.0) twice,
  apply_rotary_pos_emb(k)       rotate_half, then cat((rotated 64 dims, other 960 dims)) in
                                float32 and a cast of all 1024 back to bfloat16
```

About 11 kernels per application, 22 blocks, 10 steps, q and k. The DiT rotates only the first 64
of the 1024 flattened head dims, yet the whole tensor was rebuilt in float32 and cast back.

The slice casts the norm output once, takes cos and sin once per Euler step in `rope`, and rotates
the 64 dims in place with the same float32 arithmetic in the same order.

## 3. Probe (`stage2/s8_dit_exact_cuts.py`, one process, serving entry points)

Every row: repeatable, bit identical to main, max abs diff 0.

| call | kernels | kernel ms | wall ms |
|---|---|---|---|
| hop, 1 row, 50 frames | 14,998 to 11,938 | 63.5 to 58.0 | 186 to 173 |
| hop, 4 rows, 1,096 frames | 14,004 to 10,944 | 285 to 274 | 279 to 265 |
| hop, 8 rows, 5,200 frames | 13,920 to 10,860 | 1,088 to 972 | 1,078 to 961 |
| hop, 16 rows, 5,920 frames | 12,982 to 9,922 | 1,482 to 1,306 | 1,495 to 1,317 |
| final, 16 rows, 6,016 frames | 12,588 to 9,528 | 1,445 to 1,263 | 1,454 to 1,270 |

3,060 kernels fewer per call: 2,620 from RoPE, 440 from the shared cast. 11 to 13 % of the device
time of a large call, which is the wide float32 concat and cast.

## 4. Serving A/B (`stage2/ab_pair.sh`: both arms at the same time, cards 4 and 5, seed 1234)

| point | metric | main | slice | delta |
|---|---|---|---|---|
| streaming c16 | req/s | 3.731 | 4.020 | +7.7 % |
| | audio s/s | 17.37 | 18.73 | +7.8 % |
| | RTF mean | 0.977 | 0.901 | -7.7 % |
| | latency mean / p95 / p99 s | 4.27 / 5.38 / 6.55 | 3.96 / 5.07 / 6.12 | -7.2 / -5.8 / -6.6 % |
| | TTFP mean / p95 s | 1.96 / 2.56 | 1.80 / 2.41 | -8.3 / -5.8 % |
| | inter chunk mean s | 1.188 | 1.111 | -6.5 % |
| | peak MiB | 16,942 | 16,480 | |
| streaming c1, 64 samples | req/s | 0.955 | 1.088 | +13.9 % |
| | RTF mean | 0.238 | 0.210 | -11.9 % |
| | latency mean / p99 s | 1.05 / 1.97 | 0.92 / 1.69 | -12.2 / -14.6 % |
| | TTFP mean / p95 s | 0.537 / 0.680 | 0.476 / 0.561 | -11.4 / -17.4 % |
| buffered c16 | req/s | 6.159 | 6.199 | +0.6 % |
| | RTF mean | 0.580 | 0.576 | -0.7 % |

No failures anywhere. Buffered is flat because it replays the padded whole solver graph through
the dense DiT, which the slice does not touch.

## 5. Identity in serving (seeded c1, two boots per tree)

The two main boots disagree on 8 of 64 samples (the per voice boot instability, still open). Of
the 56 gated samples the slice's first boot is byte identical on 55 and its second on 53. All four
differing files differ in length, which is the AR's token count, and they fall on two voices
(`103675`, `17147545`); no sample of equal length differs by a byte. A Flow change cannot move the
token count, and a Flow difference would change every file, so the vocoder path is bit identical
in serving too. One control boot is too few to call a voice stable.

## 6. Next, by what section 1 says

- bfloat16 activations between modules: the rest of the float32 pointwise work and the last cast
  per Linear. Changes numerics: the float32 truth harness of `s1_hop_cache_gate.py` first.
- One fused in place RoPE kernel (2 kernels instead of 12 per block and step): numerics change too.
- The time modulation of every block is a constant of the step: about 1,100 tiny kernels a call,
  a c1 gain, exact if the constants come from the same ops.
- HiFT is about 250 ms of an 870 ms step at c16, row by row over whole histories: plan 04.
- The GIL capture (`s7-r2`, trace on the box) is not analysed yet.
