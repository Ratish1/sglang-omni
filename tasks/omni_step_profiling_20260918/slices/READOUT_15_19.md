# Runs 15 to 19 readout: the fused snake activation, decided by measurement

moss RTX 4090 D, Qwen3-TTS-12Hz-1.7B-Base, main `ebd577ea0`. Branch
`perf/qwen3-tts-fused-snake` `6574bc743` (3 files, +39 -3): main's kernel, its launch
registered as an opaque custom op, `_MAX_T` set to the CUDA grid bound, the factory flag on
by default. The full rewrite is kept on `perf/qwen3-tts-fused-snake-rewrite-record`
(`7da30c7b0`). Outputs: `artifacts/moss_omni_step_profiling/omni_step_profiling/run15`,
`run16_19`.

## 1. Was main's kernel wrong

- Numerics: no. Bitwise equal to eager on all 65,536 bf16 encodings per channel count and
  on all 44 captured decode shapes (run 15, run 16).
- Launch layout: not for this decoder. SnakeBeta runs once at 1536 channels x 4 samples
  per frame, and seven times each at 768 x 32, 384 x 160, 192 x 640 and 96 x 1,920; the
  time is in the long stages, where one row per program with 1,024-element blocks is
  already SGLang's packed layout.
- Integration: yes. With `fused_snake_activation` on, the decoder's
  `torch.compile(fullgraph=True)` of the 8-frame shape raises `Unsupported: Attempted to
  inline function marked as skipped ... device_of.__init__` inside the Triton launcher;
  the graph runner catches it and disables itself for every width (3 of 4 runners
  disabled in run 17), and the server reports healthy on the eager fallback. The new
  unit test `test_fused_snake_beta_survives_a_fullgraph_compile` fails on main with that
  error and passes on the branch.
- `_MAX_T = 65536` has no kernel reason; the 96-channel stage passes it beyond 34 frames.

## 2. The rewrite (flat, then tiled launch, precomputed constants): not shipped

Per activation over the 220 real shapes (run 15): flat launch geometric mean 0.89 of
main's kernel but up to 1.55x slower at 0.3M to 9M elements (a per-element divide and
modulo for the channel); tiled launch (1,024 elements, 4 warps) geometric mean 0.83, worst
1.03, 205 of 220 at or below main's kernel. Both bit-identical. At the decode level (run
16, 44 shapes) the tiled kernel is 0.993 to 1.003 of main's kernel on widths 1 to 32 and
0.82 to 0.95 on the 64-frame windows; main's kernel with only `_MAX_T` lifted equals it
there (0.999 to 1.001). Nothing measurable is left for the new layout, which has not run
on another GPU, and the precomputed constants can go stale.

Findings kept from that work: SGLang's `round_bf16_to_fp32` turns the all-ones NaN that
`sin(inf)` returns into -0.0 (232 to 13,926 mismatching elements per channel count on the
all-encodings tensor); the cast-pair rounding needs `enable_fp_fusion=False` (mismatch on
random inputs without it); 64-bit offsets cost about 2x at small shapes. V-e8's absolute
microseconds at mid sizes imply more than the card's bandwidth and are used as ratios
only; the full-decode tables use a timer checked against a device copy (904 GB/s).

Open, bench only: two CUDA graphs over two deep-copied decoders fault on replay in
`vocoder_fused_snake_bench.py` (also eager against main's kernel); single-graph runs are
clean and serving runs many graphs over one decoder.

## 3. The shipped slice, verified

| gate | result |
| --- | --- |
| unit tests on the 4090 | 7 passed: parity at five shapes including [1, 96, 122880], the fullgraph compile bitwise equal to eager |
| full decode, 44 shapes, against main's kernel (run 18) | every waveform bitwise equal to eager; 0.998 to 1.004 on widths 1 to 32 (same binary), 0.82 to 0.95 on 64-frame windows; against eager 0.76 to 0.90 |
| boot at `mem_fraction_static` 0.80, clean cards | 29 modules fused, 4 runners captured, 0 disabled; peak 20,730 MiB (main 20,778); KV 106,464 tokens (main 105,312) |
| startup | 130 s both with a warm compile cache; 175 s on the branch's first boot (one compile-cache miss) |

## 4. Server A/B (run 19: both arms at once, cards 5 and 6, swapped rounds, one pid per card, 0.80)

| round | arm | card | req/s | audio s/s | RTF mean | TTFC mean ms | TTFC p95 ms | ITL mean ms | ITL p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stream c16 | main | 5 | 13.932 | 57.71 | 0.2784 | 144.4 | 195.8 | 123.9 | 223.6 |
| stream c16 | branch | 6 | 14.241 | 59.19 | 0.2716 | 140.1 | 190.4 | 121.0 | 212.5 |
| swapped | main | 6 | 14.216 | | 0.2702 | 137.2 | 181.8 | 120.9 | 211.5 |
| swapped | branch | 5 | 14.419 | | 0.2690 | 136.7 | 180.3 | 119.9 | 209.4 |
| long-form c16 | main | 5 | 2.272 | 79.57 | 0.1831 | | | 107.6 | 140.4 |
| long-form c16 | branch | 6 | 2.276 | 80.58 | 0.1838 | | | 107.7 | 136.3 |
| long-form swapped | main | 6 | 2.323 | 81.60 | 0.1799 | | | 106.1 | 132.2 |
| long-form swapped | branch | 5 | 2.277 | 81.49 | 0.1796 | | | 106.7 | 133.8 |
| seeded c1 | main | 5 | 1.963 | | 0.1243 | 67.9 | 90.0 | 54.9 | |
| seeded c1 | branch | 6 | 1.966 | | 0.1241 | 65.0 | 87.8 | 55.2 | |

- Stream c16: the branch leads on every read in both card assignments; summed over the
  two rounds req/s +1.8 percent, TTFC and ITL about 1.5 to 2 percent better. Card 6 is
  1 to 2 percent faster than card 5.
- Long-form c16: flat (+1.3 and -0.1 percent of audio throughput, 48 requests per run).
  Steady streaming is 8-frame chunks, which main already compiles, with the snake ops
  fused by inductor inside that compile; the kernel pays on the uncompiled shapes (ramp
  chunks, cold chunk, bootstrap windows).
- Quality: c16 WER 1.00 / 1.06 percent, similarity 71.50 / 71.39; seeded c1 WER 1.080 /
  1.072 percent, similarity 71.574 / 71.579; no clip above 50 percent WER in any arm; 0
  failed requests in any boot.
- Identity: 0 of 1,088 seeded c1 WAVs byte identical, 1,088 of 1,088 the same length.
  Arm-to-arm SNR median 39.9 dB, p5 36.2, min 31.1: the 8-frame chunks now round like
  eager instead of like inductor's fusion. That is the spread this bf16 decoder already
  has between its own paths (run 08: minimum SNR against fp32 31.6 to 36.1 dB eager, 34.2
  to 39.6 production, 35.0 to 41.1 compiled).
