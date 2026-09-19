# Runs 11 to 14 readout: vocoder compile (V2) and the fused activation (V-e7)

moss RTX 4090 D, cards 1 and 7 (the other six held by another tenant of the same
container), Qwen3-TTS-12Hz-1.7B-Base, main `ebd577ea0`. Scripts: `run_boot_probe.sh`,
`vocoder_runner_memory.py`, `run_ab_pairs.sh`, `vocoder_fused_snake_bench.py`.

## 1. External factors found

- Another tenant's short jobs (about 2 GB each, 16 to 18 GB per card in total) land on
  any card that reads 1 MiB. Both first V2 boot probes on card 7 were void for this reason
  (memory rose to 20 GB before or after our vocoder capture, with the GPU startup lock held
  by their server). Every boot now logs the pids on its card; more than one voids it.
- A/B arms must bench at the same time: V2's startup compile loads the shared CPUs for
  minutes. `run_ab_pairs.sh` holds both arms at a barrier before the bench and before
  scoring.

## 2. V2: one dynamic torch.compile for every captured shape (branch
`perf/qwen3-tts-vocoder-compile-all-widths`, all widths `90ea2985a`, chunk shapes only
`d3b6f66f0`)

| read (clean boots) | main | V2 all widths | V2 chunk shapes |
| --- | ---: | ---: | ---: |
| startup s | 120 | 346, 351, 336 | 346 |
| free before talker weights GB | 19.38 | 18.40 | 19.63 |
| KV pool tokens | 114,384 | 106,613 | 116,387 |
| one warm runner's graph memory MiB (32 keys, shared pool) | 206 | 410 | |

Dynamo traces 7 graphs (size-1 and duck-size specializations of batch and width); the
inductor disk cache did not shorten later boots.

Stream c16, full corpus, both arms at once, clean cards (run 13):

| round | arm | card | req/s | RTF mean | TTFC mean ms | TTFC p95 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| c16 | main | 1 | 14.843 | 0.2608 | 126.0 | 174.8 |
| c16 | V2 | 7 | 14.829 | 0.2608 | 121.2 | 165.7 |
| swapped | main | 7 | 14.812 | 0.2605 | 127.7 | 166.4 |
| swapped | V2 | 1 | 15.100 | 0.2562 | 117.2 | 154.4 |

Mean of the two rounds: req/s +0.9 percent, TTFC mean -6 percent. WER 1.10 / 1.00
percent, similarity 71.50 / 71.40 (round 1). Verdict: not shipped; 3.6 minutes of extra
startup for about 1 percent of throughput. Main already compiles width 8, the steady
chunk, so V2 only added the ramp widths, cold shapes and windows.

Consequence for the run 10 budget: vocoder device time (46 percent on seed-tts c16) is
mostly not on the throughput-critical path at c16; cutting the non-steady vocoder shapes
by 20 to 44 percent moved throughput by about 1 percent. It is on the first-audio path.

## 3. SGLang 0.5.19 mechanics (research report
`artifacts/qwen3_tts_step_profiling_20260918/research_sglang_compile_graph_mechanics.md`;
the lines below were re-read in the pinned checkout)

- The default decode path captures eager kernels and hand-fused custom ops into full CUDA
  graphs; torch.compile is opt-in, `dynamic=False` per batch size, only for bs up to
  `torch_compile_max_bs` 32, with Dynamo cache limits raised to 1024
  (`compilation/torch_compile_decoration.py:56-83`,
  `model_executor/runner/base_cuda_graph_runner.py:96`).
- Fused ops and compile exclude each other: `_to_torch` swaps `BaseFusedOp` modules to
  compile-safe forwards while compile is active (`torch_compile_decoration.py:31`).
- The piecewise backend compiles no per-size graphs (`compile_sizes` empty,
  `compilation/cuda_piecewise_backend.py:84`); per size it only captures, one warmup.

## 4. V-e7: the in-tree fused SnakeBeta (#1794, off by default) in plain CUDA graphs

All 44 captured keys, eager against fused, same codes and zero state (run 14):

- Waveform bitwise identical at 44 of 44 keys (max abs 0).
- Replay time -10 to -23.5 percent at 40 keys; w1 b1 +3 percent (2.61 to 2.69 ms);
  the 64-frame windows -5 to -12 percent: the kernel's `_MAX_T = 65536` sends the last
  stages (64 x 1920 = 122,880 samples) to the eager chain (453 elementwise kernels left
  against 383 elsewhere).
- 261 kernels fewer per decode (about 1,050 to 790). Fusing costs 0.01 s at startup.
- Left after fusing: 383 elementwise kernels (0.5 to 2.5 ms per decode) and 156 to 190
  copy and concatenation kernels (0.2 to 1.3 ms); convs, cuDNN layout transposes and the
  transformer GEMMs are untouched by either fusion or compile.
- For the steady 8-frame chunk main's static compile is still faster than fused eager
  (w8 b8 about 7.5 against 9.1 ms), so compile can only go once the remaining chains are
  fused by hand too.
