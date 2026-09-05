# [Qwen3-TTS] Remove the dead work from the code predictor replay and batch the feedback write

## Summary

Every talker token runs the code predictor chain, 16 one-token forwards replayed as one CUDA graph of 1371 kernels. A kernel census of that replay on H100 (one launch, 0.45 us between kernels, kernels of 1 to 6 us) showed 149 of those kernels do no work that reaches the output: per sub-step, four `index_select` copies of the sampling parameters through an identity index, a clamp of a value the host already staged, three elementwise kernels rebuilding the sampler's seed position, and an argmax plus a `where` that select the sampled token against itself whenever every row of the batch samples, which the checkpoint's default sampling always does. On the host side, the feedback embedding for the next step was written one row per launch inside a Python loop.

This branch removes that work. The replayed kernels that remain are the same kernels in the same order, so the outputs are the same bits: the c1 full corpus is byte identical, 1088 of 1088 WAVs. The replay is 149 kernels shorter, 0.31 ms at 1 row and 0.55 ms at 16 rows, and the full corpus runs 3.4% faster at c1 and 3.8% faster at c16.

## Changes

- `prepare_decode_buffers` stages the subtalker temperature already clamped to the sampler's floor, so the sub-steps read it without a kernel.
- The seed positions of the 15 sub-steps of one decode position are one table computed once per predictor call, `_sub_seed_positions`, and each sub-step passes its row to the sampler. The sampler reads temperatures, top k, top p and seeds as slices of the staged buffers, which are views, in place of the four `index_select` copies through the identity index. `_sub_identity_row_indices_tensor` and `_select_semantic_positions` go away with the copies.
- A fifth graph signature term, whether the batch has argmax rows. The graph of a batch where every row samples returns the sampled tokens and runs no argmax and no `where`. A mixed batch keeps today's path under its own key. `prepare_decode_buffers` sets the term, the capture state saves and restores it in signature order, and the startup capture builds both variants of the default signature, so a mixed batch never captures inside a serving step. The startup log reads `Captured 12 Qwen3-TTS predictor CUDA graphs for signatures=[...] in 3.3 s`, against 6 in 2.8 s before, with the same key budget.
- `_write_feedback_buffers` stacks the staged feedback rows and the next text rows of the whole batch and adds them in one call, four launches per step in place of one per row plus the stack. Rows without a staged feedback row, the first decode after a prefill or a retract re-prefill, keep the per row embedding of their token id. The rule that decides which rows have a staged input lives in `QwenTalkerModelRunner._peek_next_decode_inputs` and `_pop_next_decode_inputs`, and the Qwen3-Omni per row helper is written on the same two functions. The history rows a retracted request replays are views of one clone per step.

No new kernel, no new configuration, no chosen constant. The temperature floor of 1e-5 moved from a kernel on the device to the host staging with the same value.

## Test results

H100 80GB HBM3, driver 580.126.20, CUDA 13.0, SGLang 0.5.18, torch 2.13.0, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, default engine config. `pytest tests/unit_test/qwen3_tts -q`: 394 passed.

A is upstream main `91e9c3095`, B is this branch with that main merged, `2c00eb688`.

Kernel census of the predictor replay, torch profiler window of 12 requests at c1 and 192 at c16, one fresh server per arm, one unprofiled warmup request:

| Per replay | A, 1 row | B, 1 row | A, 16 rows | B, 16 rows |
| --- | ---: | ---: | ---: | ---: |
| kernels | 1371 | 1222 | 1371 | 1222 |
| replay busy | 3.892 ms | 3.647 ms | 4.428 ms | 3.943 ms |
| replay wall | 4.714 ms | 4.405 ms | 5.258 ms | 4.706 ms |
| step wall p50 | 8.052 ms | 7.767 ms | 9.151 ms | 8.489 ms |

The 149 removed kernels are 60 `indexSelectSmallIndex`, 74 elementwise and 15 `reduce_kernel` argmax, the count the change derives. No family grew, and the GEMM, norm, attention, rope and activation kernels are unchanged in count and time.

Full corpus A/B on the seed-tts-eval English split, 1088 samples, voice clone with references, `--seed 1234`, no warmup request, one fresh server per point, profiling off, order A c1, B c1, B c16, A c16. WER by Qwen3-ASR-1.7B, speaker similarity by the fine tuned WavLM head.

c1:

| Metric | A | B | Delta |
| --- | ---: | ---: | ---: |
| Mean latency | 0.445 s | 0.430 s | -0.015 s (-3.4%) |
| Median latency | 0.436 s | 0.422 s | -0.014 s (-3.2%) |
| p95 latency | 0.643 s | 0.622 s | -0.021 s (-3.3%) |
| p99 latency | 0.744 s | 0.721 s | -0.023 s (-3.1%) |
| QPS | 2.246 | 2.323 | +0.077 (+3.4%) |
| WER | 1.00477% | 1.00477% | 0 |
| Speaker similarity | 71.30515 | 71.30515 | 0 |
| WAVs byte identical to A | reference | 1088 of 1088 |

c16:

| Metric | A | B | Delta |
| --- | ---: | ---: | ---: |
| Mean latency | 1.058 s | 1.020 s | -0.038 s (-3.6%) |
| Median latency | 1.031 s | 0.990 s | -0.041 s (-4.0%) |
| p95 latency | 1.542 s | 1.445 s | -0.097 s (-6.3%) |
| p99 latency | 1.866 s | 1.825 s | -0.041 s (-2.2%) |
| QPS | 15.038 | 15.611 | +0.573 (+3.8%) |
| WER | 1.07176% (128 errors) | 0.99640% (119 errors) | -0.075 pp |
| Speaker similarity | 71.32257 | 71.20087 | -0.122 |

At c16 the batch composition differs between arms, so every sample's audio differs by the rounding of a different GEMM shape, in both directions: 535 samples score higher in B and 553 in A, 25 transcripts improve and 17 worsen. Both arms sit inside the spread of c16 boots of identical kernels measured on this model, 116 to 128 errors over 11943 words and similarity 71.18 to 71.32 (three earlier boots plus this pair).

Memory, GPU total sampled once a second including startup: c1 76863 MiB (A) and 76887 MiB (B), c16 81055 MiB (A) and 80735 MiB (B). A process wide allocator snapshot of the c16 window on both arms showed equal allocated memory at start and end and no out of memory event on either arm.

Serving logs: no lazy capture, no fallback to eager, no retract, no CUDA error on either arm at either concurrency.
