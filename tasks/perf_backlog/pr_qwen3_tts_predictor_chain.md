# [Qwen3-TTS] Remove the dead work from the code predictor replay and batch the feedback write

## Summary

Every talker token runs the code predictor chain, 16 one-token forwards replayed as one CUDA graph of 1371 kernels. A kernel census of that replay on H100 (one launch, 0.45 us between kernels, kernels of 1 to 6 us) showed 149 of those kernels do no work that reaches the output: per sub-step, four `index_select` copies of the sampling parameters through an identity index, a clamp of a value the host already staged, three elementwise kernels rebuilding the sampler's seed position, and an argmax plus a `where` that select the sampled token against itself whenever every row of the batch samples, which the checkpoint's default sampling always does. On the host side, the feedback embedding for the next step was written one row per launch inside a Python loop.

This branch removes that work. The replayed kernels that remain are the same kernels in the same order, so the outputs are the same bits: the c1 full corpus is byte identical, 1088 of 1088 WAVs. The replay is 149 kernels shorter, 0.31 ms at 1 row and 0.55 ms at 16 rows, and the full corpus runs 3.4% faster at c1 and 3.8% faster at c16.

## Changes

- `prepare_decode_buffers` stages the subtalker temperature already clamped to the sampler's floor, so the sub-steps read it without a kernel.
- The seed positions of the 15 sub-steps of one decode position are one table computed once per predictor call, `_sub_seed_positions`, and each sub-step passes its row to the sampler. The sampler reads temperatures, top k, top p and seeds as slices of the staged buffers, which are views, in place of the four `index_select` copies through the identity index. `_sub_identity_row_indices_tensor` and `_select_semantic_positions` go away with the copies. A frame whose rows all take the argmax builds no table: the table only feeds the sampler, and the flag that decides it is already a term of the graph signature.
- A fifth graph signature term, whether the batch has argmax rows. The graph of a batch where every row samples returns the sampled tokens and runs no argmax and no `where`. A mixed batch keeps today's path under its own key. `prepare_decode_buffers` sets the term, the capture state saves and restores it in signature order, and the startup capture builds both variants of the default signature, so a mixed batch never captures inside a serving step. The mixed variant skips bucket 1, which no batch can reach, so the default ladder of six buckets captures 11 graphs against 6 before.
- The startup set is captured whole. The cap of 32 keys that predates this branch now bounds the captures beyond the startup set, the ones a client's own sampling values trigger, in place of the whole cache. Before, a ladder of more than 16 buckets, which a running cap above 96 or a dense explicit bucket list produces, filled the cap during startup and left the smaller mixed buckets and every later signature eager for the life of the server. The constant is `_PREDICTOR_GRAPH_MAX_LAZY_KEYS` and the warning names the keys beyond the startup set.
- `_write_feedback_buffers` stacks the staged feedback rows and the next text rows of the whole batch and adds them in one call, four launches per step in place of one per row plus the stack. Rows without a staged feedback row, the first decode after a prefill or a retract re-prefill, keep the per row embedding of their token id. The rule that decides which rows have a staged input lives in `QwenTalkerModelRunner._peek_next_decode_inputs` and `_pop_next_decode_inputs`, and the Qwen3-Omni per row helper is written on the same two functions.
- The history rows a retracted request replays are views of one clone per step, so a request holds the clone of every step it ran in. While a request advances with the batch, the clones alive are those of the last max_new_tokens steps, at most 16 rows of 4 KiB each at the defaults, 128 MiB for 2048 tokens, the same ceiling the per row allocation had. That figure covers the batch snapshots alone: a compacted history and the copy that makes it are the retracted request's own allocations on top of it. A request that leaves the running batch, by KV retraction or by the retract pause, would keep every clone of its run alive while it waits, so the scheduler copies its rows into storage of their own before it requeues them, one stack per retraction, on a path that already pays a full re-prefill. `OmniScheduler._add_request_to_queue` wraps the upstream method for that.
- The two history fields, `prefill_input_embeds` and `decode_input_embeds`, are declared once on `ARRequestData`. The scheduler clears both on every finished request and now reads the history of every retracted request, whatever the model, and MOSS-TTS Local's request data had neither, so a retraction on that stage would have raised. Every class the scheduler attaches derives from `ARRequestData`, and the duplicate declarations on the SGLang and MOSS-TTS classes go away. A test requeues a retracted request with the real MOSS-TTS and MOSS-TTS Local data classes.

No new kernel, no new configuration, no chosen constant. The temperature floor of 1e-5 moved from a kernel on the device to the host staging with the same value, and a test stages temperatures at zero, below, at and above the floor against the former device order of conversion then clamp.

## Test results

H100 80GB HBM3, driver 580.126.20, CUDA 13.0, SGLang 0.5.18, torch 2.13.0, `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, default engine config. `pytest tests/unit_test/qwen3_tts -q`: 397 passed on the final head.

A is upstream main `91e9c3095`, B is this branch with that main merged, `2c00eb688`. The commits after `2c00eb688` are host Python and tests: the direct field reads in the talker runner, the graph cache accounting, the greedy table, the history compaction, the base class fields, and their tests. The device path of the default sampled replay is unchanged by them: a census of the final head counts the same 1222 kernels per replay at 1 and 16 rows with the same kernel names, and its c1 full corpus output is byte identical to B's below, 1088 of 1088. The greedy replay changes by design, the seed table gate removes its one table kernel, and the final head's argmax replay counts 1221 kernels with the same names as main's.

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

The history compaction and the base class fields run on every model whose engine stage uses `OmniScheduler`, and the decode input helpers run on the Qwen3-Omni talker, so the final head was also checked there: the full unit suites pass (6005 CPU, 292 accelerator), the Qwen3-Omni talker under retraction forced by `SGLANG_TEST_RETRACT` on its stage completes 50 of 50 with every re-prefill at N plus 1 tokens, Qwen3-TTS under the same switch at c16 completes 192 of 192 over 13 retractions, and MOSS-TTS Local, the model whose request data lacked the fields, completes 16 of 16 over 3 retractions with no exception.

## Reproduction

One fresh server per point, `CUDA_VISIBLE_DEVICES` set to one GPU, the arm selected by `PYTHONPATH`, GPU memory sampled once a second from before the boot. A is `91e9c3095`, B is `2c00eb688`, the final head is `d26ac7a1e`.

```bash
nvidia-smi -i $GPU --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > mem.csv &
python -m sglang_omni.cli serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml --host 127.0.0.1 --port 31001
python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta zhaochenyang20/seed-tts-eval-arrow \
  --ref-format references --lang en --seed 1234 --warmup 0 --port 31001 \
  --max-concurrency $C --output-dir $DIR
python -m benchmarks.eval.benchmark_tts_seedtts --transcribe-only \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta zhaochenyang20/seed-tts-eval-arrow --lang en \
  --asr-model-path Qwen/Qwen3-ASR-1.7B --asr-concurrency 1 --port 31010 --output-dir $DIR
python -m benchmarks.eval.benchmark_tts_seedtts --similarity-only \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta zhaochenyang20/seed-tts-eval-arrow --lang en \
  --device cuda:0 --output-dir $DIR
```

Kernel census: one unprofiled request, then a torch profiler window of 12 requests per concurrency unit opened and closed through `/start_profile` and `/stop_profile` with `enable_torch` set, and the trace reduced to kernels per predictor replay by the census script kept with the result archives.

Forced retraction: Qwen3-TTS with `SGLANG_TEST_RETRACT=1 SGLANG_TEST_RETRACT_INTERVAL=64` in the server's environment at c16 over 192 requests; Qwen3-Omni on the bf16 colocated H100 profile with `stages.talker_ar.env` carrying `SGLANG_TEST_RETRACT: "1"` and `SGLANG_TEST_RETRACT_INTERVAL: "50"`, 50 voice clone requests at c16; MOSS-TTS Local with the same two keys on `stages.tts_engine.env`, 16 requests at c4. The WAV hash manifests, serve logs, traces and memory samples of every run are kept in the result archives named by date, available on request.
