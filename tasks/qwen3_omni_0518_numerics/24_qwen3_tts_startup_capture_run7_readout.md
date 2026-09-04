# 24. Run 7: A/B of the rebuilt startup capture, and the c16 WER question

Source: `artifacts/qwen3tts-startup-capture-validation-b6fdc94d7-slim.tar.gz`,
extracted to the scratchpad as `run7/`. Arm A is upstream main `fb6cd93e8`,
arm B is `b6fdc94d7`, the seven commit rebuild of doc 21 Slice A on
`perf/qwen3-tts-predictor-startup-capture` with the capture mode flag, the
mixed eager sampler and the sm90 pin removed. Physical H100 GPU 2, one
server per point, order A c1, B c1, B c16, A c16, `--warmup 0`,
`--seed 1234` per request, full seed-tts-eval English split, 1088 samples.
Scripts: `scripts/run7_wer_runs.py` and `scripts/run7_c1_pairs.py`, over the
`wer_results.json` and `speed_results.json` of runs 5, 6 and 7.

## 1. Speed

| Point | Latency mean s | Median s | p95 s | p99 s | RTF mean | QPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A c1 | 0.443 | 0.428 | 0.653 | 0.777 | 0.1088 | 2.256 |
| B c1 | 0.444 | 0.430 | 0.657 | 0.771 | 0.1089 | 2.251 |
| A c16 | 1.110 | 1.023 | 1.572 | 4.541 | 0.2743 | 14.335 |
| B c16 | 1.052 | 1.027 | 1.510 | 2.202 | 0.2591 | 15.115 |

B minus A: the first c1 request is 0.895 s faster, the first c16 batch
2.748 s per request faster, c16 p99 51.5% lower, c16 QPS 5.4% higher.
Both B serving logs carry one startup line, six graphs in 3.0 s, and no
lazy capture line. Arm A captured lazily, one key at c1 and six at c16.
This reproduces doc 23 on the rebuilt history.

## 2. Output identity and quality

| Point | Equal WAV SHA-256 | WER | Errors over 11943 words | Similarity |
| --- | ---: | ---: | ---: | ---: |
| A c1 | reference | 1.00477% | 120 | 71.3052 |
| B c1 | 1088 of 1088 | 1.00477% | 120 | 71.3052 |
| A c16 | reference | 0.98803% | 118 | 71.1779 |
| B c16 | 0 of 1088 | 1.07176% | 128 | 71.3138 |

## 3. The c16 WER delta is not a numerical change

Four independent observations.

### 3.1 The served kernels are the same

c1 byte identity over 1088 samples covers the bucket 1 graph end to end.
The other buckets are captured from the same
`_code_predictor_forward_incremental` with the same fused kernels, and
the accelerator tests compare every startup captured bucket against the
eager path bit for bit (`test_graph_bit_identity_sampled` at 1, 2, 4, 8
and 16 rows, `test_startup_capture_builds_the_ladder_for_one_signature`),
all of which passed on the box in this run. Main's own copy of the first
test compares its ATen eager path against its fused graph and passes, so
main's graph, B's graph and B's eager path form one numeric chain.

### 3.2 c16 does not reproduce for any revision

c16 error counts of every arm that replays identical kernels, runs 5, 6
and 7:

| Arm | Errors |
| --- | ---: |
| run 5 A (main) | 120 |
| run 6 A (main) | 118 |
| run 6 B (startup capture, `961e608f5`) | 116 |
| run 7 A (main) | 118 |
| run 7 B (startup capture, `b6fdc94d7`) | 128 |

Pairwise, samples whose error count differs and the net error delta:

| Pair | Samples differing | Net errors |
| --- | ---: | ---: |
| run 5 A vs run 6 A, main against main | 39 | -2 |
| run 6 A vs run 7 A, main against main | 21 | 0 |
| run 6 B vs run 7 B, startup capture against itself | 35 | +12 |
| run 6 A vs run 6 B | 21 | -2 |
| run 7 A vs run 7 B | 42 | +10 |

Two boots of the startup capture code differ by 12 errors, more than A
against B in run 7. About 40 samples flip by one word each, so the net of
40 signs has a standard deviation near 6 and a net of 10 is inside one and
a half of them.

### 3.3 The flipping samples are the same samples

Of the 42 samples that differ between A and B in run 7, 29 also differ
between two boots of main, 28 between the two boots of B, and 2 differ in
neither. 69 of 1088 samples flip in at least one same kernel pair. The
largest run 7 deltas are all repeat flippers: "wood is" against "woods"
(also flipped in run 5 A at c1), "tebow" against "tibo" and "tepo",
"nora" against "norah", "boy" against "boys", "manager is" against
"manager", "soldier" against "soldiers".

### 3.4 The judge adds noise of its own

At c1 the A and B WAVs are byte identical, yet 2 of 1088 transcripts
differ ("roads" against "road", "grant" against "grand"), net 0.

### 3.5 Why c16 varies

Batch composition. cuBLAS and cuDNN select algorithms by shape, so a
request in a batch of 12 rows gets different bf16 roundings than in 16
rows, the seeded sampler then draws a different token wherever two logits
are close, and the trajectory diverges. B removes the 350 ms stalls at
the first step of each bucket, so from the first batch on the two arms
batch different requests together, which is why only 103 of 1088 token
counts match between the c16 arms. The doc 23 diagnostic, 0 of 16 same
arm WAV matches across boots, is the same effect. A per commit runtime
bisection cannot resolve a delta of this size: each c16 point is one draw
from a distribution whose width is the delta itself.

## 4. c1 steady state

Paired per request, first request excluded, all 1087 pairs with equal
token counts:

| Run | B minus A mean | Median | Sd | Quarter means | Slope against tokens |
| --- | ---: | ---: | ---: | --- | ---: |
| run 6 | +0.49 ms | +0.10 ms | 19.0 ms | -3.53, +3.18, +1.27, +1.05 ms | +15.0 us per token |
| run 7 | +1.84 ms | +2.50 ms | 17.2 ms | -1.45, +2.25, +0.84, +5.71 ms | -4.2 us per token |

A per step code difference would show as a per token slope of one sign in
both runs and a delta uniform in time. Instead the delta drifts across the
quarters and the slope changes sign between runs, and B was second at c1
in both runs. The B c1 serving path replays the same bucket 1 graph as A
and differs by a dictionary lookup over six keys instead of one. This
reads as drift over the eight minute run, not as a code path cost, but
the order confound is not removed. The cheap check is c1 only with the
order swapped, B then A.

## 5. Unit tests: 351 passed, 3 failed, all three real

The pushed `b6fdc94d7` was rebuilt without a test run on the box, and the
three failures are what that cost.

- `test_qwen3_tts_engine_accepts_64_batch_policy_and_enables_cuda_graph`:
  the fake wrapper had no `_merge_generate_kwargs`, and the builder hook
  indexed the merged defaults directly. A checkpoint without
  `generation_config.json` would have raised `KeyError` at startup while
  the request path falls back to (True, 0.9, 1.0, 50). Fix: the fallbacks
  now live once in `resolve_subtalker_sampling` in `request_builders`,
  used by the request path and the builder hook, and the fake models the
  real wrapper's merge of checkpoint defaults with kwargs.
- `test_batch_invariant_mode_keeps_the_eager_gemm_on_the_graph_path`: the
  fused addmm gate was a flag resolved by the graph path, so the eager
  path could run before the flag was set. Fix: the flag and the resolver
  are gone, `_predictor_o_proj_add_residual` is static again and reads
  `is_batch_invariant_mode_enabled()` at the call (a module global read),
  and the sm90 pin went with it.
- `test_fused_raw_logit_sampler_captures_without_reference_top_k`: the
  test helper `_production_seeded_tokens` had the fused kernel mocked out,
  which turned the capture test into the fallback path. Fix:
  `_reference_seeded_tokens` (ATen) for expectations and
  `_production_seeded_tokens` (the real path) under capture.

Rebuilt history on `perf/qwen3-tts-predictor-startup-capture`, eight
commits from `fb6cd93e8`:

1. extract the predictor sampling signature rule
2. run the predictor graph kernels on the eager path too
3. gate the fused predictor addmm on batch invariant mode
4. extract the predictor graph failure accounting
5. move the predictor graph capture onto the talker
6. capture predictor graphs on one stream with the collector off
7. extract the subtalker sampling defaults
8. capture the predictor graphs at startup

Every commit compiles and references nothing from a later commit. Head
`d7e34a16c`, force pushed. That head also prunes the branch's tests to
contract and edge case tests: the two kernel count tests (monkeypatched
kernels counted across warmups and capture), the stream identity test and
the collector state test (mechanics recorded inside the forward) and the
builder test with a namespace model are gone. The ladder test now replays
every bucket from the shared pool against the eager path bit for bit, and
the failed capture test asserts the collector is enabled again. The runtime source differs from `b6fdc94d7` in
the resolver removal, the gate of section 6, the layer 0 copy of section
6, the stream restore of section 6 and the subtalker defaults extraction.
None of these touches the replayed kernels, so the c1 output should stay
byte identical to A.

## 6. Review of the rebuild

A second whole file review of the rebuild against torch 2.13 and sglang
0.5.18 sources. Applied:

- Gate of the fused addmm. sglang's `UnquantizedLinearMethod.apply`
  routes bf16 GEMMs through the cutedsl backend when the backend resolves
  to it, which `initialize_bf16_gemm_config` does for `auto` on sm100
  outside deterministic mode (v0.5.18 `unquant.py:89-119` and
  `:238-265`), so a fused `torch.addmm` there would bypass sglang's GEMM
  choice. The sm90 pin was protecting that assumption by architecture. The
  gate is now the assumption itself: `not is_batch_invariant_mode_enabled()`
  and `not get_bf16_gemm_backend().is_cutedsl()`, the same test sglang's
  `apply` makes. `get_bf16_gemm_backend()` defaults to `AUTO` before
  initialisation (`unquant.py:141-145`), and the only initialisation call
  in v0.5.18 is in sglang's own `Scheduler.__init__`
  (`managers/scheduler.py:900`), which omni does not construct, so in an
  omni process the backend stays `AUTO` and sglang's linear runs
  `F.linear` on every platform. The gate then only bites if omni ever
  initialises the backend.
- Layer 0 input ownership on the eager path. `project_input` returns its
  argument when the talker and predictor hidden sizes match, which is the
  case for `Qwen3-TTS-12Hz-0.6B-Base` (1024 and 1024, from its
  `config.json`). With the eager path now running the fused o_proj
  epilogue, layer 0 would have written into the caller's talker hidden
  state. The slice is cloned when the projection hands it back unchanged,
  and an accelerator test pins the caller's tensor.
- Current stream after a failed capture. torch 2.13's `graph.__exit__`
  calls `capture_end` before it leaves its stream context
  (`torch/cuda/graphs.py:471-484`), so a capture invalidated by an unsafe
  call would have left the capture stream current for the life of the
  process. The graph block now sits inside the talker's own stream
  context, and an accelerator test synchronises inside the capture and
  asserts the default stream is current afterwards.
- The stale comment on `_predictor_graph_enabled`, the "at startup" phrase
  in the log line (now `Captured 6 Qwen3-TTS predictor CUDA graphs for
  signature=... in 3.0 s`), the warmup multiplier in two tests, one test
  name and one fake wrapper.

Not applied, with the reason:

- Containing a startup capture failure and serving eagerly. A server that
  boots with the predictor four times slower is the flaky outcome, and
  sglang's own capture fails the boot. Kept as decided on the first review.
- `cleanup_build_failure` for the preprocessing context. The stage process
  logs the failure, destroys the process group, reclaims the GPU and
  exits (`stage_workers.py:428-440`), so there is nothing left to clean.
- A named tuple for the graph signature. Right, but it reshapes code that
  predates this change. Follow up with the file split (doc 25).
- ROCm with aiter is not excluded by the new gate, as the sm90 pin used to
  exclude it. Not a target of this work.

### 6.1 Correction after the first push of the rebuild

The rebuild first went out as `08cd988b1` with
`get_bf16_gemm_backend().is_optimized()` in the gate. That method exists
on sglang upstream main, which the local `/Users/ratish/sglang` checkout
had been switched to by mistake when the review read it (its reflog shows
the moves between `main` and `v0.5.18`). The v0.5.18 enum has `is_auto`
and `is_cutedsl` only, and the box returned 303 passed and 53 failed, all
`AttributeError: 'Bf16GemmBackend' object has no attribute 'is_optimized'`.
Every sglang fact this branch relies on was then re-read at the `v0.5.18`
tag: `is_batch_invariant_mode_enabled` and the `aten::addmm` override
without an out variant (`batch_invariant_ops.py:976-1000`),
`maybe_enable_batch_invariant_mode` after `load_model` in
`ModelRunner.initialize` (`model_runner.py:620-661`), `get_bf16_gemm_backend`
and `apply` as cited above, `multinomial_with_seed`,
`get_global_server_args` and the murmur hash kernel. Only the enum method
was wrong. `4f43776fe` differs from `08cd988b1` in that one token.

## 7. What to run next

1. `pytest tests/unit_test/qwen3_tts -q` on the rebuilt branch. Every test
   is expected to pass.
2. One deterministic c1 request, as in run 7.
3. c1 only, fresh servers, order B then A, to settle section 4. No c16
   rerun: the c16 speed result is reproduced twice (doc 23 and section 1)
   and a c16 WER point cannot answer a numerical question by construction.

## 8. Run 8: c1 with the order swapped, head `d7e34a16c`

Same box, same corpus, same seed, fresh server per arm, B first then A.
The unit tests of the pruned head are not part of this report.

| Point | Latency mean s | Median s | p95 s | p99 s | QPS | WER | Similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B c1 | 0.440 | 0.427 | 0.641 | 0.763 | 2.270 | 1.01315% | 71.30515 |
| A c1 | 0.443 | 0.430 | 0.644 | 0.766 | 2.254 | 1.01315% | 71.30515 |

B minus A past the first request: mean -2.5 ms, median -1.9 ms, B faster
in 786 of 1088 requests, first request -0.672 s. 1088 of 1088 WAVs byte
identical, durations and token counts equal. Startup line: six graphs in
2.8 s, no lazy capture line. Arm A: one lazy capture on the first request.

Read together with section 4:

| Order | B minus A past the first request, mean | Median |
| --- | ---: | ---: |
| A first, B second (run 7) | +1.84 ms | +2.50 ms |
| B first, A second (run 8) | -2.50 ms | -1.90 ms |

Whichever arm runs second is about 2 ms slower. The delta follows the
position, not the code, which is the drift section 4 described. The c1
steady state is flat.

WER is 1.01315% on both arms here against 1.00477% for the same c1 WAVs
in run 7, one more judge error on identical audio, the noise of section
3.4.
