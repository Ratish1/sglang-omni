# 20. Qwen3-TTS predictor attention: implementation plan

Plan status: Conditional. Every architectural decision is resolved from
repository and pinned source evidence. Two facts are verified only at
implementation and measurement time and neither changes the design or
the owner: whether flash attention admits the predictor's shapes on the
CI H100 (the fallback is the math backend, still free of per shape
state), and whether the backend change moves the tokenizer's and the
vocoder's numerics past the CI thresholds (the stage set A/B is the
verdict). Section 8 lists them.

Task class: cross boundary. The change is a process level policy that
reaches three stage factories and three attention callers in one
process, proven at the predictor's capture boundary.

## 1. Scope

Revisions. The fix worktree `/Users/ratish/sglang-omni/.worktrees/qwen3-tts-predictor-warmup`
is at upstream main 15c4568bb with no local changes. The root checkout
is at fda7ce40d, an ancestor, and no file under `sglang_omni/models/qwen3_tts`,
`sglang_omni/config`, `sglang_omni/pipeline`, `sglang_omni/scheduling/engine_factory.py`
or `tests/unit_test/qwen3_tts` differs between the two. Pinned
dependencies read at their tags: sglang v0.5.18 at `/Users/ratish/sglang`,
torch v2.13.0 at `/Users/ratish/pytorch`, qwen-tts 0.1.1 (the wheel,
unpacked in the scratchpad), transformers 5.12.1 (`pyproject.toml:27`).
The analysis documents are on `analysis/qwen3-omni-0518-numerics`, the
capture timing on `perf/step-ledger` at d6425827b.

Target behavior. A new batch bucket of the Qwen3-TTS code predictor no
longer costs fifteen cuDNN attention plan builds on its first eager
pass inside a serving step (doc 19 section 9). The user visible outcome
is the removal of the 430 to 640 ms stall that every rung crossing pays
today, on the c16 window five times.

Requirements, from the user and the standing rules: never patch sglang
or sgl_kernel, fix the mechanism rather than the schedule of the cost,
keep the predictor's graph against eager bit identity, validate runtime
changes by one A/B over the model's CI stage set, and question every
surrounding assumption of the branch.

Success criteria, measured on the H100 box with the capture timing:

- the first warmup pass of every capture holds no plan build gap, so
  its wall is within about two times the second pass (today twelve to
  seventeen times)
- decode `host_ms` and `forward_ms` p50 at rows 1 and 16 are unchanged
  within run to run noise, so the replay pays nothing for the change
- `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py` passes on
  the box, all 55
- the full seed-tts-eval corpus at c1 and c16 on one server per arm,
  the fix commit against 15c4568bb, shows no quality regression in WER
  and speaker similarity and no speed regression, same box, fresh
  server per point
- preprocessing p50 and the vocoder's first request cost are not worse

Non goals, each owned elsewhere: the warmup count, the shared capture
stream, the reference cycle and the collector guard (doc 19 A2),
startup capture of the ladder (A3), a startup warmup request (A4), the
talker and MOSS attentions (A5, doc 17 T26), the chain's op count and
the explicit or triton attention (T22).

## 2. Evidence ledger

Repository facts, all at 15c4568bb unless named.

- The predictor attention is `_predictor_cached_self_attention`
  (`qwen3_tts/sglang_model.py:1758-1818`). It writes k and v into
  `_predictor_k_cache[layer_idx, :batch_size]` at `cache_len` and
  calls `scaled_dot_product_attention` on the slice `[:cache_len + 1]`,
  with `enable_gqa=True` when `num_heads // num_kv_heads` is above one
  (`:1798-1815`). The cache is `[layers, max_batch, kv_heads, predictor_len, head_dim]`
  with `predictor_len = num_code_groups + 1` (`:454`, `:470-478`).
- `_predictor_forward_one_token` (`:1667-1700`) runs every predictor
  layer for one token at `cache_len`, reading positions from
  `_predictor_position_rows[cache_len, :batch_size]` (`:1676`).
- `_code_predictor_forward_incremental` (`:1358-1466`) runs, per token,
  one forward at `cache_len` 0, one at 1, then one per remaining group,
  so key lengths 1 to `num_code_groups`, and samples one code per group.
- The chain executes in four contexts, all through that one function:
  the eager fallback in `code_predictor_forward` (`:1156`, the call at `:1173`), the two
  warmup passes and the capture pass of `_PredictorDecodeGraph._capture`
  (`:137`, `:154`, and the capture body), and the replay of that graph.
  Prefill and decode both reach it through `_collect_codes`
  (`qwen3_tts/model_runner.py:210-230`, called from `post_prefill` and
  `post_decode` at `:72-89`).
- The predictor graph is enabled only for `tp_size == 1`
  (`_resolve_predictor_graph_enabled`, `:1275-1283`), so under tensor
  parallel the chain runs eagerly at every batch size.
- The attention object is `Qwen3OmniMoeThinkerTextAttention`
  (`qwen3_omni/components/thinker_model.py:142-250`), with
  `num_heads`, `num_kv_heads`, `head_dim`, `q_size`, `kv_size` and
  `scaling = head_dim ** -0.5` (`:175-189`). The 1.7B checkpoint gives
  the predictor 16 heads, 8 kv heads and head dim 128 (doc 19 section 1).
- The pipeline config places all three stages in one process named
  `pipeline` (`qwen3_tts/config.py:55-78`, `process="pipeline"` on
  preprocessing, tts_engine and vocoder). Non TP stages must declare
  a process (`config/schema.py:28-38`) and stages with the same name
  form one process group (`config/topology.py:_build_process_groups`).
  The runbook's launch moves the vocoder out with
  `--vocoder.process vocoder`, the CI launch does not.
- The three stage factories are `create_preprocessing_executor`,
  `create_sglang_tts_engine_executor` and `create_vocoder_executor`
  (`qwen3_tts/stages.py:105-203`). Preprocessing is a
  `ThreadedSimpleScheduler` with `max_concurrency` 8, so up to eight
  threads encode reference audio concurrently in that process. The
  vocoder loads the speech tokenizer and warms its CUDA graphs before
  readiness (`:158-203`, `warmup_now` at `:202`). The engine factory
  builds `Qwen3TtsEngineBuilder`, whose `pre_infra_setup` runs before
  sglang infrastructure (`scheduling/engine_factory.py:92`) and whose
  `setup_model` loads the speech tokenizer into the talker and installs
  the preprocessing context (`qwen3_tts/engine_builder.py:70-105`).
- The speech tokenizer is `Qwen3TTSTokenizer.from_pretrained` with
  `attn_implementation` forwarded only when configured
  (`qwen3_tts/stages.py:43-67`, builder default None).
- Two models in the tree already opt out of cuDNN attention:
  MiniMax at its stage factory and acoustic scheduler
  (`minimax_music3/stages.py:54`, `acoustic.py:140`, which also disable
  cuDNN entirely), and dots_tts with a backend list around its tail
  (`dots_tts/tail.py:27`, `:654`).
- No Qwen3-TTS code sets any `torch.backends` flag today (grep).
  `enable_torch_compile` is refused for this engine
  (`engine_builder.py:107-109`) and the vocoder has no torch.compile at
  this revision.
- Tests. `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`
  builds a fake talker with 2 heads, 1 kv head, head dim 4, 4 code
  groups, bf16 (`:40-48`, `:82-160`) and pins graph against eager bit
  identity through the real `_predictor_cached_self_attention`.
  `tests/unit_test/qwen3_tts/test_pipeline.py` constructs the
  preprocessing and vocoder factories with stubs (`:890`, `:1484`,
  `:3070`). The accelerator suite runs on an H100 container
  (`.github/workflows/test.yaml:56-84`). The Qwen3-TTS stage set is
  `tests/test_model/test_tts_ci.py` with `TTS_CI_MODEL=qwen3-tts`
  (`tests/test_model/tts_ci_config.py:181`, `:228`), stages
  `tts-stage-1-nonstream` and `tts-stage-2-stream`
  (`.github/scripts/run_all_wer_ci_aligned.sh:108-114`), and
  `tests/test_model/test_qwen3_tts_batch_invariance.py` (benchmark
  marker) checks batch invariance of the outputs.

External facts.

- torch 2.13 orders SDPA backends cudnn, flash, efficient, math when
  `check_prefer_cudnn_attention` holds, which it does on sm90 with cuDNN
  above 9.15 unless `TORCH_CUDNN_SDPA_DEPRIORITIZED` is set
  (`aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:80-98`, `:110-118`).
- cuDNN attention caches one execution graph per `MHAParams`, which
  include batch and key length (`aten/src/ATen/native/cudnn/MHA.cpp:198-223`),
  and builds on a miss (`:1384-1420`). The cache is a process wide
  unordered map with no eviction in that file.
- `torch.backends.cuda.enable_cudnn_sdp(False)` sets the global
  context flag `enabled_cudnnSDP` (`torch/backends/cuda/__init__.py:660-666`,
  `aten/src/ATen/Context.h:490`), read at every dispatch by
  `check_runtime_disabled_cudnn` (`sdp_utils.cpp:826-834`). The flag
  is a plain member of the global context, not thread local, and exists
  on CPU builds (verified on the 2.9.1 laptop build).
- `torch.nn.attention.sdpa_kernel` sets the same global flags for the
  duration of a context (`torch/nn/attention/__init__.py:92-106`).
- Flash attention admits dense inputs with grouped query heads, head
  dim up to 256, bf16, no mask and last dim stride one
  (`sdp_utils.cpp`, `use_flash_attention` constraints at `:940-970`,
  `backend_supports_grouped_query_attention = true` at `:956`). The
  efficient backend does not admit grouped query heads
  (`check_batch_size_and_num_heads_dense<false>` in its list). Math
  admits everything.
- The speech tokenizer encoder is a transformers `MimiModel`
  (`qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:899`),
  and the tokenizer decoder's attention goes through
  `ALL_ATTENTION_FUNCTIONS[config._attn_implementation]` (`:282-338`),
  the transformers interface whose `sdpa` entry calls
  `scaled_dot_product_attention` (`transformers/integrations/sdpa_attention.py`).
  The speaker encoder is ECAPA-TDNN, convolutional
  (`qwen_tts/core/models/modeling_qwen3_tts.py:311`).

Inferences.

- In the default layout the `pipeline` process holds three SDPA
  callers on three thread sets: the Mimi encoder on up to eight
  preprocessing threads, the predictor on the scheduler thread, and
  the tokenizer decoder on the vocoder scheduler. Each meets cuDNN's
  per shape plan cache: the predictor per bucket and key length
  (measured), the encoder per reference audio length and the decoder
  per chunk shape (not measured, same mechanism).
- A backend flag toggled around the predictor chain from the scheduler
  thread is observed by the preprocessing threads mid call, because the
  flag is process global. That makes the tokenizer's backend depend on
  timing.
- The predictor's shapes pass flash's constraints, so with cuDNN off
  the dispatcher selects flash on the H100. If a constraint not read
  here rejects them, the dispatcher falls through efficient (rejects
  grouped heads) to math, which is correct and has no plan cache.

Assumptions, each with its check.

- Flash and cuDNN attention produce outputs within bf16 rounding of
  each other for these shapes, so sampled codes change rarely and the
  CI thresholds hold. Checked by the stage set A/B and the batch
  invariance test.
- The Mimi encoder and the tokenizer decoder are not slower on flash
  than on cuDNN at their sequence lengths. Checked by preprocessing p50
  and the vocoder's per request timings in the request view.

Decisions are in section 5. Open questions are in section 8.

## 3. Current mechanics

### 3.1 The chain and its four executions

```
serving step on the scheduler thread of the pipeline process
  model_runner.py:210 _collect_codes  (post_prefill and post_decode)
    :228 model.code_predictor_forward(layer0_codes, hidden, semantic_positions)
      sglang_model.py:1156 code_predictor_forward
        :1285 _predictor_forward_graphed
          key hit  -> _PredictorDecodeGraph.replay   [graph, no host attention]
          key miss -> _PredictorDecodeGraph._capture
                        :137 warmup pass 1  -> chain, eager   <- 15 cuDNN plan builds
                        :154 warmup pass 2  -> chain, eager      per new bucket
                        capture pass        -> chain, recorded
          disabled -> :1173 chain, eager                      <- builds per batch size

chain = _code_predictor_forward_incremental :1358
  per token: forward at cache_len 0, 1, then one per remaining group
    _predictor_forward_one_token :1667  per layer:
      _predictor_cached_self_attention :1758
        qkv_proj, qk norm, rope, write k v at cache_len
        scaled_dot_product_attention(q, cache[: cache_len + 1], enable_gqa)
          torch 2.13 dispatch: cudnn first on sm90
            MHA.cpp:1384 key (b, h, s_q, s_kv, d, strides ...) miss -> build_graph
```

The plan cache is keyed by batch and key length, so one token's chain
at a new batch size misses at every key length 2 to `num_code_groups`
on the first layer that reaches it, fifteen misses on the 1.7B
checkpoint, each 23 to 37 ms on the H100 (doc 19 section 9). The
second pass and every later step hit.

### 3.2 Process topology, default layout

| process | stage | threads | attention caller | shape family |
|---|---|---|---|---|
| pipeline | preprocessing | ThreadedSimpleScheduler, up to 8 | Mimi encoder, transformers sdpa | one reference clip per call, length varies per clip |
| pipeline | tts_engine | OmniScheduler thread | predictor, `sglang_model.py:1802,1809` | batch 1 to 64, key length 1 to 16, 16 by 8 heads, dim 128 |
| pipeline | vocoder | vocoder scheduler | tokenizer decoder, transformers sdpa | chunk of frames, batched, some shapes captured in graphs at warmup |

With `--vocoder.process vocoder` the third row moves to its own
process. The backbone's attention runs through sglang's attention
backend, not SDPA, and is unaffected by any SDPA flag.

### 3.3 Cost today

Per new bucket, once per process: about 430 to 640 ms of host on the
scheduler thread with every request in the batch waiting. Per token in
steady state: 80 attention kernel launches inside the replayed graph,
one fused kernel each. The cuDNN plan cache also holds one built graph
per shape for the life of the process.

## 4. Supported state space

| dimension | variants | status after the change |
|---|---|---|
| predictor execution | graph replay, lazy capture passes, eager fallback (graph disabled, key cap reached, tp above 1) | all reach the chain, all lose the plan builds, unchanged otherwise |
| batch shape | ladder buckets on the graphed path, any batch size on the eager path | unchanged, the eager path benefits most |
| prefill and decode | both call the chain through `_collect_codes` | unchanged |
| process layout | one `pipeline` process, or the vocoder split out | the policy is set per stage factory, so both layouts behave the same |
| tokenizer attention | transformers sdpa by default, `attn_implementation` override to eager or flash_attention_2 | sdpa loses cuDNN, overrides are untouched |
| device | CUDA H100 in CI and serving, CPU in unit tests, MLX and MPS paths in the tree | the flag exists on every build, it only changes CUDA dispatch |
| deterministic inference | `enable_deterministic_inference` serialises preprocessing and vocoder decoding | orthogonal, unchanged |
| unit test talker | head dim 4, so flash and efficient reject it and math serves | unchanged, bit identity still compares the same path on both sides |

Explicitly unsupported: nothing new. Unresolved: none for the design.

## 5. Semantic ownership and design

Behavior: which attention backend serves `scaled_dot_product_attention`
in a Qwen3-TTS stage process. The flag is process global, so its owner
is whoever bootstraps the process, the stage factory, exactly where
MiniMax sets its equivalent. The predictor has no authority over the
tokenizer's and the vocoder's calls, and a runtime context on one
thread has no authority over the others.

Credible designs compared.

- **Backend context around the chain** (`sdpa_kernel` per pass).
  Rejected on ownership: it toggles the process global flags from the
  scheduler thread while up to eight preprocessing threads and the
  vocoder call SDPA, so their backend becomes a function of timing and
  a restore can tear another thread's setting. It also leaves the
  tokenizer's and the vocoder's plan builds in place.
- **Explicit masked attention in the predictor over the fixed
  `predictor_len` cache.** Correct and local, no global state, one
  shape per batch. Cost model: two small matmuls, a mask, a softmax and
  casts per attention call against one fused kernel, 80 calls per
  token, so an estimated fraction of a millisecond of replay per token
  on a 5.6 ms predictor phase, unmeasured. It changes predictor
  numerics away from every fused kernel and leaves the tokenizer's and
  the vocoder's builds in place. Kept as the T22 candidate where replay
  time is the question.
- **Sglang's triton decode kernel over the predictor cache.** Correct
  and local, batch and length agnostic by construction, two kernels per
  call, and it wants the cache slot major with an index table and
  scratch, while `_predictor_k_cache` is `[layer, batch, kv_heads, len, head_dim]`,
  so it needs a cache layout change. Kept under T22 for the same reason.
- **cuDNN attention disabled per Qwen3-TTS stage process, set by each
  stage factory.** Selected. It removes the whole per shape plan class
  for every SDPA caller in the process, is deterministic for every
  thread from before the first call, is placement independent because
  every factory sets it, follows the in repo precedent, and leaves the
  predictor's kernel count and code unchanged.
- **`TORCH_CUDNN_SDPA_DEPRIORITIZED` in the launch environment.**
  Equivalent effect on ordering, but a deployment setting the
  repository cannot enforce. Noted as the operator side equivalent, not
  the mechanism.

Decision: the fourth design. cuDNN convolutions stay enabled, unlike
MiniMax, because the speaker encoder and the Mimi convolutions use
them.

### 5.1 Target mechanics

```
pipeline process (or each stage process when split)
  stages.py create_preprocessing_executor   -> disable_cudnn_attention()  [new]
  stages.py create_sglang_tts_engine_executor -> disable_cudnn_attention()  [new]
  stages.py create_vocoder_executor         -> disable_cudnn_attention()  [new]
      torch.backends.cuda.enable_cudnn_sdp(False)   global context flag
      one INFO line the first time in the process

first SDPA call of any caller in the process, after every factory:
  sdp_utils.cpp:826 check_runtime_disabled_cudnn -> false
  dispatch continues flash -> efficient -> math          [changed: no cudnn]
  predictor: flash (dense, grouped heads, dim 128, bf16)  [expected]
  Mimi encoder, tokenizer decoder: flash or math per their shapes

predictor chain, capture passes, replay, eager fallback: unchanged code
  first eager pass at a new bucket: no plan builds        [changed cost]
```

Ordering. Each factory runs in its stage process before that stage
publishes readiness, the engine factory before sglang loads the model,
and the preprocessing context that the tokenizer serves is installed by
`setup_model` after `pre_infra_setup`. The flag therefore precedes the
first SDPA call of every caller in every layout. Idempotence: the flag
is a boolean set, repeated calls are harmless, the log line is guarded
by a module level flag.

Failure behavior. `enable_cudnn_sdp` cannot fail on a build that has
the attribute. On a torch without it the helper raises at factory time,
which is the right place for a pinned dependency mismatch to surface.

Compatibility. No configuration surface, no persisted state, no
message change. Rollback is a revert, after which the process returns
to cuDNN dispatch on its next start. Mixed versions are not a concern,
the flag is per process.

Observability. The capture timing lines on the measurement branch,
whose first warmup pass shows the absence of plan builds directly, and
the trace scripts of doc 19 for a torch trace. No log line, the flag is
a repair of a dispatch default.

Cost. Startup: none. Steady state: one kernel per attention call as
before, flash instead of cuDNN. First eager pass per bucket: the 15
plan builds are gone, the pass becomes the chain's launch cost plus the
per stream first use costs, estimated 45 to 100 ms from run 4's
remainder column. Memory: the cuDNN plan cache no longer grows.

## 6. Execution plan

Single phase, one commit on `perf/qwen3-tts-predictor-warmup` from
15c4568bb, because the change has one owner, one contract and one
proof boundary, and any slice smaller than "policy plus its tests" is
not verifiable on its own. Rename the branch to
`perf/qwen3-tts-cudnn-attention` at commit time so the name says what
it carries.

Existing files changed:

- `sglang_omni/models/qwen3_tts/stages.py`: new module function
  `disable_cudnn_attention()` that calls
  `torch.backends.cuda.enable_cudnn_sdp(False)`, with a `note(ratish)`
  comment carrying the doc 19 finding, and a call at the top of
  `create_preprocessing_executor`, `create_sglang_tts_engine_executor`
  and `create_vocoder_executor`. No log line: this is a repair of a
  dispatch default, not a policy an operator chooses, and the capture
  timing shows its effect directly.
- `tests/unit_test/qwen3_tts/test_pipeline.py`: three contract tests,
  CPU runnable, one per factory, and a fixture restoring the flag. The
  preprocessing test asserts the flag is off after the factory. The
  vocoder and engine tests assert the ordering contract itself: the
  stubbed tokenizer loader and the stubbed `build` record the flag at
  the moment they run, and the test asserts it was already off, so a
  factory that set the flag after its first attention caller would
  fail. Stubs follow the existing factory tests (`:1484`, `:3070`,
  `:306`).
- `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`: one new
  accelerator test that runs `scaled_dot_product_attention` under
  `torch.profiler` on the predictor's real shape family (batch 2, 16
  by 8 heads, head dim 128, key lengths 1 and 16, bf16, `enable_gqa`)
  with the flag off, and asserts the call succeeds and no recorded
  kernel name contains `cudnn`. This is the dispatch decision for the
  real shapes on the CI H100, which the fake talker with head dim 4
  cannot exercise.
No cookbook change: the comment on the helper is the record.

No file is generated. No configuration schema changes. The predictor
code does not change in this slice.

Exit gate for the commit: the CPU tests pass locally, the file passes
black and isort, and the accelerator module passes on the box.

## 7. Proof

| requirement or invariant | plausible violation | evidence | oracle |
|---|---|---|---|
| no plan build on the first eager pass at a new bucket | the flag is not set in the engine process, or is set after the first call | run 5: c1 and c16 on a measurement branch that is `perf/step-ledger` rebased onto the fix commit, fresh server per point | every capture line shows warmup 1 within about two times warmup 2 and no `cudaGetDeviceProperties` gap in a trace if one is taken |
| the flag is set in every stage process in every layout | a factory does not call the helper, or the vocoder split out keeps cuDNN | the three factory contract tests, and run 5 once with `--vocoder.process vocoder` and once without | flag false after each factory, no plan build gap in either layout |
| flash admits the predictor's shapes | a constraint not read here rejects them and math serves, slower | the new accelerator kernel name test on the CI H100, plus decode `forward_ms` p50 in run 5 | no `cudnn` kernel, and p50 within noise of run 3 |
| graph against eager bit identity | a backend that differs between capture and eager | the existing 55 tests on the box | all pass |
| output quality | flash rounding flips sampled codes often enough to move WER or similarity | the full corpus A/B of doc 15 section 3.2: one server per arm, `benchmark_tts_seedtts` generate only on `seed-tts-eval-arrow` at c1 and c16, then transcribe only and similarity only on each output directory, 15c4568bb against the fix commit, same box, fresh server per point | WER and speaker similarity of the fix commit within the base's run to run spread, never worse beyond it |
| tokenizer and vocoder not slower | flash slower than cuDNN at their lengths | run 5 request view, preprocessing p50 and the vocoder per request timings, against run 3, and the full corpus speed results per arm | not worse beyond noise |
| replay unchanged | a different kernel count or slower kernel in the graph | run 5 decode `host_ms` and `forward_ms` p50 at rows 1 and 16 against run 3 | within noise |

Omitted layers and why: no fault injection, the change has no failure
path beyond a raise at factory time. No distributed tests, the flag is
per process and the tensor parallel case only loses a cost. No
streaming tests, chunk semantics are untouched.

Performance qualification runs against 15c4568bb exactly, the base of
the fix branch, with the measurement branch built the same way on
both sides, one c1 and one c16 window per side, fresh servers, the
same GPU, the benchmark's default warmup inside the window as in every
run so far. Quality qualification is our own A/B with one server per
arm and the full corpus, not the CI harness, which runs a router over
two workers and exists to catch regressions, not to measure a change.

## 8. Implementation state

Implemented on `perf/qwen3-tts-cudnn-attention` (local, not pushed at
the time of writing, the commit hash is in the branch): the helper and
its three calls in `qwen3_tts/stages.py`, three CPU contract tests in
`test_pipeline.py` and the accelerator kernel name test in
`test_predictor_cuda_graph.py`. The log line and the cookbook sentence
of the first cut were removed on review: a repair of a dispatch default
is not something to announce. The CPU tests could not run on the laptop, whose
environment lacks the omni import chain, so the commit's exit gate is
open until the box runs `tests/unit_test/qwen3_tts` and the accelerator
module. Section 7 is the order of what follows.

## 9. Open items, none blocking

- Flash admission of the predictor's shapes on the CI H100. Resolved by
  the accelerator kernel name test. If math serves instead, the plan
  builds are still gone and the replay time gate decides whether the
  T22 explicit attention comes forward.
- Tokenizer and vocoder numerics under flash. Resolved by the full
  corpus A/B. If a threshold fails, the design does not change: the fallback
  is to keep the flag in the engine factory only and pass
  `attn_implementation="eager"` or a per stage decision to the
  tokenizer loaders, and that fallback is decided on the failing
  metric, not in advance.
- The run 5 measurement branch. `perf/step-ledger` sits on 15c4568bb,
  the fix branch sits on 15c4568bb, so rebasing the timing commit onto
  the fix commit is a clean cherry pick. Done at measurement time.

## 10. The PyTorch side, traced

Read on the v2.13.0 clone and its history (anchors verified):

- The dispatcher tries backends in a process wide order and takes the
  first whose checks pass (`sdp_utils.cpp:1049-1096`). The order is
  latched once per process: cudnn, flash, efficient, math when
  `check_prefer_cudnn_attention` holds, otherwise flash first
  (`:110-123`, `Context.h:480-484`).
- cuDNN became first on sm90 and sm100 in #162073 (6f7608d603,
  2025-09-04), whose whole message is "for 2.9". It had been made opt in
  in #138522 after correctness bugs in 2.5, opt in preferred in #149282,
  and the switch was renamed to the opt out `TORCH_CUDNN_SDPA_DEPRIORITIZED`
  in #166201. #171627 added the cuDNN 9.15 gate for a correctness bug.
  No commit in that chain states a performance rationale or considers
  problem size.
- The only small size check cuDNN makes is to reject key length 1
  (`:678-683`), which is why the trace showed 75 cuDNN launches for 80
  attention calls: lengths 2 to 16 go to cuDNN, length 1 to flash.
- The cuDNN plan cache is keyed on exact sizes and strides including
  batch and key length (`MHA.cpp:198-223`, `:275-280`), is thread local
  and unbounded (`:343-399`), and every miss builds plans
  (`:640-645`). PyTorch does not use the cudnn frontend's dynamic
  shape or kernel cache facilities, which the pinned frontend v1.24.0
  exposes. The opt in `TORCH_CUDNN_SDPA_AVOID_RECOMPILE` only applies
  when q, k, v and the output are BSHD contiguous (`:168-181`), so it
  cannot apply to a sliced BHSD KV cache like the predictor's, and even
  where it applies the batch stays in the key.
- Upstream, the per shape build is tracked as intentional in the open
  issue #154602 (large prefill shapes, "recompilation"), with the opt
  in above as its mitigation. The decode shaped consequence, a host
  stall per new key length for microsecond kernels, is not reported.
- Sglang core disables cuDNN attention in its diffusion runtime for a
  different reason, a correctness crash in 2.5
  (`multimodal_gen/runtime/platforms/cuda.py:43-45`). That module is
  not imported in the TTS process, so it did not protect us.

Consequences for this plan: the repair stands as designed, the same
seam sglang core uses, and it is not a workaround for a bug in our
chain. Two follow ups, tracked as tasks:

- T29 File the PyTorch issue: on torch 2.13.0, H100, CUDA 13 and cuDNN
  above 9.15, `scaled_dot_product_attention` with q `(B, 16, 1, 128)`
  and k, v `(B, 8, L, 128)` in bf16 with `enable_gqa` is routed to
  cuDNN for every L of 2 or more, whose graph cache is keyed on exact
  sizes and rebuilt on every miss, so a decode loop pays one 23 to 37
  ms host stall per new (batch, key length), fifteen per new batch
  size in our trace, for kernels that run in microseconds, and the
  documented recompile switch cannot apply to a BHSD cache. Proposal:
  skip cuDNN for tiny problems the way key length 1 is skipped, or use
  the frontend kernel cache, and until then document
  `enable_cudnn_sdp(False)` as the intended workaround for eager
  decode. Attach the run 4 capture lines and the trace accounting.
- T30 The same policy for the talker, MOSS-TTS-Local and MOSS-TTS Delay
  processes once their first shape costs are measured (doc 19 A5).
