# 21. Qwen3-TTS code predictor: startup capture, capture hygiene, decode attention kernel

Plan status: **Conditional**. Slice A (capture hygiene, startup capture,
deterministic mode repair) is Ready. Slice B (the attention kernel) has one
measured branch, the replay time gate of section 8.3, whose outcome selects
between keeping the kernel and keeping the SDPA call inside the graph. No
other decision is delegated to the implementer.

Amendment, 2026-09-03, after doc 22. The full-corpus A/B showed the cuDNN
repair (doc 20) trades a per process warm up cost for a per step replay
cost, so cuDNN attention stays on and the repair branch stays unmerged.
The decisions below that this changes: D13 replaces the base branch of
section 7, the startup capture of 6.3 now also builds cuDNN's attention
plans in its warmups, and the gate of 8.3 compares against the cuDNN
replay. The thread local plan cache note in 6.3 and tasks T39 and T40
follow from the same doc. Slice A was implemented on
`perf/qwen3-tts-predictor-capture` from upstream main on 2026-09-03 and
validated by the final A/B (doc 23 records it). Two deviations from the
text below were made on review: a startup capture failure raises and
fails the boot, as sglang's capture does, instead of degrading like the
lazy path, and the fused addmm policy (sm90 and not batch invariant) is
resolved once with the graph policy instead of per call. The names in
the code are `_predictor_capture_mode`, `_predictor_capture_session`,
`uniform_predictor_graph_signature` and `_predictor_signature_terms`.

Task class: cross-boundary. The change crosses the model file, the engine
builder lifecycle, the pipeline process and thread topology, sglang's kernel
and runner contracts, and PyTorch's capture and allocator contracts.

Revisions read in this session (every anchor below is from these):

| Repository | Path | Revision |
| --- | --- | --- |
| sglang-omni fix branch | `/Users/ratish/sglang-omni/.worktrees/qwen3-tts-predictor-warmup` | `a4f3590b2` on `perf/qwen3-tts-cudnn-attention`, base `15c4568bb` |
| sglang-omni upstream main | `upstream/main` | `fa1ea43dc`. `git diff --stat 15c4568bb upstream/main` over `sglang_model.py`, `model_runner.py`, `engine_builder.py`, `config.py` and `test_predictor_cuda_graph.py` is empty, so the predictor code this plan changes is identical on both |
| sglang | `/Users/ratish/sglang` | `71de97b264` (v0.5.18) |
| PyTorch | `/Users/ratish/pytorch` | `cf30153c4c` (v2.13.0) |
| qwen-tts 0.1.1 | scratchpad `qwen_tts_pkg/` | wheel unpacked |
| checkpoints | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | `config.json` and `generation_config.json` fetched from the hub on 2026-09-03 |

Doc 19 sections 9.3 to 9.5 and doc 20 hold the evidence this plan builds on.
This document does not restate the cuDNN finding.

## 1. Scope

### 1.1 Target behaviour

1. The predictor's CUDA graphs for the checkpoint's default sampling
   signature exist before the tts_engine stage reports ready. No serving
   step pays a capture for that signature at any batch bucket.
2. Every predictor capture, at startup and the lazy ones that remain, runs
   the way sglang runs its decode captures: one capture stream per talker
   for warmups and capture, one shared memory pool, the same code branches
   in the warmup passes and the capture pass, the cyclic collector held off
   for the duration, and no reference cycle between the talker and its
   graph objects.
3. Under `enable_deterministic_inference` the predictor graphs capture and
   replay the batch invariant GEMM that eager execution uses, so the mode's
   guarantee holds on the graph path. This repairs fault 1 of tracker issue
   #1936.
4. The predictor attention runs on sglang's decode attention kernel over a
   slot-major private cache with batch and key length as runtime values,
   one attention contract for the talker path and the backbone. Slice B,
   gated by the replay time measurement.

### 1.2 Requirements (user, this session and the standing rules)

- R1 sglang, sgl_kernel, torch and qwen-tts stay pinned and unpatched. Every
  change is in sglang-omni. The kernel is imported from the pinned sglang.
- R2 The capture routine mirrors sglang's decode capture line by line where
  the reason behind the sglang line applies to us. Every deviation is
  recorded with its reason (section 6.1).
- R3 No assumptions in the plan. Anything not read in code is a task.
- R4 Unit tests assert the contract at the moment it holds, not a mocked
  return. Fakes use the resolved production geometry.
- R5 Validation is our own A/B: one server per arm, the full seed-tts corpus
  at c1 and c16, quality and speed from the benchmark, never the CI harness.
- R6 A bug repair is not announced to the operator. The startup capture is
  state the operator reads, so one line at the end of the startup capture is
  allowed, matching the vocoder's line (`streaming_vocoder.py:323`).
- R7 Code comments in the `# note(ratish):` style, no backticks in comments
  or docstrings, sglang's code style guidance (`contribution_guide.mdx`,
  "Code style guidance"): cache per-call booleans in `__init__`, no host
  syncs on the hot path, pure functions, core data structures at the top of
  the file.

### 1.3 Non-goals

- T22, an explicit masked attention over the fixed cache in one launch.
  Recorded as the alternative the replay gate may bring forward.
- T30, the same policy for the Qwen3-Omni talker predictor
  (`qwen3_omni/components/talker.py:1715`) and the MOSS attentions.
- A4, a startup warmup request per stage.
- Fault 2 of #1936 (an illegal instruction under load with the predictor
  graph off). It is not on the predictor path and needs its own
  investigation.
- The vocoder graphs, the prefill graph, the chain's op count outside the
  attention (doc 19 option D).

## 2. Evidence ledger

### 2.1 Repository facts, omni

- F1 The predictor chain per semantic token is 16 one-token forwards over 5
  layers with a private cache of `predictor_len = num_code_groups + 1`
  slots (`sglang_model.py:454`, cache allocation `:470`), written at
  `cache_len` and read as the slice `[:cache_len + 1]`
  (`:1796`, `:1798`), attention by `scaled_dot_product_attention`
  (`:1802`, `:1809`) followed by a transpose and a reshape that copies
  (`:1816`).
- F2 One graph per `(bucket, signature)` is captured lazily inside the
  serving step that first needs it (`_predictor_forward_graphed`,
  `:1285`, construction `:1329`), by `_PredictorDecodeGraph._capture`
  (`:125`): two warmup passes on a fresh `warmup_stream` (`:133`, `:136`),
  then the capture on a second fresh `capture_stream` (`:145`) under
  `torch.cuda.graph(..., pool=shared, capture_error_mode="thread_local")`
  (`:151`). The graph object stores the talker (`self.model = model`,
  `:99`) and the talker stores the graph objects in `_predictor_graphs`,
  a reference cycle.
- F3 Three branches of the chain run only while a stream is capturing:
  the fused codec embedding gather (`:1389` to `:1392`), the fused seeded
  sampler (`:1608`), and the fused `addmm(out=)` for the attention output
  projection (`:1714` to `:1716`, `:1745`). The warmup passes therefore
  run different kernels from the capture pass.
- F4 The policy switch `_predictor_graph_enabled` is `None` at
  construction and resolved at the first decode (`:535`,
  `_resolve_predictor_graph_enabled` `:1275`, read at `:1285`) from the
  env var, `disable_cuda_graph` and `tp_size == 1`. It does not check the
  device type.
- F5 Buckets come from sglang's resolved `cuda_graph_config.decode.bs`
  clamped to `max_running_requests` with the maximum appended
  (`:1181`, `generation_batch_policy.py:28`, `:38`). With the builder
  defaults (`engine_builder.py:56`, `:58`) the ladder is
  `(1, 2, 4, 8, 12, 16)`.
- F6 The signature is `("argmax", 0, False, False)` when no row samples,
  else `("sampled", quantized max top_k, has_top_p, has_unbounded_top_k)`
  (`:1212`), computed in `prepare_decode_buffers` (`:992`, quantization
  `:1074`, `:1081`) from six per request fields.
- F7 Those fields default to `subtalker_dosample=True`,
  `subtalker_top_k=50`, `subtalker_top_p=1.0` when the merged generation
  kwargs carry no override (`request_builders.py:1241` to `:1244`), and
  the merge takes the checkpoint's `generation_config.json` before the hard
  defaults (`qwen_tts/inference/qwen3_tts_model.py:287`, `:302` to
  `:307`, called at `request_builders.py:1067`). Both Base checkpoints
  ship `subtalker_dosample: true`, `subtalker_top_k: 50`,
  `subtalker_top_p: 1.0`. The resulting default signature is
  `("sampled", 50, False, False)`, the key reported in #1936 and in the
  run 2 to 4 capture lines.
- F8 The model runner calls `code_predictor_forward` from both
  `post_prefill` and `post_decode` (`model_runner.py:72`, `:81`, `:228`)
  on the current stream inside the execution bridge's `forward_context`,
  which switches no stream (`sglang_execution.py:80` to `:101`). Async
  decode is off for Qwen3-TTS: the scheduler default is
  `enable_async_decode=False` (`omni_scheduler.py:189`, `:526`) and the
  Qwen3 builder passes no override (`engine_builder.py:127` to `:132`).
- F9 The engine builder lifecycle is: infrastructure with graphs deferred
  (`engine_factory.py:188`), `setup_model` (`:201`, loads the speech
  tokenizer and builds the qwen-tts wrapper, `engine_builder.py:70`,
  `:95`), `validate_after_model_setup` (`:209`), `compile_model` (`:211`),
  sglang's backbone graphs (`:214`), `post_cuda_graph_setup` (`:215`), then
  `setup_model_resources(model, server_args, generation_cuda_graph_enabled)`
  (`:226`) with the comment that model-local graphs must come after
  sglang's. The Qwen3-TTS builder overrides neither hook. Qwen3-ASR
  captures its encoder graphs in `setup_model_resources`
  (`qwen3_asr/engine_builder.py:327`).
- F10 The vocoder captures its graphs at factory time on one capture stream
  with one pool, two warmups per shape on that stream, and a try/except per
  shape (`streaming_vocoder.py:281` to `:326`, called from `warmup_now`
  `:555`).
- F11 Stage factories run under a per GPU startup lock
  (`stage_workers.py:848`, `:883`) and the process signals ready after
  every stage's `start()` (`:448`, `:480`). Each stage's scheduler loop is
  its own thread in the process (`pipeline/stage/runtime.py:241`). The
  default topology puts preprocessing, tts_engine and vocoder in one
  process (`config.py:57`, `:63`, `:72`). Preprocessing runs up to 8
  threads (`stages.py:127`). Under deterministic inference the vocoder
  graphs are off and the engine gets `enable_deterministic_inference`
  (`config.py:53`, `:90`).
- F12 The existing tests build a fake talker with `HIDDEN = 8`,
  `NUM_HEADS = 2`, `NUM_KV_HEADS = 1`, `HEAD_DIM = 4`
  (`test_predictor_cuda_graph.py:40` to `:43`, `:82`). They assert graph
  versus eager bit identity, one pool across keys (`:987`), the thread
  local error mode by AST (`:774`), the fused paths running only under
  capture (`:356`, `:386`), no host readback on dispatch (`:736`), and the
  `disable_cuda_graph` gate (`:1120`).
- F13 The omni process already imports sglang kernels directly
  (`moss_tts/attention.py:14`, `moss_tts/sampling_kernels.py:9`).
  `predictor_kernels.py` guards `import triton` with try/except (`:7` to
  `:11`) because the model file is imported on non-CUDA installs.
- F14 The cuDNN repair of doc 20 is on the fix branch: every Qwen3-TTS
  stage factory calls `disable_cudnn_attention()` (`stages.py:43`, `:130`,
  `:153`, `:192`).

### 2.2 Repository facts, sglang v0.5.18

- S1 The decode runner captures at startup inside `model_capture_mode()`
  (`decode_cuda_graph_runner.py:460`), `capture()` (`:997`) warms kernels
  once (`:1000`), holds `freeze_gc` (`:1035`), enters `graph_capture()`
  which allocates one stream and makes it wait on the current stream
  (`parallel_state.py:586`, `:593`, `:606`), binds that stream and the
  process pool in `capture_session` (`full_cuda_graph_backend.py:71` to
  `:75`, pool from `pool.py:34`), and captures the buckets in descending
  order "so cuda graphs share memory better" (`:1065`, `:1067`).
- S2 Per shape, `capture_one` runs two warmups on the bound stream with a
  device synchronize and a TP barrier before each (`:103` to `:113`), then
  captures with `torch.cuda.graph(graph, pool=self._pool,
  stream=self._capture_stream)` (`:114`, `:128`). Replay is
  `graph.replay()` returning the static outputs (`:144`).
- S3 `model_capture_mode()` sets a process global read by model code to
  pick capture time branches (`capture_mode.py:32`, `:46`, `:85`, `:90`).
  It is set for the warmups and the capture alike, because it wraps the
  whole `capture()`.
- S4 `freeze_gc` collects, freezes for the capture, unfreezes and collects
  (`base_cuda_graph_runner.py:46` to `:61`).
- S5 The decode attention kernel: `decode_attention_fwd` (`decode_attention.py:1163`)
  asserts `max_kv_splits == attn_logits.shape[2]`,
  `q.shape[0] <= kv_indptr.shape[0] - 1` and `q.shape[0] <=
  attn_logits.shape[0]` (`:1186` to `:1188`), dispatches on
  `kv_group_num = q_heads // kv_heads` (`:1194`), and the grouped path
  reads batch and key length from `kv_indptr` at runtime by design
  (`:609` to `:612`). Stage 1 (`_fwd_grouped_kernel_stage1` `:540`,
  launcher `:776`, `BLOCK = 32` `:797`, `BLOCK_H = 16` `:822`,
  `num_warps = 4` `:828`, grid `(batch, head_tiles, max_kv_splits)`
  `:846`) computes `qk = tl.dot(q_k, k)` (`:680`) and `acc += tl.dot(p.to(v.dtype), v)`
  (`:742`). Stage 2 (`:911`, launcher `:997`) merges the splits and
  writes `o` scaled by `v_scale` (`:992`). The MLA tuning branch is HIP
  only (`:151`, `:164`, `:1125`), so on CUDA the launch plan is
  `(False, 0)` with no runtime context read.
- S6 The kernel's buffer contract in the triton backend: `kv_indptr` int32
  of size `max_bs + 1` (`triton_backend.py:304`), `kv_indices` int64
  (`:1047` to `:1049`), `attn_logits` fp32 `(tokens, heads, max_kv_splits,
  v_head_dim)` (`:1012`), `attn_lse` fp32 `(tokens, heads, max_kv_splits)`
  (`:1030`), `num_kv_splits` int32 filled with `max_kv_splits` (`:1037`),
  `o` as `torch.empty_like(q)` (`:1805`), `sm_scale=layer.scaling`,
  `k_scale`, `v_scale` `1.0` when unquantized (`:1912` to `:1934`). The
  backend wraps the kernel in `torch.compiler.disable` (`:160`). KV
  buffers are 3-D `[slots, kv_heads, head_dim]` with page size 1
  (`decode_attention.py:195`, `_extract_kv_strides`).
- S7 sglang's own test builds `kv_indptr` as a cumsum of lengths,
  `kv_indices` as `torch.arange`, scratch `attn_logits` and `attn_lse`
  sized by `max_kv_splits`, `num_kv_splits` int32, and compares the bf16
  output against a fp32 stable-softmax reference at `atol=1e-2,
  rtol=1e-2` (`test_triton_attention_kernels.py:106`, `:687`, `:746`).
  The grouped path is exercised down to head dim 13, which rounds to a
  16 wide tile (`:855`).
- S8 Deterministic inference: `ModelRunner.initialize` loads the model
  (`model_runner.py:630`) and only then enables batch invariant mode
  (`:661`, `:757`), which re-registers `aten::mm`, `aten::addmm`,
  `aten::_log_softmax`, `aten::mean.dim`, `aten::rms_norm`,
  `aten::mm.dtype` and `aten::bmm` (`batch_invariant_ops.py:980` to
  `:1002`) and exposes `is_batch_invariant_mode_enabled` (`:976`,
  exported in `batch_invariant_ops/__init__.py:8`). The `.out` overloads
  are not registered.
- S9 sglang serves no model with a small private growing cache inside one
  step, and its torch native backend has no backend control (doc 19,
  section 9.5).

### 2.3 External facts, PyTorch v2.13.0 and CUDA

- P1 `torch.cuda.graph.__enter__` synchronizes the whole device and
  empties the caching allocator before `capture_begin`
  (`torch/cuda/graphs.py:439`, `:449`, `:462`). The class keeps one
  default capture stream when none is passed (`:408`, `:423`). Its
  docstring: "For effective memory sharing, if you pass a pool used by a
  previous capture and the previous capture used an explicit stream
  argument, you should pass the same stream argument to this capture"
  (`:398`).
- P2 The allocator only reuses a cached block on the stream that freed it
  (`CUDACachingAllocator.cpp:3710`, `:3720`). With a fresh capture
  stream per capture, blocks the shared pool freed under an earlier
  capture cannot serve a later one.
- P3 `capture_begin` rejects the default stream (`CUDAGraph.cpp:101`,
  `:116`), routes allocations of the capturing stream into the pool
  (`:139`, `:150`, filter `:85`), and `reset` releases the pool
  (`:307`, `:347`). Doc 19 section 9.3 recorded the effect of a
  `CUDAGraph` finalizer running while another capture is open.
- P4 The cuBLAS handle is created lazily per thread on first use
  (`CublasHandlePool.cpp:400`, `:429`, `:104`). A first use inside a
  capture is the `cublasCreate` failure of #1936.
- P5 The CUDA graphs note: "Before capture, warm up the workload to be
  captured by running a few eager iterations. Warmup must occur on a side
  stream" (`docs/source/notes/cuda.md:1396` to `:1397`).
  `make_graphed_callables` uses three warmups (`graphs.py:494`).
- P6 `torch.compiler.config.force_cudagraph_gc` defaults to `False`
  (`torch/compiler/config.py:182`), so `torch.cuda.graph` does not collect
  before capture.
- P7 PyTorch documents that only one capture may be underway at a time in
  a process (`torch/cuda/graphs.py:421`). Two threads capturing at once is
  outside the contract.

### 2.4 Checkpoint facts

- C1 Both Base checkpoints have `num_code_groups = 16`, a predictor of 5
  layers, 16 attention heads, 8 key value heads, head dim 128, hidden
  1024, vocab 2048 (config.json `talker_config.code_predictor_config`).
  So `predictor_len = 17`, `kv_group_num = 2`, `head_tiles = 8`, and
  16 of the 17 slots are written per token (cache_len runs 0 to 15).
- C2 Both ship the default sampling of F7.

### 2.5 Inferences

- I1 From F3 and P4: any first use that only the capture branches perform
  lands inside the capture. Today that is the cuBLAS handle under batch
  invariant mode (#1936) and the first launch of each fused Triton kernel
  (its JIT compile and module load). With a warm Triton disk cache the
  second cost is small, which is why runs 3 and 4 saw capture passes of
  33 to 49 ms. On a cold cache it is a compile inside the capture. The
  size of that cold cost is not measured (task T33).
- I2 From F2, P1 and P2: a lazy capture stalls every stream of the process
  (the vocoder's included) twice per key, once per `torch.cuda.graph`
  entry and once for the second stream's wait, and empties the allocator
  cache of the whole process each time. Startup capture moves both to
  the startup window where they are free.
- I3 From F2 and P3: the cycle plus a cyclic collection during a capture
  is the mechanism of the order dependent test failure of doc 19 section
  9.3. In production the talker never becomes garbage, so the same
  finalizer can only run if some other cyclic garbage holding a
  `CUDAGraph` is collected during a capture. Removing the cycle and
  disabling the collector for the capture window closes both.
- I4 From S5 and C1: at `max_kv_splits = 1` the stage 1 grid is
  `(batch, 8, 1)` and stage 2 is `(batch, 16)`, two launches per
  attention against one to two for the flash path (the flash split-kv
  combine is a second launch when its heuristic splits, not read in this
  session, task T34). The replay cost difference is bounded by
  roughly 80 extra graph nodes per step and is the subject of the gate in
  section 8.3.
- I5 From S5: `tl.dot` operands are `[16, BLOCK_DMODEL]` by
  `[BLOCK_DMODEL, 32]` and `[16, 32]` by `[32, BLOCK_DV]`. sglang's
  smallest tested head dim is 13, tile 16 (S7). The test fake's head dim
  of 4 (F12) would give tiles of 4, which sglang has never run. The fake
  moves to the production geometry (R4), which removes the question for
  the tests. Whether Triton 3.7.1 accepts tiles below 16 for `tl.dot`
  stays unverified and unneeded (task T35 records it).
- I6 From S8: `is_batch_invariant_mode_enabled()` is `False` while the
  talker's `__init__` runs, so the policy must be read after
  `ModelRunner.initialize`, that is in the builder hook of F9 or at the
  lazy resolution of F4.

### 2.6 Decisions

- D1 Startup capture lives in a new `Qwen3TtsEngineBuilder.setup_model_resources`
  override (F9 hook, Qwen3-ASR precedent), gated by
  `generation_cuda_graph_enabled` and the talker's resolved policy.
- D2 The startup set is every bucket of the ladder for the default
  signature of F7, captured in descending bucket order. Other signatures
  keep the lazy bounded capture (32 keys, 8 failures). Reason: the
  signature is batch level, one request with `top_p < 1` moves the whole
  batch to a new signature, and one bounded capture per new signature
  beats running that batch eager for its lifetime.
- D3 Two warmups per shape, the sglang count (S2). The startup window
  makes the count immaterial, and the count is not tuned here.
- D4 One capture stream per talker, created once, reused by every capture
  (startup and lazy), warmups on it, capture on it, the shared pool as
  today. Reason: P1 and P2.
- D5 An explicit talker-scoped graph mode flag replaces the three
  `is_current_stream_capturing()` reads (F3) and the `for_capture`
  parameter. It is set for the warmups and the capture pass, never for
  eager execution or replay. It mirrors S3 in semantics but is not
  sglang's global, because sglang's global also flips
  `disable_dispose_tensor` and is read by sglang's own model code, none of
  which the predictor path touches.
- D6 The fused `addmm(out=)` is taken only when the graph mode flag is on
  and batch invariant mode is off. Under batch invariant mode the graph
  path runs `o_proj` plus the residual add, the same `aten::mm` the eager
  path runs (S8), which keeps the mode's guarantee on the graph path.
- D7 The graph object stops referencing the talker. `_capture` receives
  the talker as an argument, `replay` needs only the object's own buffers
  and graph.
- D8 The collector is disabled for the duration of every capture session
  with `gc.disable()` and restored in `finally`, no collection. Reason:
  sglang's collect and freeze (S4) exists for the cost of scanning a large
  heap during dozens of captures at startup. Our hazard is the finalizer
  of I3, which disabling covers, and a collection inside a serving step is
  a latency cost with no benefit for us.
- D9 The predictor graph policy is resolved once, at the builder hook (or
  at the first decode for talkers built without the builder, as the tests
  do), and adds `device.type == "cuda"` to the existing three conditions.
- D10 Slice B keeps the SDPA call for non-CUDA devices and imports the
  kernel under the same optional import guard as `predictor_kernels.py`.
  The kernel path is the CUDA path.
- D11 The `capture_error_mode="thread_local"` stays. Reason: the tts_engine
  scheduler thread captures while the preprocessing threads and the
  vocoder's scheduler and worker threads issue CUDA work in the same
  process (F11). sglang captures with the default global mode because its
  process has no such threads. The existing AST test keeps guarding it.
- D12 Slices A and B are separate branches and separate A/B runs. Slice A
  changes no kernel and must be bit identical to today's graph path
  outside deterministic mode. Slice B changes the attention numerics and
  needs its own quality verdict.
- D13 cuDNN attention stays on. Both slices branch from upstream main and
  not from the cuDNN repair branch. Reason: doc 22, the repair's replay
  cost of 0.13 ms per step at one row and 0.21 ms at sixteen rows outlives
  its per process savings after about 700 requests.

### 2.7 Open questions

None that change the architecture. Section 10 lists the measured gate and
the tasks.

## 3. Current mechanics

### 3.1 The call path

```text
OmniScheduler loop (thread scheduler-tts_engine, runtime.py:241)
  -> ModelRunner.execute (model_runner/base.py:229)
     -> _execution_context = bridge.forward_context (sglang_execution.py:80)   no stream switch
     -> _prepare_and_forward: before_decode -> prepare_decode_buffers (model_runner.py:69)
        -> tp_worker.forward_batch_generation  (sglang backbone graph replay or eager)
        -> _sample_next_token_ids
     -> post_decode / post_prefill (model_runner.py:81 / :72)
        -> _collect_codes (:210) -> model.code_predictor_forward (:228)
           -> _predictor_forward_graphed (sglang_model.py:1285)
              key = (bucket, *signature)          bucket :1206, signature :1212
              hit  -> graph.replay(...)            :167   copies + zero pad + replay, slices
              miss -> _PredictorDecodeGraph(...)   :1329  capture INSIDE this step
              None -> _code_predictor_forward_incremental (:1358) eager
```

### 3.2 The capture today, per key

```text
_PredictorDecodeGraph.__init__ (:80)      self.model = model (:99)   cycle with talker._predictor_graphs
  _capture (:125)
    _predictor_graph_capture_state(bucket, signature)   swaps the sub-sampling state, restores in finally
    warmup_stream = Stream()  (:133)      wait_stream(current)
      2 x _code_predictor_forward_incremental(for_capture=True)   (:136)
         branches taken: eager sampler shape (row uniform), o_proj via F.linear, ATen embedding,
                         SDPA over the cache slice
    current.wait_stream(warmup_stream)
    capture_stream = Stream() (:145)      wait_stream(current)
      torch.cuda.graph(graph, pool=shared, stream=capture_stream, thread_local)  (:151)
         __enter__: torch.cuda.synchronize(); torch.cuda.empty_cache()   (graphs.py:439, :449)
         branches taken: fused sampler (:1608), fused embedding gather (:1389), fused addmm (:1716)
                         SDPA over the cache slice
    current.wait_stream(capture_stream)
```

Costs measured before the cuDNN repair (doc 17, doc 19 sections 7 and 9):
every key 496 to 565 ms inside a serving step, of which the first warmup
carried the cuDNN plan builds. After the repair the expected residual per
key is two warmups plus the capture pass plus instantiate, unmeasured until
run 5 (doc 20 section 7).

Defects, each tied to a fact:

| Id | Defect | Fact |
| --- | --- | --- |
| K1 | Capture inside a serving step, two device-wide syncs and one allocator flush per key, at every new bucket of the ramp | F2, P1, I2 |
| K2 | Fresh warmup and capture streams per key, pool blocks not reusable across keys | F2, P1, P2 |
| K3 | Warmup passes and capture pass run different kernels, first uses land inside the capture | F3, I1, #1936 |
| K4 | Fused addmm escapes batch invariant mode, capture fails and would break the guarantee if it succeeded | F3, S8, #1936 |
| K5 | Talker to graph cycle, collector free to run during capture | F2, I3 |
| K6 | Policy resolved at first decode, no device check | F4 |
| K7 | Attention on PyTorch's dispatcher over a BHSD slice with a transpose copy per attention, backend policy set per process | F1, F14, doc 20 |

### 3.3 sglang's capture, the reference

```text
DecodeCudaGraphRunner.__init__
  with model_capture_mode():                     capture_mode.py:85   global flag ON for warmups and capture
    capture()                                    decode_cuda_graph_runner.py:997
      warmup()                                   :1000  once per process (kernel loads, autotune)
      with freeze_gc(enable_cudagraph_gc):       :1035  base_cuda_graph_runner.py:46
        with graph_capture() as ctx:             parallel_state.py:2075 -> :586
           stream = Stream(); stream.wait_stream(current)     :593, :606
           with backend.capture_session(stream): full_cuda_graph_backend.py:71  pool bound once
             for bs in reversed(capture_bs):     :1067
               capture_one_shape(bs)             :1118
                 forward_batch = capture_prepare(bs)     static buffers sliced
                 backend.capture_one(key, run_once)      :81
                   2 x (synchronize; tp barrier; run_once)   :105 to :113   on the bound stream
                   graph = CUDAGraph()
                   with torch.cuda.graph(graph, pool, stream): out = run_once()   :128
                   graphs[key] = graph; outputs[key] = out
replay: load_batch fills static buffers, backend.replay(key) -> graph.replay(); slice [:raw_bs]
```

### 3.4 Deterministic inference on the predictor

`ModelRunner.initialize` loads the model then enables batch invariant mode
(S8). From then on `aten::mm` and `aten::addmm` dispatch to sglang's
Triton GEMMs. The predictor's eager path calls `o_proj` (`F.linear`,
`aten::mm`) and is covered. The graph path calls `torch.addmm(...,
out=residual_2d)` (`:1745`), `aten::addmm.out`, which is not registered,
so it reaches cuBLAS. Nothing else in the process touched cuBLAS, so the
first cuBLAS use is the handle creation inside the capture (P4), and the
capture is invalidated. #1936 reproduces this without a model.

### 3.5 Topology

| Participant | Process (default layout) | Thread | Touches CUDA during a lazy capture |
| --- | --- | --- | --- |
| preprocessing executor | pipeline | up to 8 worker threads | yes, speech tokenizer encode |
| tts_engine scheduler | pipeline | `scheduler-tts_engine` | the capturing thread |
| vocoder scheduler and its two async workers | pipeline (or `vocoder` with `--vocoder.process vocoder`) | `scheduler-vocoder`, `qwen3-tts-vocoder-initial`, `qwen3-tts-vocoder-followup` | yes, decode and D2H |

Startup captures run inside the factory under the GPU startup lock
(F11), so the vocoder's own startup capture and the predictor's cannot
interleave. The process reports ready after all factories returned
(`stage_workers.py:480`).

## 4. Supported state space

| Dimension | Variants | Status after this plan | Invariant, consequence, proof |
| --- | --- | --- | --- |
| Capture time | startup default signature, lazy other signatures | newly supported (startup), unchanged (lazy) | keys are `(bucket, signature)`, startup set is the ladder for the default signature, lazy set bounded by 32 keys and 8 failures. Proof: T-A5, T-A6, ledger run 6 shows no capture line inside serving steps of a default-signature run |
| Warmup and capture branches | same kernels in warmups and capture | migrated | the graph mode flag is on for both. Proof: T-A2 counts fused sampler launches across the three passes |
| Streams | one capture stream per talker | migrated | every `torch.cuda.graph` call of one talker receives the same stream, warmups run on it. Proof: T-A3 |
| Pool | one shared pool per talker | unchanged | Proof: existing `test_graph_keys_share_memory_pool` |
| Collector | disabled inside capture sessions | newly supported | `gc.isenabled()` is `False` inside `run_once` under capture and restored after. Proof: T-A4 |
| Object graph | no talker to graph cycle | migrated | the graph object holds no reference to the talker. Proof: T-A7 |
| Deterministic inference | batch invariant mode on | migrated from failing to supported | fused addmm off, graphs capture, graph equals eager bits. Proof: T-A8, box run with the flag (section 8.2) |
| Device | CUDA sm90 | validated | fused addmm keeps its sm90 condition (`:1718`), kernel path CUDA only |
| Device | other CUDA architectures | unchanged, eager addmm branch, kernel path on | not validated on the box, T36 |
| Device | XPU, CPU, MPS | explicitly unsupported for graphs and the kernel | policy adds `device.type == "cuda"`, SDPA path retained. Proof: T-A9 |
| TP | `tp_size > 1` | unchanged, eager | policy unchanged (`:1275`) |
| Server flags | `disable_cuda_graph`, `SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=0` | unchanged, eager, no startup capture | Proof: existing tests plus T-A9 |
| Buckets | resolved ladder, live batch below or equal to bucket, padded rows zeroed | unchanged | Proof: existing padded bucket tests |
| Batch of zero rows | eager return | unchanged (`:1296`) | |
| Sequence length | one token per row (prefill's first code and every decode step) | unchanged | |
| Async decode | off for Qwen3-TTS | unchanged | F8 |
| Retraction and re-prefill | predictor is stateless across steps except the per-step cache | unchanged | |
| Process layout | shared pipeline process, split vocoder process, preprocessing in its own process (PR #1923, open) | unchanged | thread_local error mode and the gc window cover the shared layouts, split layouts are the easier case |
| Concurrent captures from two threads | unsupported by PyTorch (P7) | unchanged, not reachable today | the vocoder captures at startup only (F10) and the predictor's startup capture runs under the factory lock before serving. A future lazy vocoder capture (PR #1855, open) would need serialization against the predictor's lazy capture, recorded as T38 |
| Shutdown | graphs die with the process | unchanged | no capture in flight after `stop()` since the scheduler thread has exited |
| Kernel geometry (Slice B) | 16 heads, 8 kv heads, head dim 128, 17 slots, both Base checkpoints | newly supported | C1, tables sized from the config, no hardcoded 17 |
| Kernel geometry (Slice B) | other head dims | supported by the kernel for powers of two at or above 16, unvalidated elsewhere | test T-B1 covers 128 and 64 |

## 5. Ownership

| Behaviour or resource | Owner after the plan |
| --- | --- |
| When the default signature's graphs exist | `Qwen3TtsEngineBuilder.setup_model_resources`, through the talker's `capture_predictor_graphs` |
| Graph policy (enabled, batch invariant, device) | the talker, resolved once in `_resolve_predictor_graph_policy`, called from the builder hook or the first decode |
| Capture stream, pool, collector window, mode flag | the talker, in `_predictor_capture_session` and `_predictor_graph_mode` |
| One graph's buffers, graph object, outputs | `_PredictorDecodeGraph`, without a reference to the talker |
| Key cache, capacity, failure counters | the talker, unchanged |
| Predictor cache layout, index tables, scratch (Slice B) | the talker's `__init__` |
| Attention numerics (Slice B) | sglang's `decode_attention_fwd`, pinned |
| Default signature | derived by the talker from the merged generation defaults the builder hands it |

## 6. Target design

### 6.1 The capture routine, sglang line by line

```text
talker.capture_predictor_graphs(signature)                 NEW, startup
  keys = [(b, *signature) for b in reversed(self._predictor_graph_batch_sizes)]
  with self._predictor_capture_session():                  NEW
     for key in keys: self._capture_predictor_graph(key)    NEW, shared with the lazy path

talker._predictor_capture_session()                        NEW
  stream = self._predictor_capture_stream (created once per talker)
  stream.wait_stream(torch.cuda.current_stream(device))
  gc_was_enabled = gc.isenabled(); gc.disable()
  try: yield
  finally: current.wait_stream(stream); if gc_was_enabled: gc.enable()

talker._capture_predictor_graph(key)                        NEW body, replaces _PredictorDecodeGraph._capture
  graph = _PredictorDecodeGraph(bucket, signature, buffers)  no model reference
  with self._predictor_graph_capture_state(bucket, signature), self._predictor_graph_mode():
     with torch.cuda.stream(stream):
        for _ in range(2): run_once()
     with torch.cuda.graph(graph.graph, pool=self._predictor_graph_memory_pool(),
                           stream=stream, capture_error_mode="thread_local"):
        graph.result_codes, graph.summed_embeddings = run_once()
  on exception: graph.graph.reset(), re-raise (today's cleanup, :113 to :122)
```

| sglang element | Anchor | Omni element | Same or deviation, reason |
| --- | --- | --- | --- |
| `model_capture_mode()` around the whole capture | S3 | `_predictor_graph_mode()` around warmups and capture of each key | same semantics, talker scoped (D5) |
| `warmup()` once per process | `:1000` | none | deviation: sglang warms flashinfer autotune and allreduce workspaces, nothing the predictor uses |
| `freeze_gc` | S4 | `gc.disable()` window | deviation, D8 |
| `graph_capture()`: one stream, `wait_stream(current)` | S1 | `_predictor_capture_session` | same |
| `capture_session`: bind stream and pool once | S1 | same session | same |
| descending bucket order | `:1067` | same | same |
| static buffers sliced per shape | `capture_prepare` | per graph buffers as today (`:103` to `:109`) | same idea, kept per key because each key has its own outputs |
| two warmups on the bound stream | S2 | same | same |
| `synchronize()` and TP barrier before each warmup | `:106` to `:107` | none | deviation: the barrier aligns TP ranks, there is one rank. The device sync precedes the barrier for the same reason. Ordering is by `wait_stream` at session entry and by `torch.cuda.graph.__enter__` (P1) |
| `torch.cuda.graph(graph, pool, stream)` default error mode | `:128` | plus `capture_error_mode="thread_local"` | deviation, D11 |
| failure raises and aborts startup | `decode_cuda_graph_runner.py:462` to `:465` | key disabled, counters, warning with traceback, eager fallback | deviation: the predictor has an eager fallback and the vocoder precedent (F10) degrades the same way. A startup failure is still visible in the log |
| replay returns static outputs, runner slices | S2 | `replay` copies inputs, zeroes padding, replays, slices (`:167`) | same, padding zeroed because padded codes index an embedding |

### 6.2 The graph mode flag

`self._predictor_graph_mode_active: bool`, `False` at construction. Set by
the context manager `_predictor_graph_mode()` for the two warmups and the
capture pass. Read at the three sites of F3 in place of
`torch.cuda.is_current_stream_capturing()`, and in
`_code_predictor_forward_incremental` in place of the `for_capture`
parameter, which is removed. Eager execution (graphs disabled, no bucket,
capacity fallback, non-CUDA) never sets it, so the eager sampler keeps the
ATen path with its full shape coverage as today. Replay runs no Python in
the chain, so the flag is irrelevant there.

The guard at `:1304` (`is_current_stream_capturing()` returns `None` from
`_predictor_forward_graphed`) stays. It prevents a nested capture if a
caller ever invokes the predictor while another graph is capturing.

### 6.3 Policy resolution and the startup hook

`_resolve_predictor_graph_policy()` computes and stores:

- `_predictor_graph_enabled`: env var, `disable_cuda_graph`, `tp_size == 1`
  (as today) and `self.device.type == "cuda"`.
- `_predictor_batch_invariant`: `is_batch_invariant_mode_enabled()`, read
  after `ModelRunner.initialize` (I6).

Called from the builder hook, and from `_predictor_forward_graphed` when
`_predictor_graph_enabled` is still `None` (talkers built without the
builder).

```text
Qwen3TtsEngineBuilder.setup_model_resources(model, server_args, generation_cuda_graph_enabled)   NEW override
  if not generation_cuda_graph_enabled: return
  defaults = self.wrapper._merge_generate_kwargs()          F7, same merge the request builder uses
  signature = model.predictor_graph_signature_for_sampling(
      do_sample=defaults["subtalker_dosample"], top_k=defaults["subtalker_top_k"], top_p=defaults["subtalker_top_p"])
  model.capture_predictor_graphs(signature)                  returns after the ladder, logs one line (R6)
```

`predictor_graph_signature_for_sampling` is the pure function extracted
from `prepare_decode_buffers` lines `:1056` to `:1082` (the top_k
quantization and the flags), applied to one row's values. It becomes the
single owner of the signature rule for both the batch path and the
startup path.

The startup capture runs on the factory thread, the lazy captures on the
scheduler thread. The cuBLAS handle is per thread (P4). On both threads the
first use of every branch now happens in warmup 1, the fused addmm included,
so no first use lands inside the capture on either thread.

With cuDNN attention on (D13) the warmups of the startup capture also build
cuDNN's plans for every (bucket, key length) of the default signature, so
the serving steps of the default path never pay a plan build, and the
captured graph keeps the fused cuDNN kernel at replay. The cuDNN plan cache
is thread local (doc 20 section 10), so the lazy capture of a non default
signature on the scheduler thread still builds its plans, about 350 ms per
new key (run 3). That is today's cost, bounded by the key cache, not a
regression.

`capture_predictor_graphs` returns without work when the resolved policy
is off. It does not raise on a key failure (section 6.1 table). It records
`_predictor_graph_capture_count` as today.

### 6.4 Lazy capture

`_predictor_forward_graphed` keeps its structure: key lookup, capacity
check, `_capture_predictor_graph(key)` inside a
`_predictor_capture_session()`, failure accounting, replay. The only
behavioural change is that the capture body is the shared routine of 6.1.

### 6.5 The graph object

`_PredictorDecodeGraph.__init__(batch_size, signature, *, device,
hidden_size, hidden_dtype)` allocates the three input buffers and the
`CUDAGraph`. It has no `model` attribute. `replay` is unchanged.

### 6.6 Slice B, the attention kernel

Layout and tables, all built in `__init__` from the config:

```text
_predictor_k_cache, _predictor_v_cache: [layers, max_batch, predictor_len, kv_heads, head_dim]   slot-major
   per layer view: [max_batch * predictor_len, kv_heads, head_dim]          3-D, page size 1 (S6)
   write at step L-1: cache[layer, :batch, cache_len].copy_(k.view(batch, kv_heads, head_dim))
_predictor_kv_indptr[L]  = arange(max_batch + 1, int32) * L                 for L in 1..predictor_len
_predictor_kv_indices[L] = (arange(max_batch)[:, None] * predictor_len + arange(L)[None, :]).reshape(-1), int64
_predictor_num_kv_splits = ones(max_batch, int32)
_predictor_attn_logits   = zeros(max_batch, heads, 1, head_dim, fp32)
_predictor_attn_lse      = zeros(max_batch, heads, 1, fp32)
_predictor_attn_out      = zeros(max_batch, heads, head_dim, bf16)
```

Row b's key slots for length L are `b * predictor_len + 0..L-1`, which is
the prefix of the table for any batch at or below `max_batch`, so one
table per L serves every bucket and the batch is a grid extent only (S5).

Per attention (`_predictor_cached_self_attention`, CUDA branch):

```text
q = q.view(batch, heads, head_dim)                      no transpose
write k, v at cache_len                                 two strided copies (as today)
L = cache_len + 1
decode_attention_fwd(q, k_layer_view, v_layer_view, attn_out[:batch],
                     kv_indptr[L][:batch + 1], kv_indices[L],
                     attn_logits[:batch], attn_lse[:batch], num_kv_splits[:batch],
                     max_kv_splits=1, sm_scale=attn.scaling, k_scale=1.0, v_scale=1.0)
return attn_out[:batch].view(batch, heads * head_dim)   contiguous, feeds o_proj and the fused addmm as today
```

Contract checks against S5 and S6: `max_kv_splits == attn_logits.shape[2]`
holds (1), `q.shape[0] <= kv_indptr.shape[0] - 1` holds (batch rows of
a `max_batch + 1` table), `q.shape[0] <= attn_logits.shape[0]` holds,
`attn_logits.shape[-1] == head_dim` holds (the `// Lv` addressing of
stage 1 and 2). No zero length row exists since every row has length L.
`kv_group_num = 2` selects the grouped path with `VALID_BLOCK_H = 2` and
`head_tiles = 8`. Under batch invariant mode the single split and the per
row programs make the result independent of batch mates.

The non-CUDA branch keeps today's SDPA code over a BHSD view of the same
slot-major cache (a transpose view, no copy for the read).

The import: `decode_attention_fwd` under the optional import guard of
`predictor_kernels.py` (F13), `None` when triton or the kernel module is
absent, in which case the CUDA branch is not taken.

### 6.7 Alternatives and why not

| Alternative | Rejected because |
| --- | --- |
| Use sglang's `model_capture_mode()` as the flag | it also sets `disable_dispose_tensor` and is read by sglang model code on paths we do not run, coupling without a consumer (D5) |
| Use sglang's `freeze_gc` for the lazy captures | a full collection inside a serving step (D8). Startup could use it, one mechanism for both windows is simpler and covers the hazard |
| Capture every signature at startup | the signature set is open (top_k ladder times top_p times unbounded), only the default is known at startup (D2) |
| Drop lazy capture | one non default request would run its whole batch eager for its lifetime (D2) |
| Break the cycle with a weakref | the graph object never needs the talker after capture, passing it as an argument removes the reference entirely (D7) |
| Pin the SDPA backend with `sdpa_kernel` around the chain | process global, and the tokenizer threads share the process (doc 20 section 4) |
| Explicit masked attention in one launch (T22) | one shape per bucket and one launch, but new kernel code to own. The replay gate decides whether the two launch kernel is good enough. Deferred, not rejected |
| sglang's global `gc.freeze` at startup only, lazy path unguarded | leaves the hazard of I3 on the lazy path |

### 6.8 Contracts changed

| Boundary | Before | After |
| --- | --- | --- |
| `_PredictorDecodeGraph(model, batch_size, signature, *, hidden_size, hidden_dtype)` | captures in `__init__`, stores the model | `(batch_size, signature, *, device, hidden_size, hidden_dtype)`, holds buffers and graph only, captured by the talker |
| `_code_predictor_forward_incremental(..., for_capture=False)` | parameter selects the graph safe sampler | parameter removed, the flag selects it |
| `Qwen3TTSTalker.capture_predictor_graphs(signature)` | none | new, startup entry, no host reads of device values |
| `Qwen3TTSTalker.predictor_graph_signature_for_sampling(do_sample, top_k, top_p)` | rule inline in `prepare_decode_buffers` | new pure function, used by both |
| `Qwen3TtsEngineBuilder.setup_model_resources` | inherited no-op | override of 6.3 |
| Predictor cache layout (Slice B) | `[layers, max_batch, kv_heads, predictor_len, head_dim]` | `[layers, max_batch, predictor_len, kv_heads, head_dim]` |
| Attention numerics (Slice B) | PyTorch SDPA, flash on the fixed branch | sglang decode attention on CUDA, SDPA elsewhere |
| Log lines | one info per lazy capture (`:1354`) | unchanged, plus one info at the end of the startup capture with key count and wall time |

## 7. Execution plan

Both slices branch from upstream main (`fa1ea43dc`), whose predictor files
are identical to the fix base (revision table). The cuDNN repair branch is
not a base and is not merged (D13).

### 7.1 Slice A, `perf/qwen3-tts-predictor-capture`

Objective: K1 to K6 closed, predictor graphs bit identical to today's graph
path outside deterministic mode.

Files and symbols:

- `sglang_omni/models/qwen3_tts/sglang_model.py`
  - `_PredictorDecodeGraph`: constructor of 6.5, delete `_capture`, keep
    `replay`.
  - `Qwen3TTSTalker.__init__`: `_predictor_graph_mode_active = False`,
    `_predictor_capture_stream = None`, `_predictor_batch_invariant = False`.
  - new `_predictor_graph_mode()` context manager.
  - new `_predictor_capture_session()` context manager (6.1).
  - new `_capture_predictor_graph(key)` (6.1), including the reset on
    failure.
  - new `capture_predictor_graphs(signature)` (6.3).
  - new `predictor_graph_signature_for_sampling(...)` and its use in
    `prepare_decode_buffers`.
  - `_resolve_predictor_graph_enabled` becomes `_resolve_predictor_graph_policy`
    (6.3).
  - `_predictor_forward_graphed`: policy resolution when `None`, capture
    through the session and the shared routine.
  - `_code_predictor_forward_incremental`: drop `for_capture`, read the
    flag for the fused embedding gather and the sampler choice.
  - `_sample_subtalker_token`, `_sample_subtalker_token_seeded`: read the
    flag.
  - `_predictor_o_proj_add_residual`: no longer static, reads the flag and
    `_predictor_batch_invariant`.
- `sglang_omni/models/qwen3_tts/engine_builder.py`: `setup_model_resources`
  override of 6.3.
- `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`: fake gains
  the new attributes, the fake's `code_predictor` keeps its shape, tests
  T-A1 to T-A9 added, tests reading `for_capture` updated.
- `tests/unit_test/qwen3_tts/test_pipeline.py`: T-A10.

Exit gate: all of section 8.1 Slice A rows pass on the box, ledger run 6
shows no capture line inside serving steps for the default signature at
c1 and c16, the deterministic mode run of 8.2 captures every key, and the
A/B of 8.2 is flat.

### 7.2 Slice B, `perf/qwen3-tts-predictor-attention`

Objective: K7 closed on CUDA.

Files and symbols:

- `sglang_omni/models/qwen3_tts/sglang_model.py`
  - guarded import of `decode_attention_fwd` next to the other optional
    kernel imports.
  - `__init__`: layout and tables of 6.6, sized from
    `cp_attn.num_heads`, `cp_attn.num_kv_heads`, `cp_attn.head_dim`,
    `predictor_len`, `max_batch_size`.
  - `_predictor_cached_self_attention`: CUDA branch of 6.6, SDPA branch
    over the transposed view.
- `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`: fake geometry
  moves to `NUM_HEADS = 16`, `NUM_KV_HEADS = 8`, `HEAD_DIM = 128` with
  `HIDDEN` unchanged, tests T-B1 to T-B3.

Exit gate: section 8.1 Slice B rows pass, the replay gate of 8.3 selects
the kernel, the A/B of 8.2 for Slice B is within the quality gates and
not slower than Slice A at c1 and c16. If the gate selects SDPA, Slice B
ships the layout and tables only if they carry their own measured gain,
otherwise Slice B is closed with the measurement recorded and T22 opens.

## 8. Proof

### 8.1 Unit tests (new, `pytest tests/unit_test/qwen3_tts -q` on the box)

| Id | Contract | Test | Oracle |
| --- | --- | --- | --- |
| T-A1 | graph equals eager bits after the refactor | existing bit identity tests unchanged | `torch.equal` |
| T-A2 | warmups run the capture branches | spy on `sample_from_logits_with_seed_top_k_top_p` and `gather_codec_embedding_and_add` during one capture at bucket 2 | fused sampler called `3 * (NUM_CODE_GROUPS - 1)` times, gather called `3 * (NUM_CODE_GROUPS - 1)` times, zero during the following eager call |
| T-A3 | one capture stream per talker | spy subclass of `torch.cuda.graph` recording `stream` for two keys, plus the warmup stream identity recorded by a spy on the model's `_code_predictor_forward_incremental` reading `torch.cuda.current_stream()` | both captures receive the same stream object, the six warmup calls ran on it, it is not the default stream |
| T-A4 | collector disabled during capture, restored after | spy inside `run_once` recording `gc.isenabled()` | `False` for all three passes, `True` after the call, also `True` after a capture that raises |
| T-A5 | startup capture builds the ladder for one signature in descending order | `capture_predictor_graphs(("sampled", 8, False, False))` on the fake | keys equal `{(b, "sampled", 8, False, False) for b in BUCKETS}`, capture order recorded by the stream spy is descending, a following forward at batch 3 replays without a new key |
| T-A6 | signature rule shared | `predictor_graph_signature_for_sampling` versus `prepare_decode_buffers` on uniform batches for the ladder points, top_p in `{1.0, 0.9}`, top_k in `{0, 3, 50, 2048}` | equal tuples |
| T-A7 | no talker reference from the graph object | `gc.get_referents` closure of a captured `_PredictorDecodeGraph` | the talker is not reachable, and `del talker` followed by `gc.collect()` returns zero collected objects that are `CUDAGraph` instances |
| T-A8 | batch invariant mode uses the eager GEMM on the graph path | monkeypatch `is_batch_invariant_mode_enabled` to `True` before policy resolution, spy `torch.addmm` | zero `addmm` calls in warmups and capture, graph equals eager bits |
| T-A9 | policy gate on device and flags | fake on `cpu` with graphs enabled | `capture_predictor_graphs` captures nothing, `_predictor_graph_enabled` resolves `False` |
| T-A10 | builder hook derives the default signature from the merged defaults and calls the talker | stub `wrapper._merge_generate_kwargs` returning `subtalker_top_k=7`, `subtalker_top_p=0.5`, `subtalker_dosample=True`, stub `model.capture_predictor_graphs` recording its argument | called once with `("sampled", 8, True, False)`, not called when `generation_cuda_graph_enabled` is `False` |
| T-B1 | kernel path equals a fp32 reference | `_predictor_cached_self_attention` on random q, k, v for `L in {1, 2, 16, 17}`, batches `{1, 3, 16}`, head dims `{128, 64}` versus the reference of S7 | `allclose(atol=1e-2, rtol=1e-2)` on fp32 casts |
| T-B2 | index tables match the slot layout | tables versus a Python enumeration of `(row, position)` slots | equal |
| T-B3 | graph equals eager bits with the kernel in both | existing bit identity tests at the new fake geometry | `torch.equal` |

Tests that would still pass if the implementation regressed are not
listed. Each of T-A2, T-A3, T-A4, T-A7, T-A8 fails on today's code.

### 8.2 Box runs

Every run follows doc 15 (one server, cookbook launch, the ledger commit
cherry-picked for ledger runs).

| Run | Arm | Reads |
| --- | --- | --- |
| run 6, ledger | Slice A | startup log has the capture line with 6 keys, no `Captured Qwen3-TTS predictor CUDA graph` line inside a serving step at c1 or c16 for default requests, first step latency at c1 and c16 versus run 5 |
| run 6d, ledger | Slice A with `--enable-deterministic-inference true` (the flag of #1936) | every key captures, no `CUBLAS_STATUS_NOT_INITIALIZED`, one request at c1 completes |
| A/B A | upstream main versus Slice A, c1 and c16, full corpus, generate-only then transcribe-only and similarity-only, arms alternated, a fixed `--seed` (T39) | WER and similarity flat within run to run noise, paired per sample latency not worse (doc 22 section 4 method) |
| A/B B | Slice A versus Slice B, same protocol | quality within the gates of doc 15, speed not worse, plus the gate of 8.3 |

### 8.3 The replay time gate (Slice B)

With the ledger on both arms at c1 and c16, compare the decode step's GPU
span minus the backbone forward (the predictor replay, sampling and
collect, doc 22 section 2) and the per step predictor time from a
`/start_profile` window of 20 steps (doc 15). The Slice A arm replays the
fused cuDNN kernel (D13). Accept the kernel when the c16 per step time is
not worse than Slice A beyond the run to run spread of two repeats.
Otherwise keep SDPA on cuDNN and close Slice B as in 7.2.

## 9. Rollout, observability, rollback

- Rollout: Slice A first. It adds startup time of roughly the ladder size
  times one capture (expected well under two seconds after the cuDNN
  repair, measured by run 6) and removes the same work from serving.
- Kill switches unchanged: `SGLANG_OMNI_QTTS_PREDICTOR_GRAPH=0`,
  `disable_cuda_graph`. Both now also skip the startup capture.
- Observability: the startup line with key count and wall time, the
  existing per lazy capture line, the existing warning with traceback on
  failure. The ledger reads the rest.
- Rollback: revert the branch. No persisted state, no protocol change, no
  config change. A process on the old code and a process on the new code
  serve the same API.

## 10. Tasks and the measured gate

- G1 The replay time gate of 8.3 decides Slice B's attention path.
- T31 Write Slice A after this plan is confirmed.
- T32 Write Slice B after run 6 and A/B A.
- T33 Measure the cold Triton cache cost of the first fused kernel launch
  under today's code (a capture with `~/.triton` cleared) to record what
  Slice A moves out of the capture pass.
- T34 Read PyTorch's flash split-kv heuristic for `seqlen_q = 1` to state
  the launch count of the SDPA path inside the graph. Only informs the
  gate's interpretation.
- T35 Confirm Triton 3.7.1's `tl.dot` minimum tile on the box with a one
  line script. Informational, the tests use the production geometry.
- T36 Run the Slice B unit tests on a non sm90 CUDA box when one is
  available.
- T37 Open the tracker note on #1936: fault 1 is closed by Slice A, fault
  2 stays open.
- T38 If a lazy vocoder capture lands (PR #1855), serialize it against the
  predictor's lazy capture in the shared process (P7).
- T39 Run every A/B with a fixed `--seed` so both arms sample the same
  trajectories where their numerics agree and a runaway can be replayed.
  Add to doc 15.
- T40 The speech tokenizer and the codec decoder build a cuDNN plan per new
  sequence length, 38 ms and 34 ms at the p50 of the run 3 window (doc 22
  section 3). A startup sweep over a length ladder for both is decided
  after Slice A lands.
- T22, T29, T30, A4 remain as recorded in docs 19 and 20.
