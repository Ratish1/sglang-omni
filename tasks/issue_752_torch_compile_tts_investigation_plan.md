# Issue #752: TTS `torch.compile` Investigation Plan

## Objective

Build a profiling-backed answer for whether `torch.compile` should be used in the MOSS/TTS autoregressive path, and where it can produce real end-to-end gains without duplicating the open MOSS compile work.

The expected output is not "add `@torch.compile` where missing." In SGLang, normal LLM decode compile is already tied to CUDA graph capture, batch-size buckets, and replay semantics. For MOSS Local, the question splits into two separate support surfaces:

- the Qwen3 backbone, which can already use SGLang's native `torch.compile` + CUDA graph capture path through `server_args.enable_torch_compile`;
- the MOSS-specific frame-local decoder, which lives outside SGLang's normal decode graph runner and needs its own opt-in compile-before-frame-capture path.

The branch should support both experiments without changing defaults, then use H100 profiles to decide whether either path is worth making a recommended configuration.

## Branch Strategy

Work happens on:

```text
perf/issue-752-moss-tts-compile-investigation
```

This branch stays based on current `upstream/main`. We will not check out open PR branches as the working base. Open PRs are comparison evidence only:

- `#751`: existing MOSS Local backbone compile experiment;
- `#755`: existing MOSS Local quantizer compile experiment;
- `#736` / `#757` / `#759`: frame graph and state-pool direction.

If we need behavior from an open PR, inspect or manually port the minimal idea after profiling proves it is the right direction. Do not build this branch on top of another PR branch.

## Current Evidence

### GitHub Issue And PR State

- Issue `#752` asks for a profiling-backed conclusion and predictive go/no-go rule for TTS AR `torch.compile`.
- PR `#751` is the direct MOSS-TTS-Local AR backbone experiment. It reports:
  - CUDA graph alone improves RTF by roughly 35-41%.
  - `torch.compile` on top of CUDA graph is within noise: about `+1.4%` slower at concurrency 8 and `-1.2%` faster at concurrency 16.
  - compile-only is not a fair comparison because disabling backbone CUDA graph also removes fixed batch padding and causes shape churn/recompiles.
  - recommendation: do not enable MOSS Local AR backbone compile by default.
- Higgs PR `#579` reached the same general conclusion: eager plus CUDA graph is already as good as or better than compile plus CUDA graph in the measured path.
- S2-Pro PR `#625` is the useful exception: compiling a small repeated codebook loop improved a local metric, but end-to-end latency/QPS/RTF was nearly flat.
- PR `#755` already covers MOSS Local quantizer compile behind an opt-in flag. Do not duplicate that work.
- Issue `#736` and issue/PR work around `#757`/`#759` target the likely higher-value path: making MOSS Local frame decode graph use pool-resident tensors and avoid per-frame Python/scalar glue.

### Architecture From `docs/developer_reference`

The serving path is:

```text
HTTP API
  -> Client
  -> Coordinator
  -> Stage
  -> Scheduler
  -> SGLang ModelRunner
  -> model.forward / custom model runner hooks
  -> output processing
```

For TTS, the documented logical stages are:

```text
preprocessing/reference encoding
  -> tts_engine
  -> vocoder
```

The important implementation boundary is that `Stage` owns IO and lifecycle, while the scheduler and model runner own batching, forward execution, CUDA graph replay, KV/cache handling, and token/frame emission. Request builders must keep device tensors on device in hot paths and use typed request dataclasses.

The profiler docs already provide the right first pass:

- request-level JSONL events for queueing/stage timing;
- `/start_profile` and `/stop_profile` for torch profiler traces;
- shape/memory/stack options for short windows when diagnosing graph breaks or unexpected allocations.

### Local Codebase Findings

MOSS Local code:

- `sglang_omni/models/moss_tts_local/stages.py`
  - defaults `enable_torch_compile=False`;
  - enables SGLang decode CUDA graph buckets `[1, 2, 4, 8, 16]`;
  - initializes SGLang device graphs first, then MOSS Local frame decode graphs.
- `sglang_omni/models/moss_tts_local/sglang_model.py`
  - `forward()` runs the Qwen3 backbone and returns hidden states;
  - `_decode_frame_graphable()` contains the branchless frame-local decode loop;
  - `init_frame_decode_graphs()` captures per-bucket CUDA graphs for local frame decode;
  - `decode_frame_graphed()` pads to bucket size, copies into static inputs, replays the graph, and returns sliced static outputs.
- `sglang_omni/models/moss_tts_local/model_runner.py`
  - `before_decode()` writes feedback embeddings into `_decode_input_embedding`;
  - `_collect_frame()` still does per-request Python work each frame: row acquisition, sampling parameter setup, `generation_steps` tensor construction, repetition penalty branching, output clone/copy, radix token update, and journal construction;
  - no-repetition-penalty rows can use `decode_frame_graphed()`, while repetition penalty falls back to eager decode.
- `sglang_omni/models/moss_tts_local/state_pool.py`
  - owns persistent row-indexed tensors for feedback embeddings and sampling params;
  - currently does not own active-row or generation-step tensors, which is exactly the kind of state needed to remove Python scalar glue from the frame path.

Post-branch audit of the exact hot path:

```text
OmniScheduler._run_batch
  -> ModelRunner.execute
  -> ModelRunner._prepare_and_forward
  -> MossTTSLocalModelRunner.before_decode
       copies pool.feedback_embeds[row_t] into _decode_input_embedding.weight
       rewrites forward_batch.input_ids to 0..batch-1
  -> tp_worker.forward_batch_generation
  -> MossTTSLocalSGLangModel.forward
       SGLang CUDA graph replay covers the Qwen3 backbone decode
       returns hidden states with dummy logits
  -> MossTTSLocalModelRunner._collect_frame
       Python gathers rows/data/generation_steps
       uses MOSS Local frame CUDA graph when repetition penalty is off
       computes generated-row radix ids on GPU
       writes next feedback embedding to state_pool
       attaches MossTTSLocalDecodeJournal
  -> ModelRunner._finalize
       output processor materializes RequestOutput
       journal appends emitted rows
       generation_steps increments unless skipped for chunked prefill
```

The first source of truth for generated frame content is the per-step `MossTTSLocalDecodeJournal`; the first source of truth for next-step decode input is `MossTTSLocalDecodeStatePool.feedback_embeds`; the first source of truth for sampling position is still `data.generation_steps` on the scheduler request data.

That ownership split is semantically valid today, but it explains why a wider frame graph cannot simply read all launch metadata from the pool yet: generation steps and active row slots are still host-side/request-list derived.

Latest remote B/D2 evidence from `/data/moss_local_issue752_bd2_scoped_20260612_110825`:

- B, n50: `3.160` QPS, `0.6232` RTF mean, `2.465s` latency mean, collect-frame `11.606ms` avg, frame-decode `0.895ms` avg.
- D2, n50: `3.609` QPS, `0.5246` RTF mean, `2.092s` latency mean, collect-frame `7.383ms` avg, frame-decode `0.563ms` avg.
- D2 vs B: `+14.2%` QPS, `+15.8%` better RTF, `+15.1%` better latency.
- Both cases used frame CUDA graph for every frame, with no fallback and no repetition-penalty rows.
- n8 torch trace agrees directionally: `decode_frame_graphed` drops from `1.223ms` avg to `0.583ms`, `cudaGraphLaunch` total drops from `270.480ms` to `158.442ms`, and `cudaStreamSynchronize` total drops from `1433.220ms` to `810.828ms`.
- Startup cost increased from roughly `30s` to `60s`; earlier cold-ish D2 startup was much worse, so cold/warm compile cache behavior still needs a controlled measurement.
- Chrome trace still did not preserve `moss_tts_local.*` `record_function` labels, even though the helper was entered. The branch now has opt-in JSONL fine-scope events via `SGLANG_MOSS_TTS_LOCAL_FINE_FRAME_EVENTS=1` so the next scoped run can get scope-level evidence independent of Chrome trace label export.

Working interpretation: frame-local compile is a real candidate because it compiles the small repeated local decoder before explicit frame CUDA graph capture, matching SGLang's compile-with-cudagraph philosophy but avoiding the graph-break-heavy Qwen backbone surface. It is not yet a production recommendation until ABAB repeats, quality parity, cold/warm startup cost, and fine-scope attribution are confirmed.

Latest remote fine-scope B/D2 evidence from `/data/moss_local_issue752_bd2_fine_20260612_141125` at commit `bb6d71d5`:

- B, n50: `2.870` QPS, `0.6583` RTF mean, `2.636s` latency mean.
- D2, n50: `3.202` QPS, `0.5960` RTF mean, `2.383s` latency mean.
- D2 vs B: `+11.6%` QPS, `-9.5%` RTF mean, `-9.6%` latency mean.
- Every frame used `frame_decode_path=cuda_graph`; no fallback and no repetition-penalty rows.
- Frame event avg drops from `1.224ms` to `0.849ms`; collect-frame avg drops from `12.525ms` to `8.474ms`.
- Torch trace n8 agrees on the compute/replay mechanism: `decode_frame_graphed` avg drops `1.244ms -> 0.585ms`, `cudaGraphLaunch` avg drops `0.764ms -> 0.461ms`, and `cudaStreamSynchronize` total drops `928.90ms -> 523.75ms`.
- Fine JSONL scopes show the largest apparent delta in `feedback_write` (`3.835ms -> 0.436ms`), but this should be treated as a boundary/backlog symptom, not proof that the Python assignment itself became faster. Scope timestamps wrap asynchronous CUDA work plus synchronous JSONL event emission, so downstream scopes can absorb upstream GPU backlog. The trace-level graph replay and stream-sync deltas are the stronger attribution.
- Startup cost remains non-trivial: total ready `35.04s -> 55.06s`, with D2 frame compile/capture about `22.23s`.

Updated read: D2 is now the strongest `torch.compile` surface for issue #752, but the remaining optimization stack is below it: pool-resident launch state, graph-boundary widening, and async-safe state transitions.

Adjacent upstream/open work that affects next steps:

- PR `#751` is a negative result for Qwen/backbone compile on top of CUDA graph. Do not duplicate that path unless maintainers request another confirmation.
- PR `#755` compiles the preprocessing codec/quantizer. It is valid, but it is a separate preprocessing surface and should not be mixed into D2 attribution.
- Issue `#757` and PR `#759` move active rows, `generation_steps`, and repetition-penalty history into the MOSS Local state pool. This directly targets our remaining `param_gather` / `sampling_state` / launch-prep layer. Treat it as the next baseline delta to compare against D2, not as a replacement for D2.
- Issue `#736` is the larger target: make the frame path native over the row-indexed state pool and remove per-frame host/copy glue around the already captured frame graph.

Latest B/D2/P/P+D2 matrix from `/data/moss_local_issue752_p_vs_d2_20260612_143506`:

- B/D2 ran on this branch at `2cd6b905`; P/P+D2 ran on a local PR `#759`-equivalent integration at `ab012cb8`.
- No D1/D3; no PR `#755` preprocessing/quantizer compile.
- End-to-end n50:
  - B: `3.186` QPS, `0.6139` RTF mean, `2.440s` latency.
  - D2: `3.462` QPS, `0.5536` RTF mean, `2.188s` latency.
  - P: `2.894` QPS, `0.6703` RTF mean, `2.671s` latency.
  - P+D2: `2.526` QPS, `0.7908` RTF mean, `3.061s` latency.
- D2 is again the only end-to-end win: about `+8.7%` QPS and `-9.8%` RTF vs B.
- All cases used frame CUDA graph for every frame, with no fallback and no repetition-penalty rows.
- P improves scoped launch-state work (`param_gather`, `sampling_state`) but regresses end-to-end. P+D2 improves scoped `_collect_frame` and graph metrics but regresses end-to-end even more.
- Trace-level P+D2 counters look locally attractive (`_collect_frame` avg `9.052ms`, `cudaStreamSynchronize` total `168.4ms`) while n50 request throughput is worst. This means the regression is outside the current MOSS fine scopes or the scoped n8 trace window is not representative of n50.

Decision from this matrix:

- Keep D2 scoped to `_decode_frame_graphable()` as the current supported compile candidate.
- Do not recommend P or P+D2 from this integration.
- Do not conclude that pool-resident state is conceptually bad. PR `#759` moves work out of `_collect_frame()` into `before_decode()` / finalize hooks, so scoped `_collect_frame` wins can hide total scheduler-loop cost. The next analysis must include stage breakdown, scheduler loop time, `before_decode`, `_finalize`/generation-step commit, and result/vocoder handoff.
- The P regression should be root-caused before any attempt to widen the MOSS frame graph boundary.

MOSS Delay code:

- `sglang_omni/models/moss_tts/stages.py` also defaults `enable_torch_compile=False`.
- `sglang_omni/models/moss_tts/model_runner.py` samples delay-pattern rows/channels outside the backbone forward, with significant request/control-flow handling in Python.
- `sglang_omni/models/moss_tts/sglang_model.py` keeps Qwen3 backbone execution separate from per-channel logits/sampling.

Other TTS compile precedents:

- `sglang_omni/models/fishaudio_s2_pro/stages.py` compiles individual `forward_kvcached` methods for a codebook decoder loop, then disables generic engine-level compile before graph init.
- `sglang_omni/models/qwen3_tts/stages.py` compiles decoder layers directly and disables generic engine-level compile afterward.

SGLang compile implementation, from the local sibling checkout and vendored patterns:

- Normal LLM decode compile is not decorator-based. It is installed during CUDA graph capture through SGLang's decode graph runner and `patch_model(...)`.
- Decode graph replay pads requests to captured batch-size buckets. This is central to avoiding shape churn.
- Multimodal generation compiles large diffusion components (`transformer`, `unet`, denoising stage modules) with component-level `.compile()`/`torch.compile(...)`. That pattern is not directly transferable to MOSS AR because the AR hot path is split between SGLang backbone graph replay, frame-local graph replay, and Python orchestration.

## First-Principles Interpretation

`torch.compile` can help when it sees a stable, fusible compute region with enough repeated work to amortize compile overhead and without frequent graph breaks or recompiles.

It is unlikely to help when:

- CUDA graph replay already removed launch overhead for the same region;
- the region is dominated by cuBLAS/attention kernels that Inductor cannot improve;
- dynamic batch/sequence shapes trigger recompiles;
- the remaining bottleneck is Python orchestration, host syncs, tensor allocation/copy glue, or cross-stage queueing.

For MOSS Local, the Qwen3 backbone is already under SGLang decode CUDA graph. The frame-local decode loop is also CUDA-graph captured, but its current replay boundary still requires Python and copy glue in `_collect_frame()`. That points to a graph-boundary/state-layout problem before it points to a missing `torch.compile`.

## Investigation Plan

### Phase 1: Reproduce The SGLang-Realistic Baseline On Remote H100

Goal: establish the cost breakdown before adding or changing compile code.

Primary comparison:

- B: CUDA graph only;
- D1: SGLang backbone compile plus CUDA graph;
- D2: MOSS frame-local compile plus frame CUDA graph;
- D3: both backbone and frame-local compile plus CUDA graph.

These are the comparisons that matter most because SGLang's normal decode compile path is designed to run with CUDA graph capture and fixed batch-size buckets. Treat compile-only as a diagnostic, not as a production-relevant strategy.

Optional diagnostic comparisons, only if logs or reviewer questions require them:

- A: eager backbone, no CUDA graph, no compile;
- C: compile only;

Run the primary comparison on `main`, then on the `#751` branch if available.

Use fixed model/workload:

- `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`;
- SeedTTS English full set or the exact issue `#751` benchmark set;
- concurrency `8` as the first gate;
- concurrency `16` only if c8 is noisy, contradicts `#751`, or maintainers need confirmation;
- at least three repetitions per setting;
- record RTF mean/median/p95/p99, QPS, audio throughput, first-token/first-frame timing, and request queueing.

Use event profiling first, then short torch profiler windows:

```bash
SGLANG_TORCH_PROFILER_RECORD_SHAPES=1 \
SGLANG_TORCH_PROFILER_WITH_STACK=1 \
sgl-omni serve --model-path OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --port 8021
```

For compile runs, add:

```bash
TORCH_LOGS=recompiles,graph_breaks,dynamo
SGLANG_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
```

Backbone compile is enabled through the existing stage server-args surface:

```bash
sgl-omni serve ... --talker_torch_compile on --talker_torch_compile_max_bs 8
```

Frame-local compile is enabled through the MOSS Local TTS engine factory args:

```yaml
stages:
  - name: tts_engine
    factory_args:
      frame_decode_torch_compile: true
      frame_decode_torch_compile_mode: max-autotune-no-cudagraphs
```

Decision gate:

- If D1 vs B remains within run-to-run noise, do not recommend MOSS Local backbone compile.
- If D2 vs B remains within run-to-run noise, keep frame-local compile as a diagnostic-only surface and prioritize state-pool/frame-boundary work.
- If D1, D2, or D3 is reproducibly better by more than 5% end-to-end with no quality regression, inspect logs and generated regions before proposing a default or cookbook recommendation.
- Local subpath wins alone are not enough. The acceptance bar is end-to-end RTF/QPS/latency.
- Use the SGLang `llm-torch-profiler-analysis` scripts as the trace triage layer after capture, but drive them with MOSS TTS benchmark traces rather than the default synthetic LLM prefill/decode workload.

### Phase 2: Attribute The Hot Path Precisely

Goal: split request time into stage, scheduler, backbone, frame decode, vocoder, and orchestration costs.

Use the existing profiler events and add minimal scoped events only if current traces cannot answer:

- preprocessing/reference encode duration;
- scheduler queue time;
- prefill duration;
- decode/frame loop duration;
- `_collect_frame()` wall time;
- frame graph replay duration;
- vocoder duration;
- first streamed chunk latency.

In torch profiler traces, explicitly inspect:

- CUDA graph replay blocks versus eager kernels;
- tensor allocation/copy operations around `decode_frame_graphed()`;
- CPU self time in `_collect_frame()`;
- host-device synchronization;
- shape churn/recompile events during compile runs;
- whether repetition penalty disables frame graph replay for meaningful traffic.

Decision gate:

- If `_collect_frame()`/frame glue is material, prioritize state-pool/native graph work.
- If codec/quantizer dominates at higher concurrency, coordinate with PR `#755`.
- If queueing/stage transfer dominates, compile is the wrong lever.

### Phase 3: Compare D2 Against Pool-State Cleanup, Not Against More Backbone Compile

Goal: isolate the next layer after frame-local compile.

Do not fold preprocessing compile (#755) into this matrix. It is a valid
end-to-end serving improvement but answers a different question.

Primary matrix once #759-style pool-state changes are available on the branch
or on an upstream baseline:

- B: CUDA graph only.
- D2: frame-local compile before frame CUDA graph capture.
- P: pool-resident launch-state cleanup only (`generation_steps`, active rows,
  and repetition history in pool tensors).
- P+D2: pool-state cleanup plus frame-local compile.

Readout:

- If P and D2 are additive, support both: D2 reduces the captured frame graph's
  replay cost, while P reduces launch prep and state marshalling.
- If P absorbs most of D2's end-to-end gain, keep D2 as optional and recommend
  P first because it has no compile startup cost.
- If P+D2 worsens startup or introduces graph breaks, keep D2 scoped exactly to
  `_decode_frame_graphable()` and do not widen compile until graph-boundary
  ownership is proven.

The next instrumentation should report both request-event fine scopes and torch
trace runtime counters. Fine JSONL scopes are useful for relative boundaries,
but any large late-scope delta must be cross-checked against
`cudaStreamSynchronize`, `cudaGraphLaunch`, and function-level trace spans before
calling it the root cause.

Because P/P+D2 regressed end-to-end despite better `_collect_frame` scopes, the
next diagnostic run must add or extract timings outside `_collect_frame`:

- `before_decode` / `_write_decode_input_embedding`, especially row preparation
  moved there by P.
- `post_process_outputs` and `_finalize`, especially `on_generation_steps_advanced`
  / pool generation-step commits.
- scheduler-loop iteration time around `run_batch` and `process_batch_result`.
- stage breakdown for `tts_engine/stage_input_received -> stage_complete`,
  `tts_engine/scheduler_prefill_start -> scheduler_first_emit`, and handoff to
  vocoder.
- queueing/hop time into vocoder, because a slower tts-engine drain can reduce
  end-to-end QPS even when frame scopes shrink.

If these show that P merely moves work out of measured collect scopes, then P
should be either rejected as currently implemented or reduced to only the pieces
that are proven additive with D2.

### Phase 4: Do Not Duplicate Existing Compile PRs

Goal: align with open work instead of creating a parallel implementation.

Actions:

- Treat PR `#751` as the current MOSS Local backbone compile experiment.
- Treat PR `#755` as the current quantizer compile experiment.
- Track issue/PR `#757`/`#759` for pool tensorization and vectorized frame launch.
- If any of these merge before implementation, rebase the plan on the merged code and rerun Phase 1/2 baselines.

Decision gate:

- If `#751` closes as negative, document it in the issue and avoid adding another backbone compile path.
- If `#755` shows reproducible end-to-end gains, validate and help finish that PR rather than reimplementing it.

### Phase 3.5: Add The Missing MOSS Frame Compile Experiment Surface

Goal: make issue `#752` directly testable in this branch without changing production defaults.

Implementation:

- keep `create_sglang_tts_engine_executor(..., server_args_overrides=...)` as the backbone compile control; this is already consumed by `--talker_torch_compile`;
- add `frame_decode_torch_compile=False` and `frame_decode_torch_compile_mode=None` factory args to the MOSS Local TTS engine;
- compile `MossTTSLocalSGLangModel._decode_frame_graphable` during `init_frame_decode_graphs(...)` before CUDA graph warmup/capture;
- use `SGLANG_TORCH_COMPILE_MODE` or the explicit factory arg mode, defaulting to `max-autotune-no-cudagraphs`;
- do not compile `decode_frame_graphed()`, because that wrapper owns copy-in and `graph.replay()` rather than the compute region;
- fail loudly on opt-in compile errors rather than silently falling back, so the experiment does not produce misleading "compile enabled" results.

Status: implemented on `perf/issue-752-moss-tts-compile-investigation`.

### Phase 4: Target The Likely High-Value MOSS Local Change

Goal: reduce frame-loop overhead by moving launch-critical state into persistent GPU tensors and widening CUDA graph capture boundaries.

Candidate implementation slices on our branch:

0. Add profiler observability around the MOSS Local frame path, gated by the existing request profiler:
   - batch-size and graph/eager fallback metadata for `_collect_frame()`;
   - frame graph replay interval;
   - repetition-penalty fallback counts;
   - no tensor metadata, no `.item()`, no `.cpu()`, and no extra sync in the hot path.
   - Status: implemented on `perf/issue-752-moss-tts-compile-investigation` with `moss_tts_local_collect_frame_*` and `moss_tts_local_frame_decode_*` request-profiler events, profiler-view interval aggregation, docs, and CPU-only coverage.

0.5. Add torch-profiler ranges inside `_collect_frame()`:
   - Purpose: split the measured ~12 ms collect-frame boundary into pool row/param setup, sampling state, graph replay, static-output clone, row build, radix hash, feedback write, and journal creation.
   - Why before implementation: the main question is still whether compile+CUDA graph has any remaining target. SGLang already compiles the Qwen backbone inside decode CUDA graph capture; these ranges show whether the remaining cost is fusible compute, graph-boundary copy/glue, or Python/state orchestration.
   - Status: implemented as `moss_tts_local.*` `torch.profiler.record_function` ranges gated by `torch.autograd._profiler_enabled()`.

1. Extend `MossTTSLocalStatePool` with persistent GPU tensors for:
   - active row ids or row slots;
   - generation steps;
   - optional per-row status needed by graph replay.
2. Update request lifecycle so generation steps are committed through the same journal/finalization contract already used for emitted rows.
3. Replace per-frame Python list-to-tensor construction in `_collect_frame()` with tensor views/gathers from the pool.
4. Add a pool-native frame graph path, for example `decode_frame_pool_graphed(row_t, hidden_states)`, that reads sampling params and generation steps from pool tensors and writes feedback/code outputs through stable row slots.
5. Keep repetition penalty on the eager fallback until it can be represented without dynamic Python-side histories.
6. Preserve deterministic sampling semantics: same seed, same row, same generation step must produce the same output as the current graph/eager path.

This should be coordinated with `#736`, `#757`, and `#759`. If `#759` already implements part of this, review and extend that direction instead of creating an overlapping branch.

Acceptance bar:

- measurable end-to-end improvement at concurrency `8`/`16`;
- no quality regression in WER/SIM/CER checks used by the existing benchmark flow;
- frame graph parity against current implementation on fixed-seed GPU tests;
- no increase in memory fragmentation or graph recapture frequency.

### Phase 5: Re-Evaluate MOSS Delay Separately

Goal: avoid overgeneralizing MOSS Local results to the delay-pattern model.

MOSS Delay has different sampling structure and per-channel decode behavior. For it:

- first profile whether backbone, channel logits/sampling, or Python delay-pattern management dominates;
- check whether CUDA graph already captures the expensive backbone region;
- only consider compile on a small stable inner loop if it is a material fraction of end-to-end time;
- use the S2-Pro pattern as the template: compile targeted codebook/layer methods, not the entire serving path.

Decision gate:

- If the delay-pattern bottleneck is Python/control flow, do tensorization/graph-boundary work first.
- If there is a stable repeated codebook loop like S2-Pro and it is material, prototype a narrow opt-in compile path.

### Phase 6: Produce The Go/No-Go Rule

The final issue response should include a table like this:

| Region | Likely Decision | Reason |
| --- | --- | --- |
| MOSS Local Qwen3 backbone decode | No default compile | SGLang CUDA graph already removes launch overhead; PR `#751` shows D vs B within noise. |
| MOSS Local frame-local decode compute | Prefer native CUDA graph/pool work first | The compute is already graph captured; remaining cost is replay boundary, copies, and Python orchestration. |
| S2-Pro codebook loop | Narrow compile can be acceptable | Small repeated loop saw local gain, but end-to-end impact must still be checked. |
| Quantizer/audio encoder | Opt-in compile candidate | PR `#755` already targets this; only useful if profiling shows stage share is material. |
| Host orchestration, request loops, queueing | Do not use compile | Use tensorization, async staging, batching, and graph boundary changes. |

## Validation Checklist

Local Mac validation:

- markdown/docs only for this plan;
- syntax-only checks for any later Python changes;
- no performance claims from this machine.

Remote GPU validation:

- run the Phase 1 c8 primary comparison with three repetitions;
- add c16 only if c8 is inconclusive or contradicts existing PR evidence;
- collect request profiler JSONL and torch profiler traces;
- capture `TORCH_LOGS` for compile runs;
- compare generated audio quality metrics with existing benchmark tooling;
- report both local subpath metrics and end-to-end metrics, with end-to-end metrics as the source of truth.

Regression tests for later code changes:

- existing MOSS Local pipeline tests;
- fixed-seed parity test for `decode_frame()` versus `decode_frame_graphed()`;
- state-pool lifecycle test for acquire/release/generation-step updates;
- graph fallback test for repetition penalty rows;
- benchmark smoke test at concurrency `1` and `8` before full H100 runs.

## Risks

- Maintaining a forked Qwen3 backbone compile path is not justified unless end-to-end gains are reproducible.
- Compile-only comparisons can be misleading if they remove CUDA graph batch padding.
- Dynamic shapes, recompiles, or graph breaks can hide behind good single-request traces.
- CUDA graph static buffers and state-pool tensors must keep stable addresses.
- Changing generation-step ownership can subtly break deterministic sampling or request finalization.
- Quantizer/encoder improvements can be real but irrelevant if they are not on the dominant path for the chosen deployment mode.

## Immediate Next Step

Start with Phase 1 and Phase 2 on the H100 container using this branch. Compare B, D1, D2, and D3 before making a recommendation. Do not implement a separate MOSS Local backbone compile path: use SGLang's existing compile+CUDA-graph runner for the backbone and the new MOSS-local opt-in path only for frame decode.

## 2026-06-12 Correctness Update: D2 Is Not Yet A Safe PR

The full SeedTTS EN run showed a real speed win for the frame-local compile
candidate (`D2`), but subsequent code-hash tracing changed the decision gate.
On the same debug commit, `B` versus `D2` produced different generated audio
code hashes for all `c1` stochastic `n=200` samples while preserving shape and
token counts. A greedy-ish `n=200` run also changed almost every code hash. That
means the compile candidate is changing the frame decoder's model output
directly; this is not just ASR variance, c8 batch packing, vocoder behavior, or
queue timing.

Current mechanical interpretation:

- The pipeline boundary is `preprocessing -> tts_engine -> vocoder`; the code
  hash is captured before vocoder, so the quality drift is inside the AR/local
  decode path.
- The SGLang Qwen3 backbone is not the D2 change. D2 compiles the MOSS-specific
  frame-local callable before manual CUDA graph capture.
- CUDA graph replay is active in both B and D2. CUDA graph only replays the
  captured work; it does not prove that the captured compiled work is equivalent
  to the uncompiled work.
- The compiled function is not a pure feed-forward block. It includes seeded
  sampling, feedback accumulation, and repeated calls to
  `local_transformer.step()`.
- `local_transformer.step()` mutates a persistent module-owned KV cache with
  indexed writes, then immediately reads slices from that cache through SDPA.
  That stateful mutation/read contract is the highest-risk compile boundary.
- `sample_seeded_branchless()` calls SGLang's already-compiled
  `multinomial_with_seed()`. This nested compiled sampler is a secondary suspect,
  but the greedy-ish drift means sampler randomness is not the only possible
  cause.

Do not open the clean `D2` performance PR as a safe optimization until direct
frame-level parity proves semantic equivalence or we narrow the compile boundary
to an equivalent subgraph.

### Next Strict Debug Pass

Goal: isolate the first operation whose raw/eager and compiled outputs diverge,
without running another full end-to-end benchmark first.

Required remote tests:

1. Direct callable parity for `_decode_frame_graphable`.
   - Load the real MOSS Local model on H100.
   - Build identical static inputs for `bs in [1, 2, 4, 8]`.
   - Compare raw callable versus `torch.compile` callable before CUDA graph
     capture.
   - Sweep compile modes: `default`, `reduce-overhead`,
     `max-autotune-no-cudagraphs`.
   - Report stop-token mismatches, per-channel code mismatches, feedback
     `max_abs`, `mean_abs`, and first divergent channel/position.

2. Direct CUDA graph parity.
   - Capture raw `_decode_frame_graphable` and compiled `_decode_frame_graphable`
     into separate manual CUDA graphs with the same static buffers.
   - Replay both on identical inputs.
   - This separates `torch.compile` drift from manual CUDA graph capture/replay
     drift.

3. Component isolation.
   - Compare raw versus compiled `sample_seeded_branchless()` on saved logits,
     seeds, and positions.
   - Compare raw versus compiled `local_transformer.step()` at each local
     position, with cache reset between trials.
   - If `local_transformer.step()` diverges, rerun with an explicitly stateless
     step variant or fresh external KV buffers to test whether module-owned cache
     mutation is the trigger.

4. Generation trace rerun only after the direct parity failure is understood.
   - Run `c1 n=50` with code trace for the best semantically equivalent candidate.
   - If no candidate is code-hash stable under c1, stop the PR and report
     "frame-local torch.compile changes generated codes" as the issue finding.

Acceptance for a compile PR:

- direct frame parity passes for raw versus compiled at all tested batch sizes;
- raw-CUDA-graph versus compiled-CUDA-graph parity also passes;
- c1 code hashes are stable or any difference is explained by an intentional,
  documented sampling change;
- full SeedTTS quality does not regress beyond the project's accepted variance;
- speed win remains material at concurrency 8 after the safe boundary is used.

If parity fails in the current full `_decode_frame_graphable` boundary, the next
engineering direction is not "tune compile harder." It is to create a smaller
compile surface: pure linear/MLP/normalization subgraphs or a stateless
functional local-transformer step whose KV buffers are explicit inputs/outputs.

### 2026-06-12 Smaller Boundary Implementation

Status: implemented on the debug branch as a new experimental compile target.

New factory arg:

```yaml
factory_args:
  frame_decode_torch_compile: true
  frame_decode_torch_compile_mode: max-autotune-no-cudagraphs
  frame_decode_torch_compile_target: logits
```

Targets:

- `full`: previous D2 behavior. Compiles the entire `_decode_frame_graphable`
  callable. This is known to fail direct parity and must not be treated as a
  safe PR candidate.
- `logits`: keeps the outer frame loop, local transformer steps, sampling, and
  feedback accumulation in the raw callable, but compiles only the text/audio
  `F.linear(...).float()` logit projection helpers before manual CUDA graph
  capture.

Why this target is the next smallest useful candidate:

- The failed direct test showed sampler-only parity was clean.
- Standalone `local_transformer.step()` parity was clean.
- The full-loop compile changed the first local-step output when compiled in
  context, so the outer recurrent/sampled loop is too wide.
- Compiling only logit projections tests whether any useful matmul compile
  surface remains while avoiding compile over the recurrent KV mutation/read
  loop and the sampled feedback loop.

Next remote run:

1. Repeat direct frame parity for `frame_decode_torch_compile_target=logits`
   with modes `default`, `reduce-overhead`, and `max-autotune-no-cudagraphs`.
2. Add logits-level metrics:
   - text logits `max_abs`, `mean_abs`, argmax mismatch;
   - per-channel audio logits `max_abs`, `mean_abs`, argmax mismatch;
   - sampled code mismatch by channel.
3. Compare raw callable versus logits-target callable before CUDA graph capture.
4. Capture raw and logits-target callables into separate manual CUDA graphs and
   compare replay parity.
5. If direct parity passes, run `c1 n=50` code trace before any c8/full benchmark.
6. If logits-target parity fails, stop and report that even compiled projection
   lowerings perturb discrete code selection. Then the next candidate is not
   `torch.compile`; it is native CUDA graph/pool work or a custom parity-tested
   kernel boundary.
