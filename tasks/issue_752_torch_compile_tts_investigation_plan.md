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

### Phase 3: Do Not Duplicate Existing Compile PRs

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
