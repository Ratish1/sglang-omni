# Qwen3-TTS hidden CPU/CUDA synchronization investigation and repair plan

## Plan status and decision

- **Status:** Conditional. Instrumentation and the H100 experiment are ready;
  production repair slices are selected only from measured critical-path evidence.
- **Repository boundary:** change only SGLang-Omni. The installed `sglang`,
  `qwen-tts`, Transformers, and PyTorch packages are read-only dependencies and
  primary-source references, not edit targets.
- **First model:** `Qwen/Qwen3-TTS-12Hz-0.6B-Base` at revision
  `5d83992436eae1d760afd27aff78a71d676296fc`.
- **First mode:** Base voice cloning with reference audio and reference text, BF16,
  one H100, default colocated `preprocessing -> tts_engine -> vocoder` process,
  default CUDA graphs/compile, deterministic request seed.
- **Planning branch:** `perf/qwen3-tts-hidden-h2d-sync`.
- **Planning worktree:**
  `/Users/ratish/sglang-omni/.worktrees/qwen3-tts-hidden-h2d-sync`.
- **Authoritative repository revision:**
  `b203b0d5f23670467241b05de1f8747102cf073d`, current `upstream/main` when this
  plan was created on 2026-08-12.

### Implementation checkpoint — 2026-08-12

Completed locally on the planning branch:

- typed `config.cuda_sync_debug_mode` transport and rejection of unknown config;
- one locked profiler/sync-debug session per PID and run ID, including colocated
  stage joins, CUDA-process ownership, matching stop, and teardown reset;
- the same-run `TorchProfiler.start` rank regression fix;
- profiler-gated Qwen3-TTS semantic ranges at the static candidates and required
  publication/host-commit boundaries;
- a fresh-process CUDA detector calibration probe under `benchmarks/profiling`;
- a streaming Kineto trace analyzer with per-occurrence attribution, causal or
  explicitly heuristic transfer matching, post-sync bubble, launch-gap, and
  queue-horizon outputs;
- synthetic analyzer tests, process-session tests, route tests, docs, formatting,
  and local compile validation.

Intentionally not implemented yet: any H2D/D2H or stream-publication repair. The
next gate is the remote H100 calibration and baseline capture described below;
that evidence selects a repair branch.

### H100 artifact checkpoint — 2026-08-13

The first returned archive is valid and proves the PyTorch article's compound
blocking-copy mechanism, but it is not yet a clean optimization baseline:

- 6,221 of 6,222 `cudaStreamSynchronize` events immediately follow a
  correlation-backed `cudaMemcpyAsync` on the same host thread;
- the corrected compound interval totals about 3.090 s, split into about 1.917 s
  inside `cudaMemcpyAsync` and 1.171 s inside `cudaStreamSynchronize`;
- correlated transfers split into 4,425 HtoD and 1,796 DtoH occurrences;
- the prior analyzer counted only the synchronization API and therefore hid most
  DtoH host blocking;
- the trace lacks scheduler-thread ATen operators and `qwen3_tts.*` ranges because
  profiling was started from a control thread after scheduler/worker threads
  already existed;
- predictor graph keys 2/4/8/12/16 were captured inside the active trace;
- the claimed 64-request miss list contained only 40 unique reference-audio
  contents (the 16-request list contained 9);
- sync-debug warning logging perturbed the same trace used for timing;
- Torch 2.11 invoked `on_trace_ready` during `stop()`, after which the old code
  attempted a second export and logged `Trace is already saved`.

The branch now corrects those evidence mechanics before any transfer rewrite:

- optional `config.target_stage="tts_engine"` uses acknowledged coordinator
  admin fanout and runs profiler start/stop on every target scheduler thread;
- start/stop responses carry PID/rank/thread ownership, reference/speaker cache
  counters, and Qwen3-TTS lazy predictor-graph keys;
- targeted stop returns only after exactly-once trace export and synchronous gzip
  completion;
- the analyzer pairs `cudaMemcpyAsync -> cudaStreamSynchronize`, reports the full
  compound host block, and subtracts all same-device GPU activity when reporting
  global idle;
- SeedTTS input manifests hash reference bytes and text, support exact replay,
  enforce content-unique reference selection, and construct disjoint warmup lists;
- the revised H100 trace uses sync-debug `default`, fully warms graph keys outside
  the window, and distinguishes pure cache miss from exact replay/hit.

No production synchronization rewrite is selected at this checkpoint. The clean
target-thread rerun is the gate that maps compound copies to exact Omni-owned
operators and ranks their non-overlapping critical-path cost.

### Decision

Start with Qwen3-TTS rather than Qwen3-ASR.

Qwen3-TTS is the cleaner first subject for the requested repository boundary:

- the talker, code predictor, request builder, model runner, sampling kernels,
  streaming-vocoder scheduler, pipeline topology, and tests live in
  `sglang_omni/models/qwen3_tts`;
- the current code contains multiple exact instances of the PyTorch article's
  Python-list-to-CUDA construction trap in request and batch hot paths;
- streaming and non-streaming paths expose distinct device-publication and output
  transfer mechanics that can be measured with existing benchmarks;
- Qwen3-ASR remains a useful later comparison, but its audio-tower forward crosses
  into the installed SGLang dependency and is not the cleanest first ownership
  boundary.

The initial investigation covers only Qwen3-TTS code owned by SGLang-Omni. A
synchronization observed inside an external dependency is attributed and reported,
but not patched in this work.

## Required outcome

Find and remove avoidable eager host/device synchronization from Qwen3-TTS without
changing codec-token semantics, seeded sampling, prompt/cache identity, reference
conditioning, stream ordering, waveform content, completion reason, CUDA-graph
behavior, memory bounds, cancellation, or shutdown.

An **unsafe synchronization** in this plan is a host wait or target-stream drain
that is not the required commit point for a host-visible value and can be replaced
by one of the following while preserving the contract:

- device-native tensor construction;
- immutable/reused same-device metadata;
- allocator-owned pinned staging plus a nonblocking transfer;
- a device-side event/stream dependency;
- retaining a tensor on device across an Omni-owned stage handoff.

A synchronization warning is a finding, not proof of wasted time. The performance
claim requires Torch Profiler and serving A/B evidence showing lost launch lead or
a causally attributable GPU bubble.

## Scope

### In scope

- Qwen3-TTS Base 0.6B first, then a 1.7B smoke/parity run because both use the same
  local implementation.
- Non-streaming and true incremental PCM streaming.
- Reference-code/speaker preparation, talker prefill/decode, predictor sampling,
  codec handoff, and streaming vocoder scheduling where the owner is SGLang-Omni.
- Unique references and repeated references/cache reuse.
- Concurrency 1 for per-request costs and concurrency 16 for batching/launch lead.
- `torch.cuda.set_sync_debug_mode("warn"|"error")`, exactly as recommended by the
  article, scoped to the CUDA-owning serving process after startup.
- Torch CPU+CUDA traces, request-event traces, workload-shaped benchmarks, and
  profiler-free matched A/B trials.

### Non-goals for the first slice

- Editing `/Users/ratish/sglang`, the installed SGLang package, `qwen-tts`,
  Transformers, or PyTorch.
- Fixing every TTS implementation before Qwen3-TTS is qualified.
- Declaring every explicit event/stream wait unsafe.
- Changing sampling defaults, graph buckets, generated-code length, streaming chunk
  thresholds, or voice-cloning quality to gain speed.
- Comparing an eager candidate against a graphed baseline.
- Treating successful CUDA-graph capture or a clean sync-debug log as exhaustive.
- Running sync-debug `error` during checkpoint load, compilation, graph capture, or
  profiler-free performance trials.

## Evidence ledger

| ID | Class | Evidence | Design consequence |
|---|---|---|---|
| E1 | External primary source | [PyTorch DevLog article](https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/) and [source at `bdba1cce`](https://github.com/pytorch/devlogs/blob/bdba1ccec005bb7c5305ec5b13e9f9f2a0e5019d/content/eager/2026-08-11-hidden-h2d-sync.md) | Distinguish transfer from synchronization, inspect advanced indexing/device construction, use sync-debug, and measure the post-sync launch bubble. |
| E2 | External source checked locally | `/Users/ratish/pytorch`, exact article-linked commit `2becd4799c88cc7774b4138e2fb34386f0a8a6c5` | Blocking CPU/CUDA copy calls `cudaMemcpyAsync` then stream synchronize; pinned nonblocking copy uses allocator event lifetime; sync-debug is process-global and incomplete. |
| E3 | Current repository | SGLang-Omni at `b203b0d5...` | Authoritative topology, model/request/runtime/profiler mechanics and tests. |
| E4 | Dependency declarations | `pyproject.toml` pins `torch==2.11.0` and `sglang==0.5.16`; the cookbook requires `qwen-tts==0.1.1` without its dependency set. | The exact installed container build must be fingerprinted; dependencies remain unchanged. |
| E5 | Model artifact | Hugging Face API reported Qwen3-TTS 0.6B Base revision `5d839924...` on 2026-08-12. | Download and serve this immutable revision. |
| E6 | Dataset | `benchmarks/dataset/prepare.py` pins `zhaochenyang20/seed-tts-eval-arrow` to `27f4c1adee83b5b29b7c4b375f6b976324bda308`. | Use the canonical repository ID so the loader applies the revision, and record selected sample IDs/order. |
| E7 | Missing runtime evidence | H100 warning logs, per-process Torch traces, request events, benchmark results, and telemetry. | Required before choosing or accepting a repair. |

The local macOS `python3` does not provide the serving Torch/CUDA environment.
Detector behavior and performance conclusions belong exclusively to the supplied
H100 container.

## Article comprehension applied to this work

The full 282-line source was read, including the disclosure, every paragraph, all
five code blocks, every row in the indexing table, conclusion, and references. The
rendered article contains no figures. The operative model is:

1. A healthy eager CPU producer keeps a queue of GPU work ahead of execution.
2. A blocking H2D can first wait for earlier target-stream work and then erase the
   host's launch lead. The expensive result is often a later GPU dispatch bubble,
   not just the visible synchronization duration.
3. Scalars/device factories can travel through kernel parameters or generate data
   on device. Python lists/tuples/NumPy advanced indices and
   `torch.tensor(python_values, device="cuda")` materialize CPU data and use a
   blocking H2D path.
4. Pinned memory alone is insufficient; the transfer must also request nonblocking
   behavior. Nonblocking behavior alone is insufficient when the source remains
   pageable or its lifetime/mutation is not controlled.
5. Basic indexes/views do not synchronize. Python/NumPy advanced indexes and CPU
   integer tensor indexes do. CUDA boolean indexing synchronizes through
   data-dependent `nonzero`; the CPU boolean-mask path has a pinned async
   optimization.
6. `torch.cuda.set_sync_debug_mode("error")` is a high-value locator, but the API is
   experimental and incomplete. At the cited commit, not every direct device/event
   synchronization and no possible driver-side blocking in a pageable nonblocking
   call is guaranteed to pass through the warning hook.
7. CUDA graph capture is a stricter check for unpinned copies inside the captured
   region, not proof about eager preprocessing, caches, metrics, or teardown.
8. Driver descriptions and the article's approximate small/large-copy thresholds
   are experimental observations, not portable H100 or PyTorch contracts.

Therefore this plan follows the article's detector recommendation and adds the
missing causal/performance proof with Torch Profiler and end-to-end serving trials.

## Current Qwen3-TTS architecture and ownership

### Topology

`Qwen3TTSPipelineConfig` declares three logical stages in one OS process named
`pipeline`, with the model and vocoder placed on `cuda:0`:

```text
API/coordinator process
  POST /v1/audio/speech
        |
        v
pipeline child process (process-global CUDA sync-debug/profiler state)
  preprocessing
    - parse task/language/reference/sampling state
    - tokenize text
    - speaker embedding + reference-code batching/cache
    - construct prompt embeddings and radix-cache identities
        |
        v  same-process dispatch
  tts_engine
    - OmniScheduler request admission/batching
    - local Qwen3TTSTalker prefill/decode
    - local code-predictor graph/eager path
    - local seeded semantic/subtalker sampling
    - generated codec chunks retained on GPU
        |\
        | +-- stream chunks -> vocoder while generation continues
        v
  vocoder
    - local Qwen3TTSStreamingVocoderScheduler
    - external qwen-tts decoder invoked by the local scheduler
    - ordered PCM chunks or final waveform
        |
        v
  terminal HTTP response/stream
```

All three stage endpoints can receive the same profiler broadcast even though they
share one process. Torch Profiler and sync-debug are process-wide, so the control
implementation must be idempotent per `(PID, run_id)` and must not start three
independent process profilers.

### Owners

- `models/qwen3_tts/config.py`: topology and default colocated deployment.
- `models/qwen3_tts/stages.py`: preprocessing/engine/vocoder construction and
  dependency loading.
- `models/qwen3_tts/engine_builder.py`: SGLang generation configuration, local model
  runner, graph/compile setup, adapters, and preprocessing context.
- `models/qwen3_tts/sglang_model.py`: local talker, prompt construction, persistent
  graph/sampling buffers, code predictor, and sampling.
- `models/qwen3_tts/model_runner.py`: prefill embeddings, sampling metadata staging,
  feedback, predictor invocation, and per-request codec chunks.
- `models/qwen3_tts/request_builders.py`: reference preprocessing/batching/cache,
  prompt/radix identities, request construction, stream/final codec handoff.
- `models/qwen3_tts/streaming_vocoder.py`: chunk state, batching, decode stream,
  CUDA graphs, waveform slicing, ordered emission, drain, and shutdown.
- `models/qwen3_tts/sampling_kernels.py`: local optional Triton sampling kernel.
- `pipeline/stage/runtime.py` and `profiler/*`: correct process-side profiling
  control after startup and graph capture.

The external `qwen_tts` package supplies tokenizer/speaker/decoder components, but
the local call boundary and all stage/lifecycle decisions remain observable. A
warning stack whose first non-framework owner is external is recorded as a
dependency residual, not patched here.

## Static synchronization inventory

Static inspection identifies search targets, not performance rank.

| Current site | Static behavior | Initial classification | Omni owner |
|---|---|---|---|
| `sglang_model.py:590-600`, `_build_tts_special_embeds` | Python nested list -> `torch.tensor(..., device=cuda)` per prompt. | Definite blocking H2D; avoidable with immutable device IDs/buffers. | Talker prompt construction. |
| `sglang_model.py:628-642`, `_build_codec_prefill` | Python list, including language-dependent value, reconstructed on CUDA. | Definite blocking H2D; bounded values can be cached/device-generated. | Talker prompt construction. |
| `sglang_model.py:671-675`, `745-749`, `829-838`, `883-887`, `940-958` | Repeated single/small token lists and `[pad] * length` become CUDA tensors. | Definite blocking H2D; article-exact candidates. | Talker prompt construction. |
| `sglang_model.py:1040-1060`, `prepare_decode_buffers` | Six Python lists become CUDA tensors before persistent GPU buffers are filled, once per prefill/decode batch. | Definite repeated H2D stream drains; high-priority hot-path candidate. | Talker sampling metadata. |
| `sglang_model.py:1096-1099`, `_extend_last_index` | Fallback constructs one-element CUDA tensor from a Python list; other path moves `extend_seq_lens`. | Definite on fallback, conditional on normal path/device. | Talker prefill. |
| `sglang_model.py:558-568`, speaker mel | NumPy/CPU mel is moved with ordinary `.to(cuda)`. | Definite blocking H2D on reference preprocessing; size-dependent impact. | Local speaker-preparation boundary. |
| `sglang_model.py:571-575` and request cache restoration | CPU-cached speaker embeddings move with ordinary `.to(cuda)`. | Definite cache-hit H2D if source is CPU. | Prompt/reference cache boundary. |
| `request_builders.py:480-486`, `build_embedding_cache_key_ids` | Entire GPU prompt embedding is converted to float32 CPU and hashed row by row before request admission. | Definite D2H/host commit; potentially avoidable only with a semantically equivalent CPU-owned key. | Prepared-request/radix identity. |
| `request_builders.py:554-555`, cached prompt storage | GPU tensors synchronously copied to CPU cache. | Definite D2H on population; cache lifetime/publication must be preserved. | Voice prompt cache. |
| `request_builders.py:663-683`, reference-code batcher | CUDA event/current stream explicitly synchronized before resolving futures. | Definite host wait, currently publication correctness; unsafe only with a carried event/consumer wait protocol. | Reference-code service. |
| `request_builders.py:1276-1293`, non-stream result | Generated/reference codes are concatenated on GPU then `.cpu()` before a same-process vocoder handoff. | Definite D2H; strong avoidable candidate in default colocated topology. | Engine-to-vocoder payload. |
| Streaming output builder | Keeps codec chunks on their source device; tests explicitly protect this behavior. | Reference good pattern, not a bug. | Engine-to-vocoder streaming handoff. |
| `streaming_vocoder.py:491-506` | Decode input moves to device in private stream, then the stream is synchronized before waveform publication. | Transfer can be no-op for same-device chunks; wait is a publication boundary requiring dependency analysis. | Streaming vocoder scheduler. |
| `model_runner.py:172-180`, `241-260` | Positions, lengths, feedback, and prompt embeddings use `.to(device)`. | Conditional: several are same-device/no-op in default topology; profile before changing. | Qwen3-TTS runner. |
| Async sampled-token collection inherited from the generic runner | Pinned D2H/event resolution makes EOS/token IDs host-visible. | Likely required host commit and overlap reference; do not “fix” only to silence a warning. | Generic model runner. |
| CUDA graph warmup/capture stream waits | Startup-only stream dependencies. | Out of target window; detector is armed after capture. | Graph initialization. |

## Supported state space

| Dimension | First qualification classes | Status and invariant |
|---|---|---|
| Checkpoint | 0.6B Base primary; 1.7B Base smoke | Same local implementation; no dependency/version change. |
| Task | Base voice clone with ICL reference text; x-vector-only reference as secondary | Prompt/reference semantics and generated codes preserved. CustomVoice/VoiceDesign are later coverage, not first repair gates unless changed code is shared. |
| Output | Non-stream WAV; incremental stream PCM | Finish reason, code order, waveform duration/content, and stream terminal order preserved. |
| Load | c1 and c16; bounded fixed sample lists | c1 detects fixed request cliffs; c16 exposes batching/launch lead. |
| Reference state | unique miss; exact replay/hit | Restart between conditions; no accidental cache-state comparison. |
| Generation | fixed seed, default sampling; one explicit alternate sampling tuple | Exact codec-token parity for the fixed seed. |
| Execution | default graph/compile; diagnostic eager only for localization | A/B comparisons use identical graph/compile states. |
| Lifecycle | startup/warm, steady state, abort, drain, shutdown | Debug mode is inactive during startup and reset on stop/failure. No event/buffer leak. |
| Topology | default colocated TP1/one H100 | Split-stage/process overrides are not optimized in the first slice; any changed payload must either remain compatible or retain an explicit fallback and test. |

## Target instrumentation design

### Process-scoped sync-debug window

1. Define a typed `ProfilerStartConfig` with
   `cuda_sync_debug_mode: Literal["default", "warn", "error"]`. Reject unknown
   config keys rather than continuing to ignore `StartReq.config`.
2. Carry the normalized mode through `StartReq`, `ProfilerControlClient`,
   `ProfilerStartMessage` serialization, and TP/process fanout.
3. Add a process-global, locked profiling-session owner keyed by `run_id`. Multiple
   colocated Qwen3-TTS logical stages join the same session idempotently; they do
   not each create or reset independent sync-debug state.
4. Arm `torch.cuda.set_sync_debug_mode(mode)` inside the CUDA-owning child only after
   model construction, compile, graph capture, and external warmup, immediately
   before target traffic.
5. On matching stop, remove each logical-stage participant and reset to `default`
   before trace export/teardown when the process session closes. Teardown/failure
   force-reset regardless of participant state.
6. Log run ID, PID, rank, process/group, logical stages, mode, and transition once.
7. Fix the existing `TorchProfiler.start` same-run early return that references
   `rank` before assignment.

### Semantic trace ranges

Add default-low-overhead `torch.profiler.record_function` ranges at the owner, not
inside every helper:

- `qwen3_tts.preprocess.reference_encode`
- `qwen3_tts.preprocess.speaker_h2d`
- `qwen3_tts.preprocess.prompt_build`
- `qwen3_tts.preprocess.cache_key.dtoh`
- `qwen3_tts.preprocess.cache.dtoh`
- `qwen3_tts.prompt.device_constants`
- `qwen3_tts.sampling_metadata.h2d`
- `qwen3_tts.engine.prefill`
- `qwen3_tts.engine.decode`
- `qwen3_tts.engine.codes.dtoh`
- `qwen3_tts.vocoder.decode`
- `qwen3_tts.vocoder.publish_wait`
- `qwen3_tts.output.host_commit`

Keep request identifiers in request-event records; do not place high-cardinality
request IDs in range names.

### Trace analyzer

Add a proposed `benchmarks/profiling/analyze_cuda_sync_trace.py` that streams
`.trace.json.gz` and emits per-occurrence JSON plus aggregate CSV/JSON:

- PID/thread/logical range and Python stack;
- synchronization API start/end/duration;
- parent `aten::to`, `_to_copy`, `copy_`, `item`, `nonzero`, or direct wait;
- correlated H2D/D2H copy direction, bytes, duration, and stream;
- prior waited GPU completion and next causally subsequent GPU launch;
- post-sync bubble and queue-horizon proxy;
- request/stage interval where correlation is available.

Test it with a small synthetic Chrome-trace fixture. Do not infer cross-process
latency from unrelated clocks.

## H100 profiling workflow

### 1. Environment and detector calibration

Record the container digest, Omni commit/status, installed package versions/build
SHAs, dependency-freeze hash, Python, Torch, CUDA runtime/toolkit, driver, GPU name
and capability, MIG, clocks/power state, CPU/NUMA/affinity, model revision, dataset
revision, topology, server config, and exact command in `manifest.json`.

Run the same isolated sync microprobe defined by the PyTorch-source audit before
trusting warnings. It must cover:

- `.item()`;
- blocking D2H `.cpu()`;
- blocking pageable H2D `.to("cuda")`;
- pinned nonblocking H2D and D2H negative controls;
- explicit stream and device synchronization.

Run every case in both `warn` and `error` modes in a fresh Python process. The
installed build's observed coverage is authoritative. If a positive control does
not warn/raise, a clean server warning log is not absence evidence.

### 2. Immutable model and server launch

```bash
MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
MODEL_REVISION=5d83992436eae1d760afd27aff78a71d676296fc
MODEL_PATH=$(hf download "$MODEL_ID" --revision "$MODEL_REVISION")
mkdir -p /tmp/q3tts-prof

SGLANG_TORCH_PROFILER_DIR=/tmp/q3tts-prof \
sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

Use the container's supported `qwen-tts==0.1.1` installation without resolving its
dependency set, as documented in `docs/cookbook/qwen3_tts.md`.

Wait for full server readiness and graph capture. Warm infrastructure with one
SeedTTS EN sample outside the measured target slice, for example offset 1000. Do
not warm the measured reference list before the unique/miss capture.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model "$MODEL_ID" \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --ref-format references \
  --sample-offset 1000 --max-samples 1 \
  --max-new-tokens 128 --seed 20260812 \
  --warmup 0 --concurrency 1 \
  --output-dir /tmp/q3tts-prof/warmup
```

### 3. Three diagnostic levels

1. **Warning discovery:** `warn`, request events enabled, Torch profiler disabled.
2. **Low-overhead attribution:** `warn`, continuous CPU+CUDA profiler, stacks,
   shapes, memory, and FLOPs disabled.
3. **Targeted stack pass:** repeat the bounded workload with
   `SGLANG_TORCH_PROFILER_WITH_STACK=1` and
   `SGLANG_TORCH_PROFILER_RECORD_SHAPES=1`. Profile memory separately only for a
   lifetime question.

Use `error` only after discovery to stop at one named site and recover its stack.
An error-mode request may fail and is never performance evidence.

For warning-only discovery, call `/start_profile` with `enable_torch:false`, an
explicit `event_dir`, and `config.cuda_sync_debug_mode="warn"`. For a Torch trace:

```bash
RUN="q3tts-c16-nonstream-miss-baseline-$(date +%s)"
BASE=http://127.0.0.1:8000

curl -fsS -X POST "$BASE/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\",\"trace_path_template\":\"/tmp/q3tts-prof/$RUN/trace\",\"event_dir\":\"/tmp/q3tts-prof/$RUN/events\",\"enable_torch\":true,\"config\":{\"cuda_sync_debug_mode\":\"warn\"}}"

# Wait for the matching child-process profiler/debug-start log before traffic.

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url "$BASE" \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --ref-format references \
  --max-samples 64 \
  --max-new-tokens 128 --seed 20260812 \
  --warmup 0 --concurrency 16 \
  --output-dir "/tmp/q3tts-prof/$RUN/client"

curl -fsS -X POST "$BASE/stop_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\"}"

# Wait for matching trace export and stable gzip size/mtime twice.
python -m sglang_omni.profiler "/tmp/q3tts-prof/$RUN/events" \
  --format json --out "/tmp/q3tts-prof/$RUN/request_report.json"
```

The current profiler HTTP response confirms broadcast, not child readiness or trace
compression completion. Do not use repeated identical starts as an acknowledgement.

### 4. Workload matrix

Use a fresh server for every baseline/candidate and miss/hit condition.

| Condition | Preparation | Active capture | Mechanic isolated |
|---|---|---|---|
| c1 non-stream unique | Infrastructure warm only. | Fixed first 16 EN samples, once, c1. | Prompt/reference fixed cost. |
| c16 non-stream unique | Infrastructure warm only. | Fixed first 64 EN samples, once, c16. | Batching, sampling metadata, final codec handoff. |
| c1 stream unique | Infrastructure warm only. | Fixed first 16, `--stream`, c1. | Streaming engine/vocoder boundary. |
| c16 stream unique | Infrastructure warm only. | Fixed first 64, `--stream`, c16. | Chunk batching and decode cadence. |
| repeated reference | Prime one exact reference/request identity before arming. | Replay a fixed request list. | Reference/cache-hit transfer and publication. |

Record the exact sample IDs, texts, reference audio SHA256, request order, seed, and
payload mode. No benchmark warmup runs inside the active trace.

### 5. Trace measurements

For each synchronization interval `[h0,h1]`, identify the waited GPU endpoint `g0`
and first causal GPU work after the wait `g1`, then report:

- count and count/request;
- host wait sum/p50/p95, `S = h1 - h0`;
- copy bytes, direction, duration, and target stream;
- post-sync bubble, `B = max(0, g1 - g0)`;
- queue horizon immediately before/after;
- fraction of preprocessing, prefill, decode, vocoder, first-audio, and total
  request critical path.

Synchronization duration alone is not waste: the GPU may be productively finishing
queued work. Accept causal attribution only when range/stack, CPU API, CUDA copy or
wait, subsequent launch gap, request events, and workload behavior agree.

## Repair-selection branches

Implement each selected owner as a separate coherent A/B commit.

### Branch A — immutable prompt/token construction

Select when prompt-building `torch.tensor(python_values, device=cuda)` occurrences
are material.

1. Register nonpersistent, same-device long buffers for immutable BOS/EOS/PAD and
   bounded codec-prefix token sequences after the model device is known.
2. Index/reuse those buffers for language/speaker variants. Keep buffer values and
   shapes authoritative to checkpoint config.
3. Replace dynamic repeated pad Python lists with `torch.full` on the target device.
4. Do not cache weight-derived embeddings before weights load. Prefer immutable ID
   buffers plus the existing embedding lookup unless profiling separately justifies
   a post-load embedding cache with explicit invalidation.
5. Preserve state-dict compatibility by using nonpersistent buffers.

### Branch B — batched sampling metadata staging

Select when `prepare_decode_buffers` H2Ds are on the prefill/decode critical path.

1. Allocate bounded pinned host staging slots for semantic seed, sub-seed,
   temperature, top-p, top-k, and sampled-row metadata, sized to
   `max_running_requests`.
2. Populate a free slot from Python request metadata on CPU.
3. Enqueue `copy_(pinned_slice, non_blocking=True)` into the existing persistent GPU
   buffers on the model's current stream.
4. Record an event after the last copy. Never mutate/reuse a pinned slot until its
   event completes. Use a bounded ring sized to the scheduler's actual in-flight
   depth; on saturation, wait for the oldest slot rather than grow locked memory.
5. Preserve per-row mapping, mixed greedy/sampled branches, top-k ladder signature,
   per-request seed derivation, and predictor graph inputs exactly.

### Branch C — prompt/radix identity without full embedding D2H

Select only when `build_embedding_cache_key_ids` is material. This branch changes a
cache-identity contract and requires stronger proof.

1. Define the current invariant: two prompt rows may share a radix key only when
   their effective embedding row is identical for the same model/checkpoint/config.
2. Prefer a CPU-owned semantic identity derived before GPU prompt construction:
   model revision, task/mode, language/voice, reference fingerprint/version,
   reference text/code identity, target token identity/position, and checkpoint
   configuration that changes embeddings.
3. Produce per-row stable 63-bit IDs with the same collision-risk class as the
   current 64-bit embedding hash. Do not substitute request-unique IDs silently;
   that disables valid prefix sharing and changes cache performance.
4. Differentially compare equality/inequality partitions between old embedding
   hashes and new semantic IDs across modes, references, languages, texts, cache
   hits, and mixed batches. If equivalence cannot be proven, retain the current
   path and report the D2H as a required admission cost.

### Branch D — keep codec payloads on device

Select when non-stream `apply_sglang_qwen3_tts_result(...).cpu()` is material.

1. Preserve `audio_codes` on its source CUDA device for the default same-process
   engine-to-vocoder handoff, matching the existing streaming behavior.
2. Make the vocoder accept source-device codes and establish an explicit producer
   stream -> decode stream event dependency. `record_stream` protects allocation
   lifetime but does not establish ordering.
3. Preserve a tested CPU/transport fallback for any supported split-process or
   non-CUDA topology that cannot retain/directly transport the tensor.
4. Move the one necessary device-to-host conversion to the terminal audio-byte
   commit, using pinned asynchronous staging where it can overlap decode/encoding.
5. Preserve reference-code prepend length, generated frame order, dtype, shape,
   finish reason, and terminal cleanup.

### Branch E — event-based reference/vocoder publication

Select only when explicit publication waits materially limit request admission,
batching, or streaming cadence.

1. Return a tensor plus readiness event from the reference-code producer. The
   consumer stream waits on the event device-side before use; the request future
   means “work published with dependency,” not “GPU globally complete.”
2. Carry equivalent readiness for vocoder waveform/chunk publication where the
   consumer can remain device-side or perform pinned async D2H.
3. Bound pending events/buffers by queue/batch limits. Define cancellation, retry,
   failure, cache insertion, eviction, drain, and shutdown reclamation.
4. Do not remove the wait if the next consumer is Python/CPU and genuinely requires
   the value. Classify and retain that host commit explicitly.

## Execution order

### Task 0 — provenance and reproducibility

Create the manifest, immutable model/dataset selection, fixed sample lists, warm
protocol, server command, cache-state protocol, and artifact SHA256 inventory.

**Exit gate:** another H100 run can reproduce the identical input sequence and
environment.

### Task 1 — profiler/debug plumbing

Likely files:

- `sglang_omni/serve/launcher.py`
- `sglang_omni/profiler/profiler_control.py`
- `sglang_omni/profiler/torch_profiler.py`
- `sglang_omni/proto/messages.py`
- `sglang_omni/pipeline/stage/runtime.py`
- focused tests under `tests/unit_test/profiler`
- proposed analyzer under `benchmarks/profiling`

Implement the typed process-scoped debug window, idempotent colocated-stage
lifecycle, logs, semantic ranges, trace analyzer, and unit tests.

**Exit gate:** detector controls are default-off, calibrated, scoped to a run, cover
all CUDA work in the child process, and always reset on stop/failure/teardown.

### Task 2 — unmodified H100 discovery and baseline

Run warning-only, low-overhead trace, and targeted-stack passes for every workload
class. Generate a ranked table classifying each occurrence as avoidable,
publication dependency, required host commit, external dependency, off-path, or
unobserved.

**Exit gate:** no repair is chosen from static inspection alone; the ranked owner
has warning/stack/range, CUDA timeline, request-event, and serving evidence.

### Task 3 — smallest measured owner repair

Apply Branch A, B, C, D, or E in that evidence order, one owner per commit. Do not
combine prompt constants, sampling staging, cache identity, and stream-publication
protocols in one performance result.

Likely local implementation/test files:

- `models/qwen3_tts/sglang_model.py`
- `models/qwen3_tts/model_runner.py`
- `models/qwen3_tts/request_builders.py`
- `models/qwen3_tts/streaming_vocoder.py`
- `tests/unit_test/qwen3_tts/test_pipeline.py`
- `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`
- `tests/unit_test/qwen3_tts/test_sampling_kernels.py`

**Exit gate per commit:** targeted occurrence and attributable bubble are reduced
without output, cache, stream, lifecycle, memory, or replacement-wait regression.

### Task 4 — matched A/B qualification

Run profiler-free alternating ABBA trials, at least five trials per candidate,
separately for c1/c16, stream/non-stream, and unique/repeated-reference conditions.
Then collect matched baseline/candidate traces with identical flags/window.

Report:

- successful/complete requests and finish reasons;
- end-to-end latency p50/p95, first-audio latency, inter-chunk gap p50/p95;
- throughput/QPS, RTF/RTFx, generated codec tokens/s and audio seconds/s;
- preprocessing, queue, prefill, decode, predictor, and vocoder request-event time;
- CPU utilization, GPU utilization/power, GPU memory, pinned/pageable host memory;
- warning/sync count, wait time, copy bytes, bubble, and queue horizon;
- reference/cache hit/miss/batch/queue statistics.

**Initial performance acceptance:** targeted occurrences fall by at least 90% where
the design should eliminate them; no replacement global/device wait; profiler-free
p95 and/or throughput improvement exceeds run variance with a confidence interval
excluding zero; c1 does not materially regress while c16 improves.

### Task 5 — correctness, lifecycle, and broader model proof

Run the focused suites:

```bash
pytest -q \
  tests/unit_test/profiler \
  tests/unit_test/qwen3_tts/test_pipeline.py \
  tests/unit_test/qwen3_tts/test_compat.py \
  tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py \
  tests/unit_test/qwen3_tts/test_sampling_kernels.py \
  tests/unit_test/benchmarks
```

Add real-CUDA tests for pinned staging/event reuse and stream ordering; mocks cannot
prove CUDA lifetime. Run the applicable Qwen3-TTS model/serving CI path and a 1.7B
Base smoke after 0.6B acceptance.

Validate:

- exact generated codec-token tensor, order, per-request seed behavior, and finish
  reason under deterministic settings;
- exact prompt/cache identity partition for Branch C;
- stream chunk IDs/order, reference prefix once, no late chunk after abort/EOS,
  final flush, and stream/non-stream semantic equivalence;
- waveform sample rate/channels/duration and numerical parity with a declared
  max-absolute/relative tolerance; WER and speaker similarity at workload scale;
- graph and eager fallback parity, mixed greedy/sampling rows, top-k ladder,
  language and ICL/x-vector modes;
- cancellation during reference encode, talker generation, transfer, and vocoder;
  timeout, failure, retry, cache eviction, drain, and shutdown;
- bounded pinned bytes/events and no reuse-before-event, leak, stale read, or
  double release.

**Exit gate:** all applicable semantic, numerical, lifecycle, memory, compatibility,
and performance gates pass on the fixed H100 environment.

### Task 6 — rollout and expansion

1. Keep profiler/debug instrumentation default-off and document the exact route,
   detector limitations, process scope, and artifact workflow.
2. Land each selected repair with its independent A/B evidence and rollback trigger.
3. Update the Qwen3-TTS cookbook/developer performance note with memory bounds,
   stream ownership, and the qualified benchmark command.
4. Expand the same detector/trace workflow to other **SGLang-Omni-implemented** TTS
   models only after Qwen3-TTS passes. Re-run ownership discovery per model; do not
   assume the same repair.
5. Revisit Qwen3-ASR separately when the desired repository/dependency ownership
   boundary is explicit.

## Proof matrix

| Invariant | Plausible failure | Real proof | Acceptance |
|---|---|---|---|
| Detector is trustworthy for the installed build | Missing hooks make a clean log look safe. | Isolated positive/negative microprobe plus trace. | Coverage recorded; clean log never used beyond observed coverage. |
| Debug state is process-scoped and bounded | Three colocated stages start/reset it inconsistently. | Message/control unit tests and server start/stop/failure logs. | One active session per PID/run; final state always `default`. |
| Prompt IDs/embeddings unchanged | Cached/device-native constants use wrong config/dtype/shape. | Old/new tensor and embedding differential over all languages/modes. | Exact equality. |
| Seeded sampling unchanged | Batched staging scrambles rows or reuses a slot early. | Mixed-row graph/eager CUDA differential with fixed seeds and delayed events. | Exact codec tokens and row mapping. |
| Radix identity is safe | Unequal embeddings share IDs or equal prompts stop sharing. | Equality-partition differential plus cache-hit trace. | No false sharing; any sharing change is explicitly rejected or approved. |
| Codec handoff is ordered | Vocoder reads producer storage before completion/reuse. | Multi-stream real-CUDA test with forced delay and allocator pressure. | Exact codes/waveform; no race under stress. |
| Streaming contract remains correct | Missing/duplicate/out-of-order/late chunks. | HTTP PCM and internal scheduler integration with abort/final flush. | Exact chunk order/terminal behavior and bounded buffering. |
| Performance gain is causal | Profiler perturbation or cache drift produces false win. | Profiler-free ABBA plus matched traces/fresh-server cache protocol. | Delta beyond variance and targeted bubble removal. |
| Resources are conserved | Pinned ring/events/cache entries leak or block shutdown. | Saturation, cancellation, failure, drain, and soak with memory telemetry. | Declared bounds; zero unreclaimed session-owned resources. |

## Artifact bundle

Every accepted run directory contains:

- `manifest.json` with repository/dependency/model/dataset/container/hardware identity,
  server configuration, workload hash, cache/warm state, and exact commands;
- server log and sync-debug output with PID/process/stage/thread attribution;
- client speed results, generated-audio metadata, and per-request records;
- request-event JSONL and rendered report;
- one Torch trace per CUDA-owning PID/rank;
- analyzer occurrence JSON and aggregate CSV/JSON;
- profiler-free trial results and statistical summary;
- SHA256 inventory of all artifacts.

## Open evidence questions

1. **Dominant owner:** Which local site creates the largest causal bubble on H100?
   Resolve with Task 2; it selects Branch A-E but does not change repository scope.
2. **Installed detector coverage:** Does Torch 2.11 in the container report explicit
   event/stream waits and pageable nonblocking behavior? Resolve with the microprobe;
   trace remains mandatory either way.
3. **Radix identity:** Can a CPU-owned semantic key preserve the exact sharing
   partition? Resolve with the differential proof before Branch C; otherwise retain
   the current D2H and classify it as required.
4. **Publication benefit:** Do event-carried reference/vocoder results improve
   admission/chunk cadence enough to justify lifecycle complexity? Resolve with a
   prototype A/B after simpler branches.
5. **Topology compatibility:** Can device-resident non-stream codes use the existing
   split-process CUDA transport for every supported override? Resolve from runtime
   transport tests; otherwise keep an explicit topology fallback.

## Rejected shortcuts

- Adding `non_blocking=True` to pageable sources.
- Pinning a tensor but retaining the blocking copy path.
- Mutating a reusable pinned buffer before its copy-completion event.
- Treating `record_stream` as an ordering dependency.
- Replacing a stream wait with `torch.cuda.synchronize()` or moving it elsewhere.
- Caching weight-derived embeddings before checkpoint weights are installed.
- Making radix IDs request-unique without acknowledging lost prefix sharing.
- Moving GPU codes to CPU merely because the next logical stage has a different
  name while it is colocated on the same device/process.
- Fixing one warning and stopping before the next launch bubble is measured.
- Running warmups or different cache states inside matched traces.
- Editing external dependencies to keep the local design simple.

## Readiness summary

- **Qwen3-TTS ownership choice:** Ready and repository-grounded.
- **Sync-debug/Torch Profiler workflow:** Ready; implementation and H100 calibration
  are the first slice.
- **Repair selection:** Conditional on H100 causal attribution.
- **Qwen3-TTS 0.6B production acceptance:** Conditional on correctness, lifecycle,
  memory, and profiler-free A/B gates.
- **Other Omni TTS models:** Deferred until the first model supplies a proven,
  reusable investigation workflow.
