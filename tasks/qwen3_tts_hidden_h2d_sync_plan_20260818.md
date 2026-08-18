# Qwen3-TTS hidden CUDA synchronization plan

## Status and scope

**Status: the six selected H2D mechanisms passed the bounded H100 mechanical
gates at `aebd777e`. Profiler-only ownership ranges for the remaining runtime
synchronizations are implemented and await one bounded attribution trace. No
serving-speedup claim has been established.**

- Repository: `sglang-omni`
- Worktree: `.worktrees/qwen3-tts-hidden-h2d-sync-v2`
- Branch: `perf/qwen3-tts-hidden-h2d-sync-v2`
- Immutable comparison base: `2cac60e8ac38cf5d3c7091ec3dd15782bc8b1f41`
- Target: detector-confirmed synchronizing CPU/CUDA operations in
  SGLang-Omni-owned Qwen3-TTS execution.
- Numerical contract: preserve tensor values, sampling parameters, generated-token
  semantics, stopping behavior, codec frames, and public request defaults.
- Non-goal: claim a serving speedup from synchronization counts alone.

## Explicit future work: repetition-penalty ownership

Qwen3-TTS currently applies repetition-penalty logit shaping in Omni and again
inside SGLang's sampler. Official Qwen generation applies the requested/default
penalty once. Correcting that ownership intentionally changes the probability
distribution and is therefore a separate future correctness PR with its own
fixed-seed numerical and quality qualification.

This hidden-sync work must not remove, square, replace, or otherwise reinterpret
the repetition penalty. Detector warnings in repetition-mask construction may
only be repaired by preserving the existing shaping operation while changing
the transfer mechanism.

## PyTorch mechanism and detector contract

The source article is:

- https://docs.pytorch.org/devlogs/eager/2026-08-11-hidden-h2d-sync/

`torch.tensor(python_data, device="cuda")` first materializes host storage and
then calls `.to(device, non_blocking=False)`. PyTorch's CUDA copy path performs
`cudaMemcpyAsync` followed by `cudaStreamSynchronize`, draining the launch queue.
Python advanced assignment can hide the same construction even when its mask
and indices are already CUDA tensors. In Torch 2.11, `cuda_tensor[indices] =
python_scalar` wraps the scalar in a CPU tensor, and `index_put_` performs a
blocking `.to(self.device())` before launching its CUDA kernel.

The primary source detector is:

```python
torch.cuda.set_sync_debug_mode("error")
```

The installed H100 Torch build empirically detects scalar `.item()`, pageable
H2D/D2H, and explicit stream synchronization. It does not detect
`cudaDeviceSynchronize`; pinned nonblocking H2D/D2H controls do not trigger it.
The detector is process-global, incomplete for some CUDA operations, and does
not cover all `torch.distributed` or `torch.sparse` synchronization.

Arm it only after model load, compile, CUDA-graph capture, and warm-up. The
existing `/start_profile` stage-control boundary carries
`config.cuda_sync_debug_mode`; `enable_torch=false` keeps Torch Profiler out of
the warning-discovery pass. `/stop_profile` resets the detector before profiler
export and cleanup.

The start endpoint broadcasts asynchronously. Do not send target traffic until
every CUDA-owning PID/rank expected for the target stage logs:

```text
CUDA sync debug enabled run_id=... mode=... pid=... rank=... participant=...
```

## Evidence ledger

### Repository and external facts

- Qwen3-TTS stages can be colocated in one OS process; PyTorch sync-debug state is
  process-global. A process-scoped idempotent lifecycle is required.
- TP leaders fan profiler control messages to follower processes, so every CUDA
  rank can arm its own process-global detector.
- Qwen3-TTS uses a deliberately single-stream SGLang execution bridge. An H2D
  enqueued on the current stream before a consumer requires no explicit event.
- PyTorch's nonblocking pinned-memory copy records the copy stream against the
  host allocation. The caching host allocator does not recycle that allocation
  until the recorded event completes.
- SGLang PR #28076 uses the same dynamic-host-metadata pattern: construct pinned
  CPU tensors and copy them to the device with `non_blocking=True`.

The corresponding PyTorch implementation was checked directly, not inferred:

- `torch/csrc/utils/tensor_new.cpp` constructs Python sequence data in a CPU
  tensor and calls `to(..., non_blocking=false)` for the requested device;
- `aten/src/ATen/native/cuda/Copy.cu` dispatches a blocking CPU/CUDA copy to
  `memcpy_and_sync`, while its nonblocking branch calls `cudaMemcpyAsync` and
  records the stream on the caching host allocation; and
- `aten/src/ATen/core/CachingHostAllocator.h` with
  `aten/src/ATen/cuda/CachingHostAllocator.cpp` records an event on free and
  withholds the pinned block from reuse until that event completes.
- `torch/csrc/autograd/python_variable_indexing.cpp` deliberately wraps a Python
  scalar assigned into a CUDA tensor as a CPU tensor, while
  `aten/src/ATen/native/TensorAdvancedIndexing.cpp` moves that scalar to the
  indexed tensor's device with a blocking `.to(...)`; and
- the `index_fill_.int_Scalar` and `scatter_.value` schemas carry a scalar into
  their CUDA kernels without materializing that CPU-to-CUDA tensor.

### Prior warning evidence

The accepted warning artifact at `e5d2a31f` found SGLang-Omni-owned locations in:

- Qwen3-TTS request construction and preprocessing;
- Qwen3-TTS model prompt/metadata construction;
- shared model-runner repetition and suppression shaping;
- sampled-token/output processing; and
- final code and waveform materialization.

The artifact provenance identifies commit `64654a34`. The warning locations were
mapped back to that exact source revision and then compared with the current
source. They authorize candidate rewrites only where the same operation and
ownership are still present. A fresh current-branch warning pass remains the
qualification gate; it determines which warnings actually disappeared and
prevents stale line numbers from being reported as current evidence.

### First v2 qualification result

The bounded H100 pass at tested HEAD `32d8bf30` completed every request but
correctly rejected the candidate for two independent reasons:

- sync-debug warned at the three advanced scalar assignments in
  `qwen3_tts/model_runner.py` that populated repetition/suppression masks; and
- the clean trace contained CUDA runtime activity but no `qwen3_tts.*` or
  scheduler-thread `aten::*` events.

The mask indices were already pinned and copied nonblocking. The remaining
hidden H2D was the Python `True` on the right-hand side. Commit `86884103`
replaces ragged advanced assignment with flat `index_fill_(..., True)` and the
one-token-per-row steady update with `scatter_(..., True)`. Both use PyTorch's
Scalar overload and preserve the existing mask bits and repetition ownership.

The profiler started on the control-plane thread after Omni's scheduler and
preprocessing threads already existed. Kineto CPU callbacks are thread-local by
default, which explains the missing CPU parents while CUPTI CUDA activity was
still present. Commit `167a2573` enables Torch 2.11's
`_ExperimentalConfig(profile_all_threads=True)`. PyTorch's own profiler test
covers both pre-existing sibling threads and threads created inside the active
window. This is diagnostic scope only; it does not change inference execution.

### Corrected Sections 1-5 qualification

The returned bundle `q3tts-hidden-sync-167a2573-20260818T165158Z` was captured
from tested revision `aebd777ee5b67f354fe00b86c3239cecd6583415` on an H100
with Torch `2.11.0+cu130`. All 177 manifest hashes verify locally.

The warning-only c1 window completed 16/16 requests. It enabled and disabled the
process-scoped detector exactly once, and emitted no warning from the six
selected Qwen3-TTS source paths. The clean c16 window completed 64/64 requests;
all scheduler-thread ATen parents and expected semantic ranges were present.
Every selected range contained correlated pinned asynchronous H2D activity and
zero stream, event, or device wait:

| Selected range | CPU range calls | Correlated H2D copies | Bytes | GPU copy time |
|---|---:|---:|---:|---:|
| `qwen3_tts.preprocess.speaker_mel_h2d` | 44 | 44 | 9,531,392 | 392.828 us |
| `qwen3_tts.preprocess.speaker_embedding_h2d` | 64 | 64 | 131,072 | 80.768 us |
| `qwen3_tts.prompt.token_ids_h2d` | 192 | 192 | 4,096 | 158.434 us |
| `qwen3_tts.prompt.ref_code_h2d` | 64 | 64 | 466,944 | 116.573 us |
| `qwen3_tts.sampling_masks.rebuild` | 113 | 309 | 7,292,892 | 801.428 us |
| `qwen3_tts.sampling_metadata.h2d` | 351 | 672 | 34,720 | 568.695 us |
| **Total** | **828** | **1,345** | **17,461,116** | **2,118.726 us** |

These values explain why a mechanical pass is not itself a speedup claim. The
copies still execute before their same-stream consumers; the rewrite removes
the host-side stream drains, not the approximately 2.12 ms of device transfer
work. The six CPU annotation ranges contain Python construction, pinned-host
allocation/copy, and device launches, so their summed range duration must not be
mistaken for eliminated time.

The same clean trace still contains 1,742 synchronization occurrences outside
the six candidates. The important remaining mechanisms are:

| Remaining mechanism | Count | Compound host/API time | GPU transfer time | Interpretation |
|---|---:|---:|---:|---|
| Unscoped scalar D2H (`aten::_local_scalar_dense`) | 854 | 479.433 ms | 2.050 ms | Highest-frequency hidden control-flow synchronization; warning locations are known, but the low-overhead trace has no Python stacks to map counts to locations. |
| Unscoped pageable H2D (`aten::copy_`) | 234 | 91.661 ms | 1.542 ms | Remaining framework/Qwen tokenizer or preprocessing copies; source-frequency attribution is still missing. |
| Cache artifact D2H | 88 | 191.451 ms | 0.244 ms | Immediate CPU cache publication under the current artifact contract; not removable by setting `non_blocking=True` alone. |
| Embedding cache-key D2H | 64 | 70.689 ms | 0.426 ms | CPU BLAKE2 consumes the rows immediately; requires an algorithm/ownership redesign. |
| Final codec-code D2H | 64 | 85.894 ms | 0.175 ms | Default placement hands the tensor locally to the vocoder, but the edge is also process-safe; any device-resident fast path must preserve the cross-process fallback. |
| Unscoped final/output D2H | 64 | 39.435 ms | 1.327 ms | Consistent with final waveform materialization; must be scoped before redesign. |
| Async completion event wait | 351 | 2,051.324 ms of host wait | n/a | Expected resolve boundary for one-step lookahead. The GPU has a median 5.57 ms queue horizon at wait entry, so this wait is not evidence of wasted GPU time. |

Across the 5.718 s GPU-active workload span, the union of GPU events is
3.039 s and the device is globally idle for 2.679 s in this profiler-perturbed
capture. The 715.666 ms union of blocking-copy host intervals shows real launch
queue drains remain, but it cannot be read as directly recoverable wall time:
preprocessing threads overlap, D2H consumers can be immediate, and causal
intervals from different occurrences overlap.

Request event timelines locate the observed c16 latency without attributing it
to a specific sync: preprocessing averages 197.036 ms, the TTS engine 873.453
ms (807.612 ms after prefill), and the vocoder 235.337 ms. Same-process hop time
is approximately 0.03 ms in either direction. These intervals overlap across
requests and therefore are not additive throughput costs.

### Current ownership table

| Current mechanism | Producer and first consumer | Ownership result | Candidate action |
|---|---|---|---|
| Speaker mel pageable H2D | CPU mel frontend; speaker encoder on the same current CUDA stream | Unsafe blocking H2D; source is immutable for the copy | Contiguous pinned one-shot source and nonblocking copy |
| Cached speaker embedding pageable H2D | CPU cache clone; prompt construction on the same current CUDA stream | Unsafe blocking H2D in the prompt call | Pinned one-shot source and nonblocking copy |
| Config-derived prompt token sequences | Python integers; embedding lookup on the same current CUDA stream | `torch.tensor(..., device=cuda)` hidden blocking H2D | Device `full` for repeated/scalar data; pinned one-shot source for heterogeneous rows |
| Repetition/suppression mask rebuild | Python request history; mask scatter and logit shaping on the scheduler stream | Unsafe blocking index metadata and advanced-assignment scalar H2D, but shaping semantics must remain exactly as-is | Pinned one-shot flat-index/penalty staging plus Scalar `index_fill_`/`scatter_`; no ownership change |
| Semantic/subtalker sampling metadata | Python request metadata; persistent predictor buffers on the scheduler stream | Previously proven unsafe and already qualified mechanically | Keep the existing pinned nonblocking restaging candidate |
| Embedding cache-key D2H | CUDA prompt embeddings; CPU BLAKE2 hash immediately afterward | Synchronization is required by the current CPU algorithm | Retain until the cache-key algorithm itself is redesigned |
| Speaker-artifact cache D2H | CUDA embedding/codes; shared CPU cache with concurrent readers | Potentially deferrable, but no event is owned by a cache entry today | Retain; requires an event-gated cache artifact protocol |
| Reference-code encoder handoff | Private CUDA encode stream; future resolved to another thread | Explicit event/stream wait is the correctness boundary | Retain |
| Prompt reference-code H2D | CPU cached code; ICL embedding lookup on the same preprocessing stream | Unsafe blocking H2D; the later mandatory cache-key D2H completes this stream before the prepared request crosses threads | Return the device copy after pinned nonblocking staging; keep the prepared-request range to prove its normalization is a no-op |
| Sampled token D2H and output `.tolist()` | CUDA sampled ids; pinned ping-pong buffer; CPU output processor | Current code already gates the first CPU read on the recorded event | Retain; verify the runtime supplies `host_token_ids` |
| Final codec-code D2H | Scheduler CUDA tensors; CPU `StagePayload` handed to vocoder | Potential overlap exists, but the payload has no completion event contract | Retain until the stage handoff owns and waits on an event |
| Vocoder private-stream synchronization | Private decode stream; returned waveform and error verdict | Explicit completion boundary, not a hidden pageable copy | Retain |

## Rewrite classification

For every warning, classify the value and consumer before editing:

| Value/operation | Required rewrite | Proof obligation |
|---|---|---|
| Scalar or regular sequence derivable on GPU | Device factory such as `full`, `arange`, or basic slices | Exact shape, dtype, device, and values |
| Python list or CPU tensor dynamic per request/batch | Pinned CPU staging plus `non_blocking=True` copy on the consumer stream | Source is immutable; allocator/event protects lifetime; consumer stream is ordered |
| Reused metadata | Persistent device destination; restage only when identity/epoch changes | No stale reuse across request-id reincarnation, retract, or batch change |
| Python-list advanced index | Basic indexing when contiguous, otherwise a prebuilt/device index | Selected elements and ordering are identical |
| D2H result with delayed CPU consumer | Pinned host destination plus event-gated consumption | No CPU read before event; cancellation and buffer reuse are safe |
| D2H result required immediately by control flow | Retain the synchronization unless the ownership protocol is redesigned | Never label an unavoidable wait as removed |

Do not convert pageable memory to `non_blocking=True` and call it asynchronous.
Do not add a new stream without a complete producer/consumer event protocol.
Do not delete functional logic merely because its implementation warns.

## Execution plan

1. **Land minimal detector plumbing.** Carry `cuda_sync_debug_mode` over the
   existing profiler control message; arm once per CUDA process after profiler
   setup; reset on stop and teardown. Do not alter profiling semantics otherwise.
2. **Capture a current-base warning inventory.** Warm a fresh Qwen3-TTS server,
   start a warning-only session, run one bounded non-streaming cache-miss corpus,
   stop, and preserve complete worker logs. Use `error` only to localize the
   first unclear stack because it intentionally aborts the request.
3. **Build an ownership table.** For every current Omni-owned warning record
   frequency class (startup/request/batch/token/final), direction, tensor
   shape/dtype/device, producer, first consumer, stream, source/destination
   lifetime, and whether CPU consumption can be deferred.
4. **Repair one mechanism class at a time.** Select the highest-frequency
   removable synchronization whose numerical and lifetime contract is proven.
   Preserve business logic and commit boundaries by mechanism, not by file.
5. **Repeat warning discovery.** The repaired call site must disappear. Record
   remaining warnings and detector coverage limitations; a clean warning log is
   not proof that every synchronization is gone.
6. **Use one clean Torch trace for prioritization.** Run with sync-debug disabled;
   require the selected `cudaMemcpyAsync -> cudaStreamSynchronize` occurrences
   and any attributable post-sync GPU bubble to fall without a replacement
   stream/event/device wait.
7. **Qualify end to end only after correctness.** Run the full 1,088-sample
   canonical corpus at most once per accepted candidate. Omit repetition-penalty
   overrides. Do not claim performance from a single noisy ordering; report a
   synchronization cleanup as such when throughput and p95 remain within noise.

## Mechanically qualified candidates

`Qwen3TTSTalker.prepare_decode_buffers` stages dynamic semantic/subtalker sampling
metadata in one-shot pinned CPU tensors, then issues nonblocking copies into
persistent device buffers. Integer and float fields are grouped by identical
destination dtype. Request-id plus per-request epoch controls reuse. PyTorch's
host allocator protects the short-lived source allocations, and current-stream
ordering protects both eager consumers and CUDA-graph replay.

This candidate changes transfer mechanics only. Its gate is disappearance of the
associated synchronization warnings/blocking copies with exact metadata values
and no replacement wait; end-to-end speedup is not assumed.

The same-stream H2D batch now covers speaker mel/embedding inputs,
config-derived token rows, and Qwen3-TTS repetition/suppression mask rebuilds.
Ragged mask entries are encoded as flat host indices, staged through pinned
memory, and applied by the CUDA `index_fill_` Scalar overload. The steady decode
update uses the CUDA `scatter_` Scalar overload directly on the per-row token
index. Every D2H and cross-thread handoff in the ownership table remains intact.
The corrected warning pass shows the exact selected source warnings are gone,
and the all-thread clean trace finds no replacement wait inside the selected
ranges. This accepts the six rewrites as synchronization cleanups, not as a
demonstrated serving speedup. Use `error` only to localize a selected remaining
source; retained D2H boundaries make a global error-mode pass abort early.

Semantic profiler ranges are enabled only while `TorchProfiler` is active. The
candidate ranges are `qwen3_tts.preprocess.speaker_mel_h2d`,
`qwen3_tts.preprocess.speaker_embedding_h2d`,
`qwen3_tts.prompt.token_ids_h2d`, `qwen3_tts.prompt.ref_code_h2d`,
`qwen3_tts.sampling_masks.rebuild`, and
`qwen3_tts.sampling_metadata.h2d`. Retained ownership boundaries also have
ranges for cache-key/cache D2H, reference-encode publication, prepared
reference-code normalization, and final codec-code D2H. This lets one clean
trace separate the selected mechanisms from intentionally retained waits
without enabling stack collection for the whole server.

The follow-up attribution revision adds only coarse owner ranges around calls
whose internals produced the remaining warnings:
`qwen3_tts.preprocess.text_tokenizer`,
`qwen3_tts.preprocess.reference_tokenizer.encode`,
`qwen3_tts.preprocess.speaker_encoder.forward`,
`qwen3_tts.preprocess.prompt.build`,
`qwen3_tts.sampling.base_pipeline`, and
`qwen3_tts.vocoder.tokenizer.decode`. Existing narrower ranges remain nested,
so the analyzer can prefer the most specific Omni-owned boundary. These ranges
do not authorize a transfer rewrite; they establish owner, frequency, host
blocking time, and correlated bytes for the next design decision.

## Proof gates

- Detector lifecycle: expected CUDA PID/rank logs enable and disable exactly once
  per run; CPU-only processes do not initialize CUDA.
- Numerical: destination metadata equals the baseline values for every active row
  and dtype before its first consumer.
- Lifetime: no pinned source can be recycled before its H2D completes; persistent
  destinations are not overwritten across unordered streams.
- CUDA graphs: no new graph key, capture failure, or replay-time host transfer.
- Mechanical: selected warning and compound blocking-copy count reach zero; no
  replacement synchronization appears.
- Functional: all requests finish; cache and graph snapshots retain their prior
  contract; sampling defaults and output shaping remain unchanged.
- Memory: pinned-host working set remains bounded by allocator size classes and
  observed batch metadata sizes; do not infer this from GPU memory.

## Open evidence before the next production rewrite

Question or missing evidence: the clean low-overhead trace contains no Python
stacks for the 854 scalar D2H and 298 unscoped pageable copies. The warning pass
identifies unique source locations, but it does not report per-location counts
or durations. Profiler-only ownership ranges now cover text tokenization,
reference-tokenizer encoding, speaker-encoder execution, prompt construction,
SGLang sampling, and vocoder-tokenizer decoding; one trace is required to
populate them.

Why it matters; owner or authoritative source; how to resolve it: run one
bounded clean c16 trace and require the unscoped scalar/H2D/D2H counts to move
into the new ownership ranges. Use a short stack-enabled trace only for any
aggregate that remains ambiguous after coarse ranges; do not profile another
full corpus.

Which design, phase, or proof changes with the answer: the attribution decides
whether the next repair is a same-stream pinned H2D, a device-resident local
stage handoff with a cross-process fallback, an event-owned delayed D2H, or an
unavoidable immediate CPU-consumer boundary. No transfer rewrite is authorized
until that ownership is established. The separate repetition-penalty semantic
future work remains out of scope.
