# Qwen3-TTS hidden CUDA synchronization plan

## Status and scope

**Status: H2D candidates implemented; acceptance pending a fresh H100 detector and trace.**

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
Python sequence advanced indexing can hide the same construction.

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

### Current ownership table

| Current mechanism | Producer and first consumer | Ownership result | Candidate action |
|---|---|---|---|
| Speaker mel pageable H2D | CPU mel frontend; speaker encoder on the same current CUDA stream | Unsafe blocking H2D; source is immutable for the copy | Contiguous pinned one-shot source and nonblocking copy |
| Cached speaker embedding pageable H2D | CPU cache clone; prompt construction on the same current CUDA stream | Unsafe blocking H2D in the prompt call | Pinned one-shot source and nonblocking copy |
| Config-derived prompt token sequences | Python integers; embedding lookup on the same current CUDA stream | `torch.tensor(..., device=cuda)` hidden blocking H2D | Device `full` for repeated/scalar data; pinned one-shot source for heterogeneous rows |
| Repetition/suppression mask rebuild | Python request history; mask scatter and logit shaping on the scheduler stream | Unsafe blocking metadata H2D, but shaping semantics must remain exactly as-is | Pinned one-shot row/token/penalty staging; no ownership change |
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

## Implemented candidates awaiting the fresh detector pass

`Qwen3TTSTalker.prepare_decode_buffers` stages dynamic semantic/subtalker sampling
metadata in one-shot pinned CPU tensors, then issues nonblocking copies into
persistent device buffers. Integer and float fields are grouped by identical
destination dtype. Request-id plus per-request epoch controls reuse. PyTorch's
host allocator protects the short-lived source allocations, and current-stream
ordering protects both eager consumers and CUDA-graph replay.

This candidate changes transfer mechanics only. Its gate is disappearance of the
associated synchronization warnings/blocking copies with exact metadata values
and no replacement wait; end-to-end speedup is not assumed.

The next local implementation batch applies the same proof to the remaining
same-stream H2D mechanisms in the ownership table: speaker mel/embedding inputs,
config-derived token rows, and Qwen3-TTS repetition/suppression mask rebuilds.
It deliberately leaves every D2H and cross-thread handoff in the table intact.
None of these candidates is accepted until a current-branch warning pass shows
the exact source warnings are gone and a clean trace finds no replacement wait
inside the selected ranges. Use `error` only to localize a selected source that
still warns; retained D2H boundaries make a global error-mode pass abort early.

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

Question or missing evidence: fresh detector output against the restacked
current-main base after PR #1462 and later Qwen3-TTS changes.

Why it matters; owner or authoritative source; how to resolve it: current code
has changed several old warning sites. Run the worker-process detector and retain
full logs; the warning source plus current owner code determines the rewrite.

Which design, phase, or proof changes with the answer: it determines the next
mechanism selected in execution steps 3-4, but does not change the detector
architecture or the separation of the repetition-penalty future PR.
