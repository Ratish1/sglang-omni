# Qwen3-TTS hidden CUDA synchronization plan

## Status and scope

**Status: Ready for instrumentation; conditional for each production rewrite.**

- Repository: `sglang-omni`
- Worktree: `.worktrees/qwen3-tts-hidden-h2d-sync-v2`
- Branch: `perf/qwen3-tts-hidden-h2d-sync-v2`
- Immutable comparison base: `2cac60e8ac38cf5d3c7091ec3dd15782bc8b1f41`
- Target: detector-confirmed synchronizing CPU/CUDA operations in
  SGLang-Omni-owned Qwen3-TTS execution.
- Numerical contract: preserve tensor values, sampling parameters, generated-token
  semantics, stopping behavior, codec frames, and public request defaults.
- Non-goal: claim a serving speedup from synchronization counts alone.

The current branch history contains an attempted repetition-penalty ownership
change and a corrective revert. Restack that history before proposing the final
change; do not use either commit as evidence for a hidden-sync rewrite.

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

### Prior warning evidence

The accepted warning artifact at `e5d2a31f` found SGLang-Omni-owned locations in:

- Qwen3-TTS request construction and preprocessing;
- Qwen3-TTS model prompt/metadata construction;
- shared model-runner repetition and suppression shaping;
- sampled-token/output processing; and
- final code and waveform materialization.

Those line numbers predate current main and are discovery evidence, not authority
for new edits. Obtain one fresh warning inventory on the restacked current-base
branch before changing any location other than the already-proven sampling
metadata transfer.

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

## Immediate accepted candidate

`Qwen3TTSTalker.prepare_decode_buffers` stages dynamic semantic/subtalker sampling
metadata in one-shot pinned CPU tensors, then issues nonblocking copies into
persistent device buffers. Integer and float fields are grouped by identical
destination dtype. Request-id plus per-request epoch controls reuse. PyTorch's
host allocator protects the short-lived source allocations, and current-stream
ordering protects both eager consumers and CUDA-graph replay.

This candidate changes transfer mechanics only. Its gate is disappearance of the
associated synchronization warnings/blocking copies with exact metadata values
and no replacement wait; end-to-end speedup is not assumed.

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
