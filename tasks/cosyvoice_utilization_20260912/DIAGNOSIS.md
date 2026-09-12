# Evidence-led diagnosis

## What the target means

Record the exact Nsight metric label, unit, aggregation, GPU UUID/MIG mode, clock policy, capture interval, and sample frequency behind “3%.” Preserve that definition for the 30% target. `nvidia-smi` GPU utilization, kernel timeline coverage, SM Active, achieved occupancy, and tensor-core activity are different measurements. The analyzer reports captured-kernel coverage separately and exports explicitly selected GPU metric samples without renaming them.

For interval W, let C be the union of captured compute intervals. `coverage=|C|/|W|`; summed kernel durations can exceed |W| when kernels overlap. Low coverage directs attention to submission/readiness/synchronization. High coverage with low SM activity directs attention to per-kernel parallelism, memory dependencies, dtype and launch geometry. Their relationship is diagnostic, not an identity. Full request residence time is not a sum of stage GPU times when work overlaps.

For fixed useful work, improving utilization alone is insufficient: padding, recomputing more history, or extending requests can increase activity while reducing throughput. Always pair the SM metric with successful audio seconds/wall second, errors, audio duration distribution, completion latency, streaming TTFA/continuity, and quality. Performance percentages from individual components cannot be multiplied into a guaranteed 10× system gain.

## Open changes that affect the reference baseline

[The dated PR audit](reports/17_open_prs.md) identifies existing implementations of buffered whole-solver graphs (#1861), isolated-vocoder placement (#1933), and reference batching with a tokenizer backend replacement (#1693). Their performance claims remain external evidence. PR #2110 proposes a timestep-row shape change and reports quality effects; #2086 and #2110 both alter FIFO message ordering. Reproduce and reconcile those proposed corrections before treating a graph/kernel/scheduler candidate as equivalent to the audited baseline. This diagnostic branch does not import any of them.

## Hypotheses and disproof

| Candidate | Established source mechanism | Evidence needed / disproof | Plan |
|---|---|---|---|
| AR host serialization | Codec `.tolist()` precedes output processing; normal scheduler loop waits for this step. Reporting layers can repeat D2H. | Correlated copies/waits and host gaps consume meaningful time with AR requests ready. Disprove dominance if GPU compute remains active throughout or copies are negligible. | [01 AR](plans/01_ar.md) |
| Acoustic batch fragmentation | Client c16 can become many singleton native Flow calls; separate 30 ms peer waits; groups depend on readiness/hop/offset. | Realized batch histogram, valid/padded frames, peer-wait spans, ready cohorts, and first/follow-up distributions. A large healthy batch with no waits disproves this as the main cause. | [02 scheduling](plans/02_scheduling.md) |
| Flow launch overhead / mask cost | Ten Python Euler iterations; 22-block estimator; repeated masks/workspaces/noise transfer; compile off. | Many short Flow kernels separated by CPU submission gaps; compare existing compiled mode after warmup. Long dense kernels with little launch gap argue for arithmetic/memory work instead. | [03 Flow](plans/03_flow.md) |
| HiFT serialization / repeated history | Streaming batches Flow but executes HiFT per row; F0 is float64; cumulative prefix grows every hop. | HiFT kernel time and bytes grow with cumulative mel, FP64 kernels visible, per-request calls dominate. Short HiFT slices reject prioritizing this. | [04 HiFT](plans/04_hift.md) |
| Conditioning stalls | Independent CPU encoders, per-session threads, full embedding D2H/hash inside finalization lock. | Long validation/reference/finalize spans, empty AR queues, CPU cgroup throttling or lock contention. Warm-cache homogeneous runs may conceal this; cold unique-reference traces must also be sampled. | [05 conditioning](plans/05_conditioning.md) |
| Colocation contention | Shared Python process and GPU; vocoder/prep scheduler threads; no explicit stage memory budget in defaults. | Time-correlated runnable/waiting CPU threads, GPU stream serialization or memory pressure/retractions. Default local object dispatch rules out stage SHM copies as the initial explanation. | [06 runtime](plans/06_runtime.md) |
| Specific inefficient kernel | Backend/dtype/shape dispatched at runtime; AR and DiT have different normalization/attention contracts. | A small set of kernels dominates Nsight Systems; a bounded Nsight Compute pass then establishes memory/compute/occupancy limits. Kernel names alone do not establish causes. | [07 kernels](plans/07_kernels.md) |

Several causes may coexist. Rank them by attributable wall-clock opportunity and effect on request critical paths after the first complete trace. A gap overlapped by a peer-wait range does not prove that wait caused the gap: inspect readiness and outstanding GPU work. Run one change at a time, then repeat profiling because the limiting stage may move.

## Corrections and retained value from the initial material

The user's playbook and analysis are retained as hypotheses and experiment guidance. Their strongest recommendations survive: verify imports/config first, observe the complete loop, separate streaming from buffered, use few samples for tracing, and evaluate full SeedTTS separately. Apply these source-grounded corrections:

- Default stages are colocated in one worker process. Stage IPC, process-launch overhead per request, and host relay copies are not assumed hot paths.
- Upstream scheduling and decode graphs are already reused. Custom Cosy prefill is eager and bypasses upstream prefill graph execution. Replacing the scheduler wholesale or enabling overlap does not supply the missing Cosy lifecycle contract.
- The current first AR flush is 28, independent of prompt padding. The cookbook's `28 + prompt_pad` description is stale. The cookbook also contains conflicting repetition penalties; source uses 1.21.
- SGLang kernel code includes the AOT tree under `python/sglang/kernels/aot`; it is not wholly absent/external. Installed wheel identity and dynamically imported FlashInfer remain runtime evidence.
- Flow batching and streaming batching already exist. TRT's current pairwise CFG execution does not preserve a large packed DiT batch as one engine invocation.
- Prompt encoding has an existing cache and single-flight service. The public HTTP path resolves a remote reference once before conditioning; the internal raw-URL hook's two loads are not proof of double HTTP download on normal public requests.
- Existing DiT compile is estimator-only. CUDA graph enablement in AR does not capture Flow or HiFT.
- The provided SQLite union calculation is useful for one-device kernel coverage; it is insufficient for SM measurement or stage overlap/causal attribution. Preserve process-scoped correlation, graph-node capture, memcpy and unknown attribution.
- Native profiler control is unacknowledged in Omni. Shared-process stage handlers skip a duplicate start, so a direct `TorchProfiler.start` rank-before-assignment defect does not automatically explain the normal c16 run. Upstream control awaits results but checks only result zero, so it is not a ready aggregate-ack solution either.
- The existing default sweep uses 256 samples, not the full evaluation corpus. The new runner defaults to the full selected split; profile mode requires an explicit small cohort.

## Required first evidence packet

Capture baseline launch/config and package/source identities, H100 UUID and CPU quota/affinity; full profiler-off en/zh SeedTTS streaming and buffered results; one small trace per mode at c16, plus c1 for dispatch comparison; input hashes; actual backend/graph logs; quality outputs and error rows. Include the original 3% report if available. Until those arrive, the root cause and the size of each recoverable opportunity remain **unmeasured**.

Timing arithmetic uses the exported integer nanosecond clock. It preserves the recorded intervals exactly; it does not make physical measurements “bit exact.” Clock resolution, sampling, collection overhead, trace loss and launch-correlation coverage must be qualified. Do not subtract `time.time_ns()` JSONL events from Nsight timestamps. The new NVTX request markers avoid that clock mismatch.

Primary profiling references: [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html), [SQLite schema and analysis](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html), [DCGM profiling metric definitions](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html), [NVTX annotations](https://nvidia.github.io/NVTX/python/annotation_types.html). Installed tool help/version takes precedence over a command copied from newer documentation.
