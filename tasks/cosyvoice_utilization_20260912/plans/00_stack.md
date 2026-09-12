# PR sequence and proof gates

This is a conditional implementation program, not a forecast that these changes sum to 30% SM utilization. Establish the metric and first complete trace, then promote only supported branches. All performance/quality execution is on the user's H100 container. No local pytest or new profiling unit tests are part of this work.

| PR / experiment | Boundary and status | Required evidence before promotion |
|---|---|---|
| D0: full-pipeline diagnostic annotations + tools | Implemented locally, static validation only; remote qualification pending. | All stages/threads visible, graph nodes present, launch correlations checked against GUI, matching request lifecycle, bounded overhead. |
| C0: reconcile open numerical/ordering changes | Qualify #2110 timestep layout and #2086/#2110 FIFO behavior; record the chosen baseline. | Independently reproduce reported effects with frozen inputs and request lifecycles. Keep correctness and performance attribution separate. |
| E0: baseline and existing compile/TRT experiments | Operator experiments, not invented implementation PRs. Run mutually exclusive modes separately. | Reproduced 3% definition; same workload; full quality gates; fallback and batch data. |
| A1: one AR host snapshot | [01](01_ar.md), design specified; implement if redundant host materialization matters. | Token rail, finish, penalty and cache behavior unchanged; fewer D2H operations and improved profiler-off service metrics. |
| S1: explicit streaming coalescing controls / cost-bound admission | [02](02_scheduling.md), independent of A1. | Fragmentation or waits are material; existing effective defaults preserved initially; fairness/latency measured. |
| F1: Flow immutable preparation / workspace ownership | [03](03_flow.md), conditional. | Setup kernels/copies materially contribute; sequential ownership and numerical parity proved. |
| F2: bounded Flow graph execution | [03](03_flow.md); overlaps open PR #1861. Reconcile that implementation before creating another. | Actual shape histogram, memory budget, no alias/race, invalid rows masked, fallback verified. |
| H1: batch compatible streaming HiFT histories | [04](04_hift.md), conditional, requires a parity proof before implementation. | Equal-length batching preserves the entire waveform prefix and emitted suffix under fixed source state. |
| H2: incremental HiFT state | [04](04_hift.md), blocked on state/receptive-field proof. | Complete phase/noise/FFT/F0/convolution continuation and final flush contracts. |
| P1: conditioning compute / cache-key preparation | [05](05_conditioning.md), conditional. | Unique-reference or finalize/CPU stalls; cache isolation and generated-token parity. |
| R1/R2: stage memory / placement | [06](06_runtime.md); vocoder placement overlaps open PR #1933. Reconcile that implementation before creating another. | Actual peak allocation, KV retractions or interpreter/context contention. |
| K1+: targeted kernel substitutions | [07](07_kernels.md), conditional per operation, separate PRs. | Dominant kernels and exact numerical contract; no broad replacement justified by a low aggregate SM number. |
| O1+: native profiler control | [08](08_profiler_control.md), optional and independent. | Process ownership, acknowledgments and export completion semantics. |
| A2: Cosy async AR protocol | [01](01_ar.md), blocked on history/commit design; not an enable flag. | Correct penalties, stop suppression, stream collection, lagged rows, abort/KV lifetime and shutdown drain. |

Before starting an optimization PR, reconcile its owner files and behavior against [the current open-PR audit](../reports/17_open_prs.md). Recheck PR state/head because this inventory is a dated snapshot. Existing accelerators/backends are assessed separately in [09](09_existing_optimizations.md); their presence does not establish correctness or suitability for the measured workload.

## Shared acceptance contract

Before changing compute, freeze exact server launch/config, Omni/SGLang/Cosy/Matcha/x-transformers/FlashInfer/sgl_kernel/Torch/CUDA/TRT identities, checkpoint revision/weight identity, GPU UUID/clocks, CPU quota/affinity, client config, SeedTTS revision/order/reference hashes, language and mode. Repeat A/B in paired order, preferably A→B→B→A, with fresh server state or identical intentional cache warmup. Run the full en/zh streaming/buffered corpus at both c1 and c16; a small concurrency sweep can explain the saturation curve, but c16 remains the target.

Use full en and zh splits independently for acceptance. No generation failures, lost requests, duplicate finals, hangs, unexpected truncation, sample-rate mismatch, or missing saved audio. Inspect output-duration/completion distributions to catch “speedup” by shorter generation. Compare successful audio seconds/wall second, QPS, p50/p95 completion, streaming p50/p95 `audio_ttfp_s`, C50/C100/C200 and max playback underrun. Keep native per-request artifacts, not only rounded summary numbers.

For scheduling/host-copy changes that preserve arithmetic and batch composition, require exact codec IDs for frozen requests and exact output structures; request-local timing/order can differ only where deliberately specified. Floating-point graph/kernel/batch changes require raw mel/waveform comparisons with max absolute/relative error, NaN/Inf checks, length/alignment, and paired full-corpus WER (character-normalized for zh), SIM and UTMOS plus listening around boundaries. Numerical/quality tolerances and latency SLOs are not present in the user inputs: record and agree them before merging a numerically changed path; do not invent thresholds after seeing results. If codec sampling changes, exact waveform equality is no longer an adequate isolation experiment—replay frozen codec/conditioning to isolate acoustics as well as doing end-to-end quality evaluation.

Every PR includes: exact source owner and contract, one measured trigger, before/after trace window and metric definition, profiler-off paired results, numerical/quality evidence, configuration/default behavior, memory cost, failure/fallback handling, and a one-step rollback. Do not merge only because SM activity rose. A winning component change must improve service goodput without violating agreed quality and tail-latency constraints, then the next trace establishes the next priority.

The final program succeeds only when the **same SM metric at c16 reaches at least 30%** in the specified representative measurement interval and the full service acceptance contract passes. If the target cannot be reached for the workload without wasting compute or violating SLOs, report the measured limit and remaining constraints rather than redefining the metric.
