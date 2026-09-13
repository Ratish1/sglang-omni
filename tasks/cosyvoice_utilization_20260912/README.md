# CosyVoice utilization investigation

The reported baseline is **3% SM utilization at concurrency 16**; the target remains **30% on the H100**, with useful audio throughput and quality preserved. Neither number has been reproduced in this workspace. The measured server command, mode, installed revisions, exact Nsight metric, and trace are still required to establish the baseline. Source findings below are not measured bottlenecks or promised speedups.

Worktree: `analysis/cosyvoice-utilization-20260912`, based on Omni `645b472cdb2d7b93a1825a5bfa6a603b62b03936`. Local SGLang is `v0.5.19`, `0bcd822377da7b5718e674eaf9c870d349424dd1`. The original user documents and diagnostics are preserved under [inputs](inputs). Research agents used GPT‑5.6 Sol at medium effort, read assigned files completely, and produced factual mechanics reports. Root reconciled those reports and owns the conclusions and plans.

Status 2026-09-13: the H100 results returned under `artifacts/cosyvoice-results-20260913/` were sliced with the [perfkit](perfkit/README.md); the measured ledgers are in [perfkit/RESULTS_20260913.md](perfkit/RESULTS_20260913.md) and the resulting root-cause-ordered program is [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md). That plan supersedes the conditional sequence in `plans/00_stack.md` where the two differ.

Read these in order:

1. [Architecture and runtime contracts](ARCHITECTURE.md): topology, model math, streaming, ownership, and existing SGLang reuse.
2. [Evidence and diagnostic decisions](DIAGNOSIS.md): confirmed mechanisms, falsifiable hypotheses, corrections to the initial material.
3. [H100 runbook](diagnostics/README.md): English baseline collection, future full-corpus A/B, small process-tree NSYS capture, analysis, quality evaluation, and interpretation.
4. [PR sequence and acceptance gates](plans/00_stack.md), then the separate component plans linked there.
5. [Open-PR overlap audit](reports/17_open_prs.md) and [existing optimization assessment](plans/09_existing_optimizations.md).
6. [Coverage audit](evidence/coverage_audit.json) and the sixteen [sector reports](reports): hashes, full read ranges, symbol spans, dependencies, and explicit external boundaries.

The audited scope contains **288 complete files / 159,173 lines** across sixteen sector reports, including the supplied research and relevant external sources. Each covered file was read to EOF; unrelated repository files and uninspected installed binaries remain outside this claim.

The worktree contains an **initial diagnostic implementation**, disabled unless `SGLANG_OMNI_PIPELINE_NVTX=1`, plus the capture/analysis scripts. No optimization is enabled by it. It adds host ranges and same-clock request markers without adding CUDA events, synchronization, tensor reads, model hooks, or a second profiler lifecycle. Actual profiler interoperability and annotation overhead need remote qualification. Full-corpus runs must have profiling and these annotations disabled.

Static validation is recorded in [validation](evidence/validation.json). No local pytest, unit tests, model runs, benchmark runs, or GPU profiling were performed. No H100 utilization improvement is claimed. The component plans are conditional on the trace and numerical gates; incremental HiFT and asynchronous AR scheduling are explicitly not ready to implement by flipping a flag.

This branch tracks the complete research directory, diagnostic scripts and source changes despite the local `tasks/` ignore rule. Clone or fetch the branch on H100 and follow [the handoff](HANDOFF.md). Current work is English-only baseline collection at c1/c16 and small captures. Full English SeedTTS A/B starts after an optimization candidate exists; return plain results under `artifacts/cosyvoice/`, excluding WAVs after remote quality scoring. No archive is required.
