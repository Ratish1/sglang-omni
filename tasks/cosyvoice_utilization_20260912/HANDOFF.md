# Branch handoff

Branch: `analysis/cosyvoice-utilization-20260912`.

Local worktree: `/Users/ratish/sglang-omni/.worktrees/cosyvoice-utilization-20260912`.

The branch tracks the source annotations, runnable diagnostics, component plans, research reports and provenance. The local `tasks/` ignore rule was overridden for this research directory when staging it. The original main source tree is unchanged. A discovery copy remains in the main checkout's `tasks/cosyvoice_utilization_20260912`; continue edits in the worktree.

From the worktree, publish the prepared commits:

```bash
git push -u origin analysis/cosyvoice-utilization-20260912
```

On H100, from an existing Omni clone, fetch the branch from the fork and create an isolated checkout:

```bash
git fetch https://github.com/Ratish1/sglang-omni.git analysis/cosyvoice-utilization-20260912
git worktree add /tmp/cosyvoice-utilization-h100 FETCH_HEAD
cd /tmp/cosyvoice-utilization-h100
```

Use a fresh worktree path if that directory already exists. Do not apply a separate patch on top of the branch. Use the server's Python environment and verify imported package paths against the recorded source identities. No tar/zip transfer is needed.

Follow [the H100 runbook](diagnostics/README.md): establish the team's actual baseline, qualify annotations with small captures, and collect the full English SeedTTS baseline at **both concurrency 1 and 16**, independently for streaming and buffered output. Full A/B starts only after an optimization candidate exists. English is the only benchmark language in scope. Use profiling only for small diagnostic cohorts. Complete quality scoring remotely before returning plain results without WAVs under `artifacts/cosyvoice/` in the local Omni checkout.

Read [the open-PR overlap audit](reports/17_open_prs.md) before implementing another component. It records the inspected PR heads and overlaps; an open PR is not evidence of an optimization already present in the baseline. [Existing optimization assessment](plans/09_existing_optimizations.md) specifies which current paths need correctness and performance qualification before replacement or reuse.

## Diagnostic code boundary

Source commit `5ab17f8c9` adds optional NVTX metadata around speech request/HTTP output, stage request events, reference/embedding preparation, AR execution/sampling/token D2H, Flow, HiFT, waveform D2H, streaming cohort selection and waits. It does not intentionally change scheduler policy, model arithmetic, sampling, dtype, cache semantics, placement or transport.

Commit `a1bcd79f7` adds the four diagnostic tools and the c1/c16 H100 runbook. The remaining research snapshot is a separate commit so the runnable diagnostic slice can be reviewed independently.

Set `SGLANG_OMNI_PIPELINE_NVTX=1` before server startup for short captures. With it disabled, decorators return original functions and NVTX is not imported; explicit range/mark calls remain no-ops. Small host-call overhead still needs remote qualification. Disable the flag to stop annotations; revert the diagnostic source commit to restore the baseline code.

Static validation covers parsing, CLI help, coverage hashes/read ranges, formatting and Git checks. No local pytest, unit tests, model runs, benchmarks or profiling were performed. CUDA behavior, profiler interoperability, numerical identity and the utilization target remain H100 gates. The branch is a committed investigation and diagnostic starting point, not a measured 30% optimization result.
