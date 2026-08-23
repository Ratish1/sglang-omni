# PyTorch optimization workflow

This directory is the local control plane for performance work. Profilers,
sync-detector hooks, trace analyzers, benchmark reports, and raw artifacts stay
here or on a local profiling branch. They are not copied into implementation
branches or production PRs.

## Repository layout

- `registry.md`: one row per independently shippable change.
- `models/<model>.md`: current mechanics, ownership decisions, dependencies,
  and unresolved work for one model family.
- `artifacts/<model>/baseline/<sha>/<contract-id>/`: reusable baseline results.
- `artifacts/<model>/candidates/<branch>/<sha>/<contract-id>/`: candidate results.
- `artifacts/<model>/profiles/<sha>/<run-id>/`: local detector and Kineto output.

Do not duplicate raw artifacts. Keep the original archive plus extracted
machine-readable summaries and checksums in one run directory; link that path
from the registry or model sheet.

## Change lifecycle

1. **Inventory locally.** Run `torch.cuda.set_sync_debug_mode("warn")` for
   source discovery and a separate clean Kineto capture for timing and causal
   attribution. Use `"error"` only for bounded probes.
2. **Prove ownership.** Record the tensor shape, producer, consumer, stream,
   source lifetime, serving variants, and semantic owner before choosing a
   replacement.
3. **Choose one mechanism.** A work item must remove one synchronization owner
   or correct one semantic owner. Split unrelated files and mechanisms.
4. **Create a clean worktree.** Branch from the exact baseline commit. Reapply
   only production code; no profiler ranges, analyzers, task files, or benchmark
   arguments.
5. **Review mechanically.** Inspect the full diff and the surrounding call
   chain. Check CPU/non-CUDA fallback, dtype/device/shape parity, buffer lifetime,
   stream ordering, graph capture, retraction, batching, and model variants.
6. **Run local checks.** Compile/import and existing focused tests. Add tests
   only when the production contract cannot be covered by existing tests.
7. **Run one remote qualification.** First prove the selected mechanism is
   exercised and gone in a bounded detector/trace run. Then run the stored
   candidate benchmark contract.
8. **Classify honestly.** A mechanical cleanup may ship without a measurable
   end-to-end speedup, but the PR must not claim a speedup unless the result is
   larger than run variance. At most one reversed/confirmation run is used for
   a borderline or contradictory result.
9. **Land independently.** Do not stack branches unless the registry declares
   an explicit semantic dependency.

## Reusable Qwen3-TTS baseline contract

The full English SeedTTS set is generated at concurrency 1, 8, and 16. WER is
computed once from the concurrency-16 audio. All three generations use the same
fixed explicit seed and the canonical public sampling defaults; do not add
penalty or token-limit overrides merely for the benchmark.

A baseline may be reused only when all of these match the candidate run:

- exact baseline commit and candidate ancestry;
- physical GPU and no competing workload;
- container image, Python environment, editable-install target, Torch, and CUDA;
- model snapshot, dataset revision, and ASR snapshot;
- server config and all effective server arguments;
- sample order, language, fixed seed, reference-cache state, and warm-up;
- CUDA-graph keys and profiler disabled for timing.

If any item changes, the baseline is stale. Store provenance, effective command,
server log, per-request results, speed summary, WER summary, capped/finish-reason
tail report, and checksums with the run. Exact WAV equality is not an acceptance
gate until the server has a proven fresh-process deterministic contract.

The canonical run uses the repository benchmark without sampling overrides:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --model "$QWEN3_TTS_MODEL" \
  --server-config examples/configs/qwen3_tts_0_6b.yaml \
  --meta "$SEEDTTS_META" \
  --lang en \
  --max-samples 1088 \
  --concurrencies 1,8,16 \
  --seed 20260823 \
  --output-dir "$RUN_ROOT"
```

After the managed TTS server exits, score only the c16 audio:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_tts_seedtts \
  --transcribe-only \
  --model "$QWEN3_TTS_MODEL" \
  --asr-model-path "$QWEN3_ASR_MODEL" \
  --meta "$SEEDTTS_META" \
  --lang en \
  --max-samples 1088 \
  --output-dir "$RUN_ROOT/c16"
```

`$QWEN3_TTS_MODEL`, `$QWEN3_ASR_MODEL`, and `$SEEDTTS_META` should resolve to
the already cached immutable model/dataset revisions. Do not pass
`--repetition-penalty`, temperature/top-k/top-p, or an explicit token limit.

## Review checklist

- [ ] One semantic/mechanical owner and one production diff.
- [ ] Exact immutable baseline SHA recorded.
- [ ] No local profiling or task code in the branch diff.
- [ ] Tensor values, shape, dtype, device, and bounds are unchanged.
- [ ] Pinned source is immutable until its async copy completes.
- [ ] Same-stream ordering is proved, or cross-stream events are explicit.
- [ ] No pageable transfer, scalar read, or replacement stream/device wait.
- [ ] Batch grow/shrink, request reuse, retract/re-prefill, and graph capture are
      either exercised or documented as out of scope.
- [ ] Shared checkpoint variants are source-audited; performance claims name the
      checkpoint actually measured.
- [ ] Existing focused tests and formatting pass.
- [ ] Remote c1/c8/c16 + c16 WER artifacts are attached or linked.
