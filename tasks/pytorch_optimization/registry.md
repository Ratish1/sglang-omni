# Optimization registry

The local profiling branch is `perf/qwen3-tts-hidden-h2d-sync-v2`. It is an
evidence branch, not a PR branch. New implementation worktrees are cut from the
exact comparison baseline and contain no commits from the profiling branch.

| ID | Repository | Worktree / branch | Scope | Evidence | State |
|---|---|---|---|---|---|
| Q3TTS-H2D-01 | SGLang-Omni | `.worktrees/qwen3-tts-sampling-metadata-h2d` / `perf/qwen3-tts-sampling-metadata-h2d` | Semantic/subtalker metadata: pinned CPU staging to persistent CUDA buffers | 1,902 selected blocking copies removed in the first trace; later clean ranges had zero waits | Candidate `a06f818c`, based on `91d4359f`, pushed for H100 qualification |
| Q3TTS-H2D-02 | SGLang-Omni | not created / `perf/qwen3-tts-model-input-h2d` | Speaker mel, cached speaker embedding, prompt token rows, reference code | Mechanically clean on 0.6B | Ready after H2D-01 |
| Q3TTS-H2D-03 | SGLang-Omni | not created / `perf/qwen3-tts-text-tokenizer-h2d` | Preserve the qwen-tts processor and pin only tokenizer IDs | 128/128 range calls exercised, zero waits | Ready after design review |
| Q3TTS-REP-01 | SGLang | not created / `fix/rebuild-penalizers-from-output-history` | Restore generated-output penalizer state when a request is re-prefilled | Ownership bug source-proved | Design required |
| Q3TTS-REP-02 | SGLang-Omni | not created / `fix/qwen3-tts-single-repetition-owner` | Remove Qwen repetition transform; retain codec suppression only | Duplicate transform source-proved | Blocked by REP-01 |
| Q3TTS-VOC-01 | SGLang-Omni | not created | Bypass tokenizer GPU-CPU-GPU decode and publish waveform asynchronously | Trace-attributed | Future design |
| Q3TTS-REF-01 | SGLang-Omni | not created | Reference tokenizer length/control and cache publication | Trace-attributed | Future design |

Shared Qwen3-TTS implementation covers the supported 0.6B and 1.7B checkpoints.
Current H100 evidence is for `Qwen/Qwen3-TTS-12Hz-0.6B-Base`; no 1.7B speedup
claim is made until that checkpoint is measured.

Current benchmark contract ID:
`q3tts-h100-seedtts-en1088-c1-c8-c16-seed20260823-v1`.
