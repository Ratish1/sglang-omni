# PR #2909 review

PR: https://github.com/sgl-project/sglang-omni/pull/2909

Reviewed head: cefab51fc709735b12b3d25793229a8fed89722a. Base: d07df0b4339cf6efcb6cec1d0da7e7ecd4b80ddf. Review date: 2026-10-18. Author worktree: /Users/ratish/sglang-omni/.worktrees/code2wav-stream-state.

## 1. Summary and technical background

The old streaming path re-decodes each chunk with at most 25 frames of left context. This both repeats convolution work and cuts the model's 72-frame local attention. Keeping each layer's KV, causal-convolution input history and transposed-convolution overlap is the principled fix. Increasing an arbitrary overlap length would continue doing repeated work and would not explicitly preserve every layer's state.

The new state is model-local and bounded. Slot capacity follows the talker's configured request cap and replica count. Ring capacity follows the attention window and maximum new chunk. First/next graph variants express actual cold/warm state differences. No new hardware, duration or dataset heuristic is introduced.

The retained-state mechanics passed review and the independent CUDA checks. Two focused changes remain: stop capturing stateful graph batches that exceed the pool's capacity, and add a real multi-row graph ownership test. There is also a small optional annotation cleanup. No audio correctness defect was found.

## 2. Code quality and correctness

### F1 — [P1-PERF] Bound stateful graph capture by the codec pool's capacity

Location: sglang_omni/models/qwen3_omni/components/code2wav_scheduler.py:1299 and the incremental_graph_keys call in the same factory.

The stream-slot count follows the talker's cap, but the scratch-slot count and stateful graph row counts still use max_replay_rows alone. A valid talker_ar.engine.max_running_requests=1 configuration therefore has one live codec slot but captures first/next graphs for rows 1–8. Fourteen of these graphs cannot run: a stateful cohort needs a distinct allocated slot per row, and slotless streams are grouped separately.

Independent trained H200 check, one stream slot, with the existing factory's row cap set to eight versus one:

| Resource | Capture up to eight rows | Capture up to one row |
|---|---|---|
| Published graphs | 60 | 46 |
| Unreachable stateful graphs | 14 | 0 |
| Capture scratch slots | 8 | 1 |
| Graph footprint | 546 MiB | 172 MiB |
| Codec pool footprint | 25.19 MiB | 5.60 MiB |

Both variants served the same two-window stream with exactly the same audio SHA-256, two graph replays and no fallbacks. The capture resource saving is approximately 393.6 MiB in this check. It is not a claim about an end-to-end throughput regression. The default 32-slot configuration is unaffected.

Fix: derive the incremental capture row cap as min(max_replay_rows, codec_state_slots) when the incremental codec is enabled; use that bound for both the scratch-slot allocation and incremental_graph_keys. When graphs are disabled, the scratch-slot count can be zero. Keep the legacy stateless path's separate row policy. No new heuristic or validation layer is needed.

Suggested direct comment:

> Cap the stateful graph row counts and scratch slots at the codec pool's slot count. With a talker cap of one, this still captures two- through eight-row first/next graphs, but only one row can own a slot. Those 14 graphs cannot run; limiting capture to one row saved about 394 MiB here with identical audio. Allocate no capture scratch slots when graphs are off.

### F2 — [P4-PROCESS] Exercise an actual multi-row incremental graph in the scheduler test

Location: tests/unit_test/qwen3_omni/test_code2wav.py:1338 and :1361.

The test driver delivers each frame through deliver_code2wav_chunk, which drains ready work immediately. When the first request reaches its threshold, its decode runs before the second request receives its threshold frame. Both streams therefore exercise one-row graphs. Setting codec_state_slots=2 does not make this a multi-row graph test.

Why this matters: the PR adds runtime per-row slot-index uploads and per-row persistent state. A row-to-slot mapping error in a two-row next graph can corrupt audio without affecting any one-row replay or the direct-codec mixed-position test.

Proof on H200: reversing slot IDs only for a multi-row next graph leaves both variants of the PR's existing scheduler test passing. A control test that ingests both streams before run_ready_step passes on the real code and fails under that mutation: 90/123 samples differ beyond tolerance in the first compared waveform. This is a derived state-ownership property under the SGLang unit-test admission rule, not a mocked happy-path assertion.

Fix: add the coalesced case to the existing test_code2wav.py module. Deliver both streams through handle_stream_chunk before each ready step, ensure the step actually groups two rows, run multiple next windows through a ring wrap, finish the tails, and compare each request's audio to the direct codec. The standalone review probe demonstrates the case; it is not a request for a new production test file.

Suggested direct comment:

> Feed both streams before each ready step so this test runs a two-row first/next graph. deliver_code2wav_chunk drains the first stream before the second reaches its threshold, so these replays all have one row. Compare each stream's audio after several next windows and the final tail to catch a row-to-slot mapping error.

### F3 — [P3-STYLE] Update the graph statistics type for the new decode field

Location: sglang_omni/models/qwen3_omni/components/code2wav_cuda_graph.py:71, stats at :960.

Code2WavGraphStats.graph_contract still declares every key value as int, but stats now includes decode with string values window/first/next. Runtime JSON is valid; the declared contract is inaccurate.

Fix: describe the key with a concrete TypedDict containing batch_size, frames and the decode Literal. The scheduler already defines the same field shape as GraphKeyMetadata; if sharing it, its defining module must sit below the scheduler to avoid a circular import. Do not add a generic schema layer for this.

Suggested direct comment:

> Update graph_contract's key type for decode. It is still dict[str, int], but decode now contains window/first/next strings.

F3 is optional cleanup. F1 is a conditional startup/resource issue; F2 is a regression-test gap. None of the findings alleges observed production audio corruption.

## 3. Goal completeness

The implementation covers state acquisition, normal completion, short streams, final partial windows, pool exhaustion, graph-key misses, disabled graphs, abort and shutdown. Stateful eager fallback continues from the same slot instead of restarting the decoder.

The independent H200 checks include the PR suite, a mutation-verified multi-row graph test, real queued-state-write abort/reuse, cohort failure after real state writes, and trained-checkpoint comparison against HF fp32. Detailed final results and serving checks are recorded in verification.md.

The PR reports four paired, card-swapped full-corpus c16/c64 runs. Those are author results, not independently repeated end-to-end throughput measurements in this review. The quality table reports WER only below 50%; that row alone cannot establish full-corpus WER or the frequency of garbled outliers. The numerical checks establish decoder fidelity, not a separate corpus WER/SIM verdict.

## 4. Coherence and system fit

Keeping the pool and convolution state code under Qwen3-Omni is appropriate. The state layout and traversal are properties of this exact decoder architecture. SGLang's Mamba depthwise kernels and AR page allocator are not replacements for the codec's dense/dilated/transposed convolutions and combined state lifetime.

The reused SGLang rotary kernel, request-indexed bounded state, retained causal tails and first/later distinction follow the related SGLang mechanisms. The existing layer math is shared between whole-window and incremental forwards instead of being copied into a second transformer implementation.

The main hot-path tradeoff is explicit: the first window pays state-write cost, while later windows stop re-decoding old input. The three faster kernel follow-ups described by the author are outside this PR's prerequisite state contract; their absence is not a blocker here.

Read [mechanics.md](mechanics.md) for the file-by-file decomposition, ring and convolution derivations, lifecycle analysis and SGLang source comparison. Read [verification.md](verification.md) for evidence and reproducible commands. No GitHub review was staged or posted, and the author branch was not modified.
