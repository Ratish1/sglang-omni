# Performance backlog

Compiled 2026-09-04 from the numerics docs (`tasks/qwen3_omni_0518_numerics`,
docs 00 to 24), the sampler survey (`tasks/sampler_reuse_survey`), the
`perf/*` and `debug/*` branches against `upstream/main` `f62fd76cc`, and
the author's PR list on `sgl-project/sglang-omni`. Every open performance
item lives here with its source, so the source doc keeps the evidence and
this file keeps the order. No `perf/*` branch carrying open work has a PR.
Plans for items in this list are written as numbered docs in this folder.

## 1. Order

1. Close the Qwen3-TTS predictor startup capture (A.1). The branch is at
   `3d020c85f`, the same change rebased onto upstream `556166dde` and
   validated as `d7e34a16c` on the previous base: 348 unit tests, c1 byte
   identical with the order swapped, c16 first batch 2.5 s per request
   faster and steady state flat. A main warmup 1 control still captured
   five buckets inside serving. PR #1947 open, body in
   `pr_qwen3_tts_predictor_startup_capture.md`. Owed: CI on the rebased
   head.
2. Observed KV reservation on the Qwen3-Omni talker (A.3). Implemented on
   `perf/observed-kv-reservation` at `81a87e474`, measured at +25% qps at
   c16 and +43% at c32 on the bf16 colocated profile (doc 08 section 1).
   Owed before the PR: the bf16 slim A/B rerun with #1910 in both arms
   (doc 12 section 5), voice clone bf16 at c1, c16 and c32 with WER and
   UTMOS, and the MiniMax cookbook A/B (doc 09 section 7).
3. T22, the predictor chain (A.1). Census and timeline done on H100
   (`02_qwen3_tts_predictor_chain_census.md`): 1371 kernels per replay
   bound by kernel count, and a 1.1 to 1.4 ms host tail per step. Plan in
   `03_qwen3_tts_predictor_chain_plan.md`. S1 implemented and reviewed
   on `perf/qwen3-tts-predictor-chain` at `1d047d541` (13 commits, plan
   section 11, no chosen number left in the series), untested on the box,
   expected bit identical, about 6% of the step. Tool `scripts/perfkit.py`,
   branch `perf/qwen3-tts-profiling` at `93684aa0b` (ledger plus S1) for
   the census diff.
4. T40, the speech tokenizer and codec decoder cuDNN plan per new length
   (A.1). 38 ms and 34 ms per new length at the run 3 medians (doc 22
   section 6). Measured at 68 to 81 ms per new length per side at c1
   (plan section 2). Plan: `01_qwen3_tts_length_plan_builds_plan.md`,
   Conditional on its section 9.
5. A4, one greedy warmup request per stage before readiness, every model
   (doc 19 section 9.5).
6. T18, rebase `perf/step-ledger` onto upstream main and rerun the doc 15
   fleet table on a quiet box with MiniMax (doc 18 section 5 item 1).
7. E2 and T23, the talker per step host syncs and the decode buffer
   rebuilds (A.3). Needs the fixed per sample seed in `run_bench.py`
   first (doc 05 section 7.5).
8. T21, the code2wav final window graphs, after #1758 lands (doc 18
   section 5 item 4).
9. T19, prefill graph buckets 1 and 2 on the Higgs and Qwen3-ASR builders,
   unblocked now that #1915 landed (doc 18 section 5 item 6).
10. T24, MOSS-TTS-Local and Delay uploads and the eager hash, after #1792
   lands (doc 18 section 5 item 7).
11. Higgs codec graphs, bucketed capture over the reachable window shapes
    (doc 18 section 5 item 8).
12. Sampler candidates, after the per model sampler profile that doc 11
    section 3 makes the gate.
13. PRs A to D on the Qwen3-Omni stage path (doc 05), each behind a
    decision the user owns (section 4 below).
14. Reviews to post on other people's PRs (doc 18 section 5 item 9).

## 2. Open items by area

### A.1 Qwen3-TTS predictor, cuDNN plans, capture

| Item | Source | Status | Blocker or hold |
| --- | --- | --- | --- |
| Slice A, startup capture and capture hygiene | doc 21 section 7.1, docs 23 and 24 | implemented, validated twice, no PR, branch `perf/qwen3-tts-predictor-startup-capture` at `4f43776fe` | the three runs of doc 24 section 7 |
| Slice B, predictor attention on sglang's `decode_attention_fwd` | doc 21 section 7.2, gate G1 in 8.3 | not written | doc 22 section 7 expects the gate to close it |
| T22, one launch masked attention over the fixed 17 slot cache | doc 17 section 7, doc 21 section 6.7 | deferred, not rejected | only if Slice B closes on the gate |
| T40, tokenizer and vocoder cuDNN plan per new length | doc 21 section 10, doc 22 sections 3 and 6 | open, decided after Slice A | none |
| T29, PyTorch issue on the cuDNN SDPA plan per batch and key length | doc 20 section 10 | open, evidence on `perf/qwen3-tts-cudnn-attention` | none |
| T30 and A5, the cuDNN policy on the Qwen3-Omni talker predictor and the MOSS-TTS attentions | doc 20 section 10, doc 19 section 9.1 | open | one trace per model first (T26), and no blanket cuDNN disable after doc 22 |
| T33, cold Triton cache cost of the first fused kernel launch | doc 21 section 10 | open | box time |
| T34, flash split-kv heuristic at `seqlen_q = 1` | doc 21 section 10 | open, informational | none |
| T35, Triton 3.7.1 `tl.dot` minimum tile on the box | doc 21 section 10 | open, informational | box time |
| T36, Slice B unit tests on a non sm90 CUDA box | doc 21 section 10 | open | hardware |
| T37, tracker note on #1936, fault 2 open | doc 21 section 10 | open | none |
| T38, serialize a lazy vocoder capture against the predictor capture | doc 21 section 10 | conditional | only if #1855 lands |
| T39, fixed `--seed` in the doc 15 runbook | doc 21 section 10 | runs 6 and 7 used it, the runbook edit is not recorded | none |
| Predictor graph key cap of 32 reachable by a benign mix, fallback counter never read | doc 18 section 4.1 | named, not scheduled | none |
| Chunked prefill emits a spurious code frame | doc 18 section 4.1 | named, not scheduled, chunking is disabled for the talker and zonos2 for this reason | none |
| Prefill runs the predictor on a cache hit | doc 18 section 2 | open | none |
| Vocoder stage capacity past c32 | doc 18 section 2 | open | none |
| cuDNN attention off in the Qwen3-TTS stage processes | doc 20 sections 6 and 8, verdict doc 22 section 6 | held on a measured regression, 0.13 ms per step at one row, no PR, branch `perf/qwen3-tts-cudnn-attention` at `a4f3590b2` | stays held |

### A.2 Qwen3-TTS host and H2D synchronisation

Source: the ledger on `perf/qwen3-tts-hidden-h2d-sync-v2` at
`tasks/qwen3_tts_performance_pr_ledger_20260823.md`.

| Item | Status | Blocker or hold |
| --- | --- | --- |
| H2D sync bundle, to be split into four PRs | held, 16 commits on `perf/qwen3-tts-hidden-h2d-sync-v2` at `e16350f70`, no PR | no repeatable end to end speedup established |
| Sampling metadata H2D async | `perf/qwen3-tts-sampling-metadata-h2d` at `a06f818c3`, no PR | needs an isolating A/B |
| Text token H2D async | `perf/qwen3-tts-text-tokenizer-h2d` at `26fb07f45`, local only | needs its own A/B |
| Next synchronization owners (vocoder decode, reference encode, cache key, speaker artifact cache, final code D2H, repetition setup) | designs sketched, prototype `debug/qwen3-tts-vocoder-direct-decode` at `175a8a6c2` | each needs a design |

### A.3 Qwen3-Omni talker

| Item | Source | Status | Blocker or hold |
| --- | --- | --- | --- |
| T15 and T11, observed KV reservation | docs 08, 09 section 7, 12 section 5, 16 section 6 | implemented on `perf/observed-kv-reservation` at `81a87e474`, no PR | the runs in section 1 item 2 |
| E2, per step host syncs (multimodal inputs skip, device bool, broadcast suppress mask, pinned staging) | doc 05 section 7.2, doc 06 section 5.2, doc 13 section 4 | partly on `perf/talker-step-syncs` at `291c73f33`, no remote | proof run not done, `run_bench.py` seed tooling first |
| E3, rid keyed decode buffer reuse | doc 05 section 7.2, doc 06 section 5.2 | partly on the same branch | user decision on penalty ownership (doc 05 section 9) |
| E4, request build off the scheduler thread, eager prefill host time | doc 05 section 7.3 | not started | after E1 and E2 are measured |
| E5, talker overlap | doc 05 section 7.4, PRs #1204 and #1320 | held on a measured TTFA regression | after E2 and E3 |
| T23, talker on its own GPU, then `prepare_decode_buffers` rebuilds | doc 17 section 7, doc 18 sections 4.3 and 5 | open | the isolated GPU measurement first |
| Talker predictor graph pads a one row prefill to 32 rows | doc 18 section 4.2 | open, folded into T23 | none |
| T16, the talker's per row device cost | doc 16 section 6 | open | none |
| Fused residual add with RMSNorm, fused qk norm with rope | doc 13 section 3 | open, about 240 kernels per step | after E2 |
| Predictor cache copy removal via a pool layout | doc 13 section 3 | open | after the two fusions |
| Split-K GEMM reduces and the small M GEMM ceiling | doc 13 section 3 | open, a kernel project with its own measurement | the ceiling measured first |
| T26, the 200 ms first shape prefills on the talker and MOSS-TTS Delay | doc 17 section 7, doc 19 section 9.1 | open, candidate named (cuDNN plan builds) | one trace per model |

### A.4 Cross model scheduler, ledger, other stages

| Item | Source | Status | Blocker or hold |
| --- | --- | --- | --- |
| T18, rebase `perf/step-ledger` and rerun doc 15 | doc 17 section 7, doc 18 section 5 | open, branch at `d6425827b`, no PR by design | box time |
| T19, prefill buckets 1 and 2 on Higgs and Qwen3-ASR | doc 17 section 7, doc 18 section 5 | open, one line on the shared ladder | none, #1915 landed |
| T21, code2wav tail window graphs | doc 17 section 7, doc 18 sections 4.4 and 5 | open | after #1758 |
| T24, MOSS-TTS-Local and Delay uploads and hash | doc 17 section 7, doc 18 sections 4.6 and 5 | open | after #1792 |
| T25, Qwen3-ASR eager prefills | doc 17 section 7 | open | none |
| T12, derived gate A/B per model, Qwen3-ASR first | doc 14 section 9, doc 16 | open | none |
| T17, request build placement for MOSS-TD and the talker | doc 16 section 6 | open | none |
| T27, MiniMax AR post decode behind a device event | doc 17 section 7 | open | verdict on the decode rows and the cookbook checksums |
| T28, MiniMax acoustic stage A/B over its knobs | doc 17 sections 7 and 8 | open | foreground launch under a multiplexer until the exit -9 is understood |
| Higgs codec graphs, bucketed capture | doc 18 sections 4.5 and 5 | open | none |
| MiniMax skip empty admission scan | `perf/minimax-music3-empty-queue-admission` at `7c0c2c2ca` | open, mixed into a correctness branch | separate from #1559 |
| T2 to T10, T13, T14 from the runtime seams doc | doc 14 section 9 | open | T5 waits on T18 |
| Reviews to post | doc 18 section 5 item 9 | open | none |

### A.5 Qwen3-Omni stage path, doc 05 PRs A to D

| Item | Status | Blocker or hold |
| --- | --- | --- |
| PR A, encoder output cache off the request path | open, `perf/encoder-cache-async` is empty | user decision on the pinned budget (doc 05 section 9) |
| PR B, preprocessing process replicas | open | placement rule rejects GPU less replicas (doc 06 section 6), user decision on the count |
| PR C, thinker synchronous step host work | open | none |
| PR D, thinker lookahead for audio output requests | open | after the note on #1258 |

### A.6 Sampler

| Item | Source | Status | Blocker or hold |
| --- | --- | --- | --- |
| Sampler share of the decode step per model | doc 11 section 3 | open, the gate for the rest | none |
| Port the #816 recipe to zonos2 | sampler survey section 3 | open | min_p and the penalty order |
| Port the #816 recipe to the Qwen3-TTS eager sub-talker branch | sampler survey section 3 | open | equivalence sweep |
| Talker renorm with sgl_kernel at the model call site | doc 11 section 2 | open | the profile |
| Seeded request policy if `sampling_backend` leaves pytorch | doc 11 section 2 | open | a policy decision |

### A.7 CI, gates, stack level

| Item | Source | Status | Blocker or hold |
| --- | --- | --- | --- |
| Recalibrate the stage 5, 8 and 9 gates | doc 00 section 5 | open | post bump per sample data |
| Logprob margins as a drift gate | doc 00 sections 5 and 7 | plumbing on the analysis branch, first H100 execution owed | box time |
| Tuned Triton fused MoE config, E=128 N=768 | doc 00 section 5, doc 04 section 3 | held by decision (doc 05 header) | stays held |
| FlashInfer autotune cache for `trtllm::fused_moe` | doc 00 section 5 | open | none |
| Stage 10 WER gate | doc 00 section 5 | open | sampling on ten clips |
| G, CI JIT cache mount or warmup set | doc 04 section 3, doc 05 header | research item | CI findings |
| Prefill FA3 per launch cost | doc 04 section 4 | open, small | a fixed shape pair |

## 3. Done, merged, superseded, abandoned

- T20 superseded by doc 19. Doc 19 A1 (cuDNN out of the predictor)
  rejected on measurement (doc 22 section 6). A2 and A3 delivered by
  Slice A. The first Slice A cut `perf/qwen3-tts-predictor-capture` at
  `bae990b05` superseded by the rebuild. Fault 1 of #1936 closed by Slice A.
- The small-k sampler clamp divergence fixed on main by #1641. #1751
  closed. #1750 merged (repetition penalty once).
- E1 dropped (doc 05 section 7.1), `perf/talker-admission-cap` dead.
  #1910 merged (talker KV pool from 20 layers).
  `perf/scheduler-observed-reservation` superseded by
  `perf/observed-kv-reservation`. Doc 08 section 5 items closed by the
  clip fix bundle. #1454 and #832 already on main through #1509. MOSS-TTS
  Delay heads done by #1665.
- Empty branches, nothing to carry: `perf/encoder-cache-async`,
  `perf/talker-lookahead`, `perf/qwen3-asr-hidden-h2d-sync`,
  `perf/qwen3-tts-async-waveform-publication`, `perf/higgs-next`,
  `perf/qwen-omni`, `debug/pre-admission-readiness`.
- `perf/qwen3-tts-hidden-h2d-sync` superseded by the v2 branch.
- MOSS branches landed as #773, #810, #822, #874, #886.
- #642 superseded by the 0.5.18 upgrade, to be closed. The broad sampler
  refactor abandoned (sampler survey conclusion).

## 4. Decisions the user owns

- E3 penalty ownership on Qwen3-TTS, device side mask or synchronous
  penalty requests (doc 05 section 9).
- PR A pinned budget for the encoder caches (doc 05 section 9).
- PR B replica count and the placement rule (doc 05 section 9, doc 06
  section 6).

## 5. Status to establish

- Stage 8 talker trace on the current image (doc 04 section 4 says the
  file was not written). Superseded by doc 06 section 3 or still owed.
- Stage 10 with `NCCL_NVLS_ENABLE=0` as #1811 did (doc 04 section 4).
- `perf/moss-tts-nonstream-rtf` (42 commits, no PR),
  `perf/moss-vocoder-cuda-graph-integration`,
  `perf/moss-vocoder-monkey-patch`, `perf/moss-local-feedback-fastpath`,
  `perf/moss-local-frame-decode-compile`,
  `perf/issue-752-moss-tts-compile-investigation`: diff each against
  upstream main for anything not in #822 and #886, and read issue #752.
- `debug/qwen3-tts-vocoder-direct-decode`: read the persisted
  differential result for a verdict.
- E2a on `perf/talker-step-syncs` against what merged since, after a
  rebase.
- `analysis/moss-similarity-bisect`, a worktree with no commits.
- Doc 18 section 2 rows that depend on #1346, #1792, #1809, #1758 and
  #1912, re read against today's PR list.
