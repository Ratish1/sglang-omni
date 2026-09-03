# 22. Run 5 and the full-corpus A/B of the cuDNN attention repair

Source: `artifacts/qwen3tts-cudnn-validation-20260903-slim.tar.gz`, extracted
to the scratchpad as `run5/`. Arm A is `15c4568bb`, arm B is `a4f3590b2`
(the repair). Run 5 is B plus the step ledger stack (`d6425827b` rebased,
local tip `b50df3e29`). All A/B arms ran on physical GPU 2 with the vocoder
in its own process, one server per arm, arms alternated (A then B at c1,
B then A at c16). Run 3 (doc 19 section 7) ran on GPU 3.

No PR exists for the repair on `sgl-project/sglang-omni` or on the fork
(searched open, closed and merged for `cudnn`, `enable_cudnn_sdp`,
`sdpa_kernel`, `predictor attention`, and by branch name, 2026-09-03).

## 1. What the repair did to the captures

From `run5/capture_c1.log` and `capture_c16.log` (phase instrument of doc
17):

| Window | Key | Ordinal in process | Total ms | Warmup 1 | Warmup 2 | Capture pass | Graph exit | Empty cache |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1 | 1 | 1 | 336.8 | 226.8 | 41.3 | 41.7 | 23.0 | 1.1 |
| c16 | 2 | 1 | 402.1 | 236.2 | 32.5 | 42.3 | 20.4 | 68.2 |
| c16 | 16 | 2 | 126.4 | 38.3 | 33.3 | 34.3 | 6.1 | 11.8 |
| c16 | 12 | 3 | 131.3 | 47.1 | 32.6 | 34.2 | 8.7 | 5.9 |
| c16 | 8 | 4 | 123.9 | 51.1 | 33.1 | 29.9 | 7.3 | 0.3 |
| c16 | 4 | 5 | 133.7 | 38.7 | 34.0 | 41.9 | 6.7 | 8.8 |
| c16 | 1 | 6 | 319.2 | 145.2 | 75.2 | 47.2 | 9.7 | 39.0 |

- The cuDNN plan ladder is gone: warmup 1 of a non-first key is 38 to 51
  ms against 32 to 34 ms for warmup 2 (run 3: 12 to 18 times warmup 2).
- The first capture of a process still carries about 190 ms of first use
  work in warmup 1 (227 and 236 ms against 41 and 33 ms). Run 4's trace
  had already shown this residual next to the plan builds (doc 19 section
  9). It is per process, and a startup capture absorbs it.
- `empty_cache` inside `torch.cuda.graph.__enter__` cost 68 ms for the
  first c16 capture and 39 ms for the last (676 MiB released), both inside
  serving steps. Ordinal 6 at c16 (bucket 1) ran with 4872 allocations in
  warmup 1 against 1497 for the others, so its warmup 1 was 145 ms.

## 2. Ledger comparability and deltas

Both ledger runs used the request profiler only (`profiler_start ...
torch=False` in both serve logs). Run 3 ran on GPU 3, run 5 on GPU 2, so
ledger to ledger deltas carry a device to device term of unknown size.
The A/B of section 4 has no such term.

Decode steps, p50, run 3 to run 5:

| Shape | cycle ms | host ms | gpu span ms | forward ms (backbone graph) | gpu span minus forward |
| --- | --- | --- | --- | --- | --- |
| c1 rows 1 (2433 and 2496 steps) | 7.525 to 7.581 (+0.7%) | 7.262 to 7.347 (+1.2%) | 7.143 to 7.247 (+1.4%) | 2.175 to 2.149 (-1.2%) | 4.968 to 5.098 (+0.13 ms, +2.6%) |
| c16 rows 16 (38 and 44 steps) | 8.502 to 8.773 (+3.2%) | 8.216 to 8.459 (+3.0%) | 8.056 to 8.276 (+2.7%) | 2.397 to 2.404 (+0.3%) | 5.659 to 5.872 (+0.21 ms, +3.8%) |

The backbone graph is unchanged or faster. The rest of the step, which is
the predictor replay, sampling and the collect, is slower by 0.13 ms per
step at one row and 0.21 ms at sixteen rows. The predictor graph replays
flash kernels in B and cuDNN kernels in A. Nothing else on that part of
the step changed between the arms.

## 3. Stage spans in the 50 sample ledger windows

The events file named `events_preprocessing_<pid>.jsonl` holds both the
preprocessing stage and the tts_engine stage of the pipeline process (102
`stage_dispatch` to `stage_complete` pairs for 51 requests). The
preprocessing pair is the first of the two per request.

| Span, c1 window p50 | Run 3 | Run 5 |
| --- | ---: | ---: |
| preprocessing `stage_dispatch` to `stage_complete` | 78.2 ms | 40.1 ms |
| tts_engine `scheduler_prefill_end` to `model_path_end` (generation) | 345.8 ms | 341.8 ms |
| vocoder `stage_dispatch` to `stage_complete` | 77.4 ms | 43.7 ms |
| coordinator `request_admission` to `terminal_response` | 521.6 ms | 441.2 ms |

Every one of the 50 requests in a window is a new reference and a new
output length for a fresh process, so run 3 paid a cuDNN plan build in the
speech tokenizer (preprocessing) and in the codec decoder (vocoder) for
nearly every request. These are per new shape costs, bounded per process by
the number of distinct lengths, and the A/B's 1088 requests per arm spend
most of the run past that bound. The corpus has 666 distinct references and
output lengths of 24 to 123 tokens.

## 4. Full-corpus A/B, paired by sample

`ab_paired.py` in the scratchpad, over `results.csv` and
`wer_results.csv` of the four arms.

| | c1, 1087 pairs (runaway excluded) | c16, 1088 pairs |
| --- | --- | --- |
| latency sum B/A | 1.0078 | 1.0114 |
| paired delta B minus A, median | +5.7 ms | +9.7 ms |
| B slower in | 596 of 1087 | 580 of 1088 |
| pairs with equal token count | 85, mean +6.5 ms, B slower in 74 | 72, mean +17.8 ms, B slower in 42 |
| latency fit a + b times tokens, A | 65.1 ms + 7.269 ms per token | 191.7 ms + 16.074 ms per token |
| latency fit, B | 59.9 ms + 7.411 ms per token | 157.0 ms + 16.970 ms per token |
| WER errors A / B over 11943 reference words | 116 / 123 | 120 / 136 |
| samples whose WER differs, B worse / A worse | 43, 26 / 17 | 46, 29 / 17 |

B was second at c1 and first at c16 and is slower in both orders, so run
order does not explain it. The per token slope is 2% higher at c1 and 6%
higher at c16, and the 74 of 85 equal token pairs at c1 is a sign test
far outside chance. Per request at c1 the delta is about 6 ms, which is 52
steps times the 0.13 ms of section 2.

Within one arm, A's error count moves by 4 between c1 and c16 and B's by
13. The arm to arm differences of 7 and 16 errors are of that size. The
readout's similarity numbers are 71.1655 against 71.3014 at c1 (1087
samples) and 71.2112 against 71.2138 at c16.

## 5. The runaway

`common_voice_en_19916471-common_voice_en_19916474`, arm B at c1:
2048 completion tokens, which is `max_new_tokens`, 163.84 s of audio,
latency 14.408 s, whisper text "We we we we we we made we made we made ...".
Arm A produced 42 tokens and 3.36 s for the same text. The 18 s of extra
wall time is the whole c1 QPS gap (499.1 s against 481.2 s of summed
latency). It also produced the 63.99 GiB similarity scorer allocation.

The benchmark ran without `--seed` (config `seed: None`), so each request
drew fresh seeds and the trajectory cannot be replayed from the artifacts.
One runaway in 2176 requests under flash numerics against zero in 2176
under cuDNN is not evidence either way. Tracker issue #1754 (T-PR17)
records 14 runaways in nine full-corpus runs under the FP8 predictor graph
and zero with the predictor graph off, so runaway counts are a quality
signal to read on every numerics change of this path.

## 6. Verdict on the repair

The repair removed per process, per shape costs and added a per step cost:

- Removed, per process: about 350 ms per predictor bucket (six buckets at
  the defaults), 38 ms per new reference length in the tokenizer and 34
  ms per new output length in the vocoder (section 3 medians).
- Added, per step, for the life of the process: 0.13 ms at one row and
  0.21 ms at sixteen rows (section 2), which the paired A/B shows as +0.8%
  to +1.4% per request at c1 and +1.1% at c16.

The first 50 requests of a fresh process are about 80 ms (15%) faster per
request in B (section 3, coordinator span). At the steady state B is about
6 ms slower per request at c1. On those two numbers the arms cross after
roughly 700 requests per process, and a serving process lives far longer.
Under the standing rule that a runtime change carries no steady state
regression, the repair is not shippable as it stands, and no PR should be
opened for it.

## 7. Consequences for doc 21, proposed

- Keep cuDNN attention on. Drop `disable_cudnn_attention()` from the
  stage factories, or keep the helper unused until a measured need.
- Slice A's startup capture pre-builds the cuDNN plans for every bucket of
  the default signature in its warmups before the stage is ready, so the
  serving step stall of the default path disappears without changing the
  replayed kernel. Lazy captures of non default signatures keep today's
  bounded stall (about 350 ms per new key) because the cuDNN plan cache is
  thread local (doc 20 section 10) and the startup capture runs on the
  factory thread. Record that as a known cost, not a regression.
- Slice B's replay time gate compares the sglang kernel against the cuDNN
  replay, not against flash. The kernel's two launches per attention start
  behind the fused cuDNN kernel, so the gate is expected to close Slice B
  unless the measurement says otherwise. T22, a single launch kernel over
  the fixed cache, is the path if cuDNN must leave the predictor.
- The deterministic mode repair (fused addmm off under batch invariant
  mode) is independent of the attention backend and stays in Slice A.
- The tokenizer and vocoder per length plan builds are a warm up cost of
  bounded size. A startup sweep over a length ladder for both is a separate
  small item (T40), decided after Slice A lands.
- T39: run the A/B with a fixed `--seed` so the two arms sample the same
  trajectories wherever their numerics agree and a runaway can be
  replayed. Add to doc 15.

## 8. Other observations in the tarball

- Two startups before run 5 died with the vocoder process at exit code -9
  while GPUs 0 and 1 held external allocations (readout header, the
  `run5-startup-killed-*` logs). Not a code path of ours.
- Unit tests: 342 passed on the box, 51 accelerator tests passed, the
  kernel name test included.
- The benchmark ran non streaming (`stream False`), so there is no first
  audio latency in the A/B.
