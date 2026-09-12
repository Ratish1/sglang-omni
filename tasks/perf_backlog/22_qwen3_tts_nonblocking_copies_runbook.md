# Runbook for slices C and B of plan 18, the two blocking copies (plan 21)

Branch `perf/qwen3-tts-nonblocking-copies`, head `83aa2c84c`, two commits stacked on
`perf/qwen3-tts-stage-ids-early` at `04b62c255` (PR #2123). A is `04b62c255`, whose boots are
readout 20's B boots plus E3, so A runs nothing new. B is `83aa2c84c`.

- `f6f22323c` restages the six sampling buffers from pinned host memory with non blocking
  copies, two slots with an event each (sglang_model.py).
- `83aa2c84c` copies a finished request's codes into pinned memory without waiting, leaves
  the event on the request data, the scheduler forwards it as result metadata and the stage
  runtime waits it off the scheduler thread before routing (request_builders.py, types.py,
  omni_scheduler.py, runtime.py).

Both are bit exact by construction: the same values reach the same device rows at the same
stream position, and the same CPU tensor reaches every reader after the same completion.

Protocol of 2026-09-12: four boots on B, the census boot with stacks, a seeded c1 boot, a c16
boot and the streaming pair. No c16 quality comparison.

Host load (readout 20 section 8): GPUs 0 to 3 share NUMA node 0, so a tenant on GPU 1, 2 or
3 slows every host bound step on whichever arm runs under it. Record GPUs 1 to 3 before each
boot. Tenants do not block the run: a pair is valid when both arms ran under the same load.
So when GPUs 1 to 3 are loaded, run A's c16 boot and A's three streaming passes in the same
session (sections 4 and 5) instead of reusing the follow up's A numbers; the census, the
identity pass and the c1 speed do not need it.

## 1. Suites on B

```bash
git rev-parse HEAD
pytest tests/unit_test/qwen3_tts/test_sampling_buffer_staging.py tests/unit_test/qwen3_tts/test_finish_payload_staging.py -v
pytest tests/unit_test/pipeline/test_scheduler.py -q -k "readiness or terminal or stream_output"
pytest tests/unit_test/pipeline/test_stage_streaming.py -q -k "readiness or terminal"
pytest tests/unit_test/qwen3_tts -q
```

The accelerator tests in the two new files fail on A by construction: a restage and a finish
return while the stream sleeps, and a slot's first copy has landed before the slot is reused.

## 2. Census and E2 on B, one boot

Server with `SGLANG_TORCH_PROFILER_WITH_STACK=1`, default config, windows of 12 requests at c1
and 192 at c16 as runbook 15 section 7, `perfkit.py` from the analysis branch at `0c9940304`
or later (the row label fix). Then at c1 rows 1 and at c16 rows 16 and rows 8:

```bash
python perfkit.py steps $OUT/b_c16/tts_engine.pkl | tee $OUT/b_c16/steps.md
python perfkit.py census $OUT/b_c16/tts_engine.pkl --rows 16 --json $OUT/b_c16/census_rows16.json | tee $OUT/b_c16/census_rows16.md
python perfkit.py hosttail $OUT/b_c16/tts_engine.pkl --rows 8 --top 40 --json $OUT/b_c16/hosttail_rows8.json | tee $OUT/b_c16/hosttail_rows8.md
python perfkit.py hosttail $OUT/b_c16/tts_engine.pkl --rows 16 --top 40 --json $OUT/b_c16/hosttail_rows16.json | tee $OUT/b_c16/hosttail_rows16.md
python perfkit.py timeline $OUT/b_c16/tts_engine.pkl --rows 8 | tee $OUT/b_c16/timeline_rows8.md
```

Read against readout 20 sections 1 and 3:

- census equal in every family, 1062 kernels per replay at both row counts (S3 is in both
  arms now, so 982 if main's count moved with it; equal on A and B is the gate);
- c1 rows 1 step unchanged at about 6.25 ms, 16 row step unchanged at about 6.78 ms;
- rows 8 whole step table: `prepare_decode_buffers` p90 back near 0.3 ms from 3.1, and
  `apply_sglang_qwen3_tts_result` p90 back near 0.1 ms from 3.0; the rows 1 to 15 steps' p50
  wall and idle down from 8.4 to 12.8 ms and 2.7 to 7.1 ms.

Archive the reports, not the pickles or traces.

## 3. Seeded c1 on B, one boot

`--seed 1234 --warmup 1 --max-concurrency 1`, full corpus. Pass: 1088 of 1088 WAV hashes equal
to A's seeded c1 hashes (readout 20 section 2's B pass, `seeded_c1/B/bench/wav_sha256.json`
in the slice A archive), and the c1 speed inside 2 percent of A's 2.58 req/s.

## 4. c16 on B, one boot

Unseeded, `--warmup 1`, full corpus. A is the follow up's slice A boot (readout 20 section
8: 17.03 req/s, median 0.914 s, p95 1.311 s, GPU 1 idle, GPUs 2 and 3 at 98 percent) when
the load is the same, otherwise one A boot at `04b62c255` in this session. Pass: B above A
on req/s, median and p95 beyond the 2 percent spread; this is the point of the slices. WER
and similarity are recorded, not compared. Peak memory as a level from the gpu samples plus
the per process reading.

## 5. Streaming on B, three passes

CI layout, two workers on GPUs 0 and 1, separate vocoder processes, warmup 1, full corpus,
c16. A is slice A's three passes from the follow up archive (readout 20 section 8: TTFC mean
0.133 to 0.208 s, inter chunk mean 0.072 to 0.076 s, 19.6 to 21.6 req/s) when the load is
the same, otherwise three passes at `04b62c255` in this session. Delta table: requests per
second, audio seconds per second, TTFC mean and p99, inter chunk mean and p99, request latency
mean and p99, RTF, continuity at 200 ms, completed. Pass: nothing worse than A's band, 3264 of
3264, and the question of readout 20 section 8 answered: TTFC at or below upstream main's
0.126 to 0.133 s mean of the E3 passes means the slice A cost was the two stalls these slices
remove.

## 6. Archive

Suites' output, the reports of section 2, the seeded hashes, the speed, WER and similarity
summaries, the gpu and process memory samples, the streaming pass summaries. Readout as doc
20's format, PR body in the #2108 format with the rows 8 host frames as the mechanism table.
