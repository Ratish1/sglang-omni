# Runbook for slice A of plan 18, the token ids staged before the predictor

Branch `perf/qwen3-tts-stage-ids-early`, head `51c5bb064`, two commits on upstream main
`b2cc93b0a` (sglang 0.5.19, S3 in #2108 still open, independent of this branch). A is
`b2cc93b0a`, B is `51c5bb064`.

The runtime diff is nine lines in `model_runner.py`. `_collect_codes` calls `_stage_token_ids`
before `code_predictor_forward`, so the pinned copy of the layer 0 ids and its event sit in the
stream ahead of the predictor and the finalize wait returns while the predictor runs. The host
tail, the next batch's preparation and the next backbone launch then overlap the predictor, and
the device sees the next backbone queued behind it. No kernel changes, bit exact by
construction. `lookahead_eligible` returns False, the runner's collect has no launch and
resolve halves. Design in plan 18 section 3, gates in section 5.

Protocol: unseeded, warmup 1, one boot per arm and point, the census is the measurement. One
seeded c1 pass per arm for byte identity, since the change is bit exact by construction.

## 1. Suites on B

```bash
git rev-parse HEAD
pytest tests/unit_test/qwen3_tts/test_decode_step_staging.py -v
pytest tests/unit_test/qwen3_tts -q
pytest tests/ -v -m "not benchmark and not accelerator" -x
pytest tests/ -v -m "accelerator and not benchmark" -x
```

New tests, `tests/unit_test/qwen3_tts/test_decode_step_staging.py`:

- `test_collect_codes_stages_the_ids_before_the_predictor_runs`, the fake predictor sees the
  host ids staged when it is called, the code and feedback rows come from the post predictor
  snapshot and survive the buffers being zeroed, the EOS row is excluded.
- `test_lookahead_is_never_eligible`, a history free batch and an empty batch, both of which
  the base gate accepts, are refused.
- `test_staged_ids_resolve_while_the_predictor_is_still_running`, accelerator: the fake
  predictor sleeps the stream, the resolved ids come back while the stream is still busy. Fails
  on A, where the event sits behind the sleep.
- `test_staged_ids_keep_the_sampled_values_when_later_stream_work_overwrites_them`,
  accelerator: the fake predictor overwrites its input view, the resolved ids are the sampled
  values.

## 2. Census and identity, both arms

Census as runbook 12 section 3, at 1 and 16 rows. Pass: every family and count equal on A and
B, replay busy equal within the run to run spread.

One c1 pass per arm with `--seed 1234 --warmup 0`: 1088 of 1088 WAVs byte identical between
A and B.

## 3. E2 timeline on B, the mechanism gate

One server on B with `SGLANG_TORCH_PROFILER_WITH_STACK=1`, default config, one profiler window
at c1 of 12 requests and one at c16 of 192, as runbook 15 section 7. Then:

```bash
python perfkit.py ingest $OUT/a_c1/trace_tts_engine_pid*_rank0.trace.json.gz -o $OUT/a_c1/tts_engine.pkl
python perfkit.py timeline $OUT/a_c1/tts_engine.pkl --rows 1 | tee $OUT/a_c1/timeline_rows1.md
python perfkit.py hosttail $OUT/a_c1/tts_engine.pkl --rows 1 --top 40 --json $OUT/a_c1/hosttail_rows1.json | tee $OUT/a_c1/hosttail_rows1.md
python perfkit.py ingest $OUT/a_c16/trace_tts_engine_pid*_rank0.trace.json.gz -o $OUT/a_c16/tts_engine.pkl
python perfkit.py timeline $OUT/a_c16/tts_engine.pkl --rows 16 | tee $OUT/a_c16/timeline_rows16.md
python perfkit.py hosttail $OUT/a_c16/tts_engine.pkl --rows 16 --top 40 --json $OUT/a_c16/hosttail_rows16.json | tee $OUT/a_c16/hosttail_rows16.md
```

Read against readout 17 and the c1 timeline of runbook 15 section 7 (step 8.56 ms, device
idle 3.28 ms, wait ending at 7.22 ms, tail 1.37 ms):

- The finalize wait, `event.synchronize` in `_resolve_host_token_ids`, is now short: the
  backbone and the sample are done by the time the predictor launch submission ends.
- The next step's backbone `cudaGraphLaunch` host start lies before the predictor replay's
  device end.
- Device idle per step down by about the tail, toward 2 ms at c1, and the step wall toward
  the device bound of about 6.5 ms. What remains is the 0.5 ms before the predictor (slice D)
  and the two synchronizations of slices B and C on finish and batch change steps.

If the idle does not fall, the timeline names the synchronization left in the tail and that is
the next slice.

## 4. Full corpus, c1 and c16, one boot per arm and point

Unseeded, `--warmup 1`, A c1, B c1, A c16, B c16. Report median, p95, p99, req/s, RTF, WER,
similarity, peak memory. Pass: B better or inside the boot spread on every latency and
throughput figure, WER inside 114 to 135 errors and similarity inside 71.12 to 71.34 at c16,
peak memory equal as a level. Repeat a point only when its delta is inside about 2 percent.

## 5. Streaming, one pair at c16, the main table

CI layout, two workers behind the router, separate vocoder processes, full corpus, c16, warmup
1, three passes per arm. Step timing is what streaming feels, so this pair is owed. Delta
table: requests per second, audio seconds per second, TTFC mean and p99, inter chunk mean and
p99, request latency mean and p99, RTF, playback continuity at 200 ms, WER, completed. Pass:
nothing worse than A's three pass band, 3264 of 3264 per arm, no traceback.

## 7. E3, added 2026-09-12, the c16 similarity streak

Six of seven unseeded c16 and c1 pairs since S3 read B below A on similarity by 0.09 to 0.23,
inside every interval but one signed. The one same-draw comparison so far (S3 seeded c1) read
B above A, and slice A is bit exact at c1, so the only mechanism left at c16 is the vocoder's
batch composition, which a faster talker changes. Two boots settle it:

```bash
# one server per arm, same as the seeded c1 pass, then on each:
python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta zhaochenyang20/seed-tts-eval-arrow \
  --ref-format references --lang en --seed 1234 --warmup 1 --max-concurrency 16 \
  --port $PORT --output-dir $OUT/seeded_c16/$ARM/bench
```

Then the WAV hashes of both arms and the similarity and WER scores of both. Read:

- Hashes equal 1088 of 1088: the c16 quality numbers of a bit exact slice are the draw, and no
  future bit exact slice runs a c16 quality comparison.
- Hashes differ: the count that differ and the paired per sample similarity delta, scored on
  the Mac as in readout 20, say whether batch composition moves similarity. That is a vocoder
  finding (backlog M1), owned by its own item, not by this slice.

Archive `seeded_c16/{A,B}` with the rest.

## 6. Archive

Suites' output, census directories, the seeded hashes, the E2 traces and their timeline and
hosttail reports, serve logs, speed and WER and similarity summaries, the memory csvs, the
streaming pass summaries. Readout as doc 17's format for the timeline and doc 16's for the
A/B, PR body in the #2108 format.
