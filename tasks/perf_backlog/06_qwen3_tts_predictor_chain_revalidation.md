# Revalidation of the predictor chain branch after the review fixes

Branch `perf/qwen3-tts-predictor-chain`, head `0a88253c6`. The five commits after the benchmarked
head `8c8ae636b` are `92bad0055` (greedy seed table gate), `00b31da66` (startup set outside the
lazy budget), `377e42a1a` (temperature floor test), `c93ba9858` (history compaction at
retraction) and `0a88253c6` (typing). None of the branches below is pushed yet.

Arms: A is `7989a5ed2`, the main commit merged into the branch at `6b233296d`, so the diff of
the arms is the branch alone. B is `0a88253c6`. Checkouts on the box follow the archive layout,
`tmp/qwen3-tts-chain-<label>-A` and `-B`, and the archive `run.sh` of
`qwen3-tts-chain-newbase-20260905` runs unchanged with its `OUT` and `select_arm` paths renamed.

## What changed at runtime, and the expected effect

| Fix | Runtime effect on the benchmarked default (16 running, all rows sample) | Effect elsewhere |
| --- | --- | --- |
| greedy seed table gate | none, the table is built when any row samples | the argmax graph loses one kernel per replay, the add at sglang_model.py:1576 |
| startup set outside the lazy budget | none, the startup set is 11 keys as before | ladders above 16 buckets keep their mixed buckets and later signatures replay in place of eager |
| history compaction at retraction | none, the A/B logs show no retraction | one stack copy per retracted request on a path that already re-prefills |
| temperature floor test, typing | none | none |

So the rerun is a no regression check: the same 1222 kernels per replay, the same c1 bits, and
c1 and c16 latency and QPS inside run to run noise of the last B run. It is not expected to move
the headline numbers.

## 1. Suites on the pinned box, both branches

Record the SHA next to every log.

```bash
git rev-parse HEAD
pytest tests/unit_test/qwen3_tts -q
pytest tests/ -v -m "not benchmark and not accelerator" -x
pytest tests/ -v -m "accelerator and not benchmark" -x
```

Expected on the chain branch: the qwen3_tts suite passes and holds the new tests
`test_greedy_prediction_reads_no_seed_state`,
`test_startup_capture_builds_the_ladder_for_both_sampled_signatures`,
`test_startup_set_stays_outside_the_lazy_capture_budget`,
`test_qwen3_tts_prepare_decode_buffers_stages_the_temperature_floor`,
and in `tests/unit_test/pipeline/test_scheduler.py`
`test_retracted_request_history_gets_its_own_storage_before_requeue` and
`test_retracted_request_without_history_is_requeued_untouched`.

Expected on the rope store branch: `pytest tests/unit_test/qwen3_tts -q` passes with the two
former failures gone (`5e4424f4a` publishes the runtime context in the rope store test).

## 2. The reviewer's probes

`tasks/pr_1971_qwen3_tts_predictor_chain_review_20260905/remote_probes.py` builds a talker
without `__init__`, so the new counter must be set. Add one line after
`talker._predictor_graph_capture_count = 0` in `graph_capacity`:

```python
talker._predictor_graph_startup_count = 0
```

Then, from the branch checkout:

```bash
PYTHONPATH=. python remote_probes.py
```

Expected: `startup_key_coverage` shows `missing_reachable_mixed_buckets: []` at every maximum,
with `captured` 11 at 16, 23 at 64 and 39 at 128 (buckets 6, 12 and 20, mixed skips bucket 1).
`temperature_staging_bits` PASS. `survivor_history` still reports amplification 16: the probe
keeps one request in a running batch of 16 for 1024 steps and never retracts it, which is the
accepted running window bound. The compaction runs at requeue after a retraction and is
asserted by the scheduler unit tests with real views.

## 3. Graph coverage on the larger configuration (F1)

The ladder is sglang's decode `cuda_graph_bs`, which omni builds from `cuda_graph_max_bs`
(engine default 32, generation_batch_policy.py:154), filtered to the running cap
(sglang_model.py:887, 1152). The validator requires the graph cap to cover the running cap, so
both flags move together:

```bash
python -m sglang_omni.cli serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml --host 127.0.0.1 --port 31001 \
  --tts_engine.engine.max_running_requests 128 --tts_engine.engine.cuda_graph_max_bs 128
```

Record from the serve log and the memory csv:

- the line `Captured 39 Qwen3-TTS predictor CUDA graphs for signatures=[...] in N s`, the review
  asked for this startup time and the memory at ready on the supported larger configuration
- memory.used at ready and its peak during the c16 run below

Then on that server, sampled and mixed traffic:

```bash
bench 16 $OUT/B/cov_c16_mixed --max-samples 192 --subtalker-dosample-ratio 0.5
bench 16 $OUT/B/cov_c16_topk64 --max-samples 64 --top-k 64
```

Expected: no `holds N keys beyond the startup set` warning, no `Disabling Qwen3-TTS predictor
CUDA graph`, and exactly one `Captured Qwen3-TTS predictor CUDA graph for key=(16, 'sampled',
64, ...)` line for the top k run, that is one lazy capture inside a budget that startup no
longer consumes. The mixed run replays the startup mixed keys and captures nothing.

## 4. The argmax graph (F3)

A greedy sub-talker run captures the argmax keys lazily and its census should show the add
kernel gone. On a default server, after the one unprofiled warmup request:

```bash
curl ... /start_profile (as in run.sh, run_id census_greedy_c1)
bench 1 $OUT/B/census_greedy_c1/bench --max-samples 12 --subtalker-dosample-ratio 0
curl ... /stop_profile
perfkit.py ingest, then census --rows 1
```

Expected: B's argmax replay matches A's argmax replay in kernel names and counts. A returns the
argmax before any seed work (sglang_model.py:1559 at `7989a5ed2`), and the gate removes the one
add kernel the branch head before it built for every frame. The sampled census stays at 1222.

## 5. Retraction (F2)

sglang's test switch retracts the request with the fewest output tokens every N forwards while
the batch holds more than one request (scheduler.py:3491-3506, schedule_batch.py:2816-2833). It
is read at import, so it goes on the server environment. The profiler records allocator events
when `SGLANG_TORCH_PROFILER_PROFILE_MEMORY=1` (torch_profiler.py:115).

```bash
SGLANG_TEST_RETRACT=1 SGLANG_TEST_RETRACT_INTERVAL=64 SGLANG_TORCH_PROFILER_PROFILE_MEMORY=1 \
  python -m sglang_omni.cli serve ... (default config)
curl ... /start_profile (run_id retract_c16)
bench 16 $OUT/B/retract_c16 --max-samples 192
curl ... /stop_profile
```

Run the same on A. Expected: `Testing retraction. #retracted_reqs: 1` lines in both logs,
every request completes with no HTTP error, and the trace's allocator series shows B's peak
allocated bytes at or below A's. B is the only arm that shares history storage across a batch,
and the compaction is what keeps a retracted request from holding the snapshots of its peers.
The unique retained storage after compaction is asserted by the scheduler unit test, the serving
trace gives the peak.

## 6. The A/B, same protocol as before

Unchanged `run.sh`: census on both arms (one warmup, 12 requests per concurrency unit at c1 and
c16), then A c1, B c1, B c16, A c16 on the full corpus with `--seed 1234 --warmup 0`, then WER
and similarity. Checks:

- `census_diff_c1.md` and `census_diff_c16.md`: 1371 against 1222 kernels, the removed families
  60 indexSelectSmallIndex, 74 elementwise, 15 reduce_kernel, no family grown
- c1 WAV SHA256: B equal to A, 1088 of 1088, and equal to `c1_wav_sha256.json` of the
  `qwen3-tts-chain-newbase-20260905` archive, the device path did not change since `2c00eb688`
- c1 and c16 latency and QPS against the last B run, inside the spread of the earlier boots
- c16 WER and similarity inside the 116 to 128 errors and 71.18 to 71.32 band of identical
  kernel boots
- memory csv peaks against 76887 MiB (c1) and 80735 MiB (c16) of the last B run
- serve logs: no lazy capture, no fallback, no retract, no CUDA error

## 7. Rope store branch

`perf/qwen3-tts-predictor-rope-store` is stacked on the chain branch, its last A arm was
`8c8ae636b`. Before its own A/B it takes the five new commits, and its A becomes `0a88253c6`. Its
c16 rerun waits for the memory provisioning slice: the last B c16 failed inside cuDNN's plan
build in the reference encoder with the card at 81077 of 81079 MiB, which the rope store does
not cause (A and B at ready 6 MiB apart).
