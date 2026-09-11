# Runbook for S3, the predictor residual add inside the norm that follows it

Branch `perf/qwen3-tts-decode-step`, head `9b036beb1`, one commit on upstream main `80b5aaed7`
(sglang 0.5.19, #2057 in). A is `80b5aaed7`, B is `9b036beb1`.

The change is one function, `_predictor_forward_one_token` in sglang_model.py. Each layer's
mlp output is added to the residual inside the next norm, in the residual form the backbone
layer uses, and the final norm takes the last mlp output the same way. sglang's fused add and
norm does both in one kernel, in place, on 2D rows. The standalone add kernel of every layer
is gone: 16 per sub step, 80 per replay, 1062 to 982 kernels. The o_proj epilogue add is
unchanged. Design in doc 04 section 4.2, numerics decided by E1 below.

Protocol from the 2026-09-09 handoff: unseeded, warmup 1, boots interleaved A B B A, two per
arm and point, one second samples of all GPUs, streaming as the main result. One seeded c1 pass
per arm only for the byte identity check, only if E1 says the kernel is bit exact.

## 1. E1, before anything else

On a checkout of B, in the box venv:

```bash
python tasks/perf_backlog/scripts/e1_fused_add_rmsnorm.py --hidden 1024
```

The script is on the analysis branch, copy it in. It compares, for rows 1 to 64 and 1000
trials each, today's path, the bf16 add then the norm, against `RMSNorm.forward_cuda(x, residual)`
on the same class the predictor uses.

- Every row count 1000 of 1000 equal on both columns: S3 is bit exact. Section 4 runs the
  seeded c1 pass and the gate is byte identity.
- Residual equal, normed not: the fused kernel normalizes the fp32 sum. S3 ships under the c16
  band, record the max difference column in the readout, skip the seeded pass.

Paste the table into the readout either way.

Result, 2026-09-11, H100, sglang 0.5.19: residual equal 1000 of 1000 at every row count,
normed equal 0 of 1000 at every row count. The fused kernel normalizes the fp32 sum, today's
path normalizes the bf16 rounded sum. S3 is the band gate, sections 4 and 5 decide, the seeded
pass is skipped. The residual carried between layers is the same bits on both forms.

## 2. Suites on B

```bash
git rev-parse HEAD
pytest tests/unit_test/qwen3_tts -q
pytest tests/ -v -m "not benchmark and not accelerator" -x
pytest tests/ -v -m "accelerator and not benchmark" -x
```

New tests, all in `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`, on a three layer
copy of the fixture predictor so the fused branch runs on the inner layers:

- `test_eager_predictor_in_place_residual_norms_match_the_out_of_place_form`, three parameter
  sets, the talker's in place forward against the same residual form on cloned operands, bit
  for bit, output and both caches. This replaced a tolerance test against the add then norm
  form after E1 on 2026-09-11 showed the two forms differ by design (below), so no tolerance
  could be pinned and the aliasing contract is what the test has to hold.
- `test_eager_predictor_adds_each_residual_inside_the_norm_that_follows`, the exact sequence of
  plain and residual norm calls for three layers, every call on 2D rows.
- `test_eager_predictor_output_survives_the_next_token`, the returned tensor is not aliased
  by the next call.
- `test_eager_predictor_accepts_a_strided_input_and_leaves_its_neighbours`, a strided input
  slice gives the contiguous result and its neighbours in the parent tensor are untouched.

Every existing graph bit identity test unchanged, they compare graph against eager on the new
code.

## 3. Census, both arms, 1 and 16 rows

As runbook 12 section 3. Pass: 1062 on A and 982 on B at both row counts, the elementwise
family down by 80 (the residual add `vectorized_elementwise_kernel`), the norm family the same
count with the fused add norm kernel name in place of the plain one for 16 of every 33 hidden
norms per sub step, no family grown, replay busy down by about 0.09 ms at both row counts
(80 adds of 1.1 us). The attention, GEMM and rope families unchanged.

## 4. Full corpus, c1 and c16

Unseeded, `--warmup 1`, order A c1, B c1, B c16, A c16 then the reverse. Pass: median and p95
latency and req/s on B inside or better than A's paired spread at both points, WER inside 114
to 135 errors and similarity inside 71.12 to 71.34 at c16, peak memory equal as a level.

If E1 said bit exact, one extra c1 pass per arm with `--seed 1234 --warmup 0`: 1088 of 1088
WAVs byte identical between A and B, and B's hashes equal to the rope archive's B c1 hashes
(`rope-20260909-results/B/ab_c1_r1/wav_sha256.json`), since main's predictor device path did
not change between `79cdfe185` and `80b5aaed7` (the 0.5.19 bump moved no kernel the census
names; the census of section 3 says so or not).

## 5. Streaming, one pair at c16, the main table

CI layout, two workers behind the router, separate vocoder processes, full corpus, c16,
warmup 1, three passes per arm. Report the delta table: requests per second, audio seconds per
second, TTFC mean and p99, inter chunk mean and p99, request latency mean and p99, RTF,
playback continuity at 200 ms, WER, completed. Pass: nothing worse than A's three pass band,
3264 of 3264 per arm, no range screen hit, no traceback.

## 6. Archive

Suites' output, census directories and diffs, serve logs, speed and WER and similarity
summaries, the memory csvs, the streaming pass summaries, and the E1 table. Readout as doc 16,
PR body from it in the #2057 format: mechanism, changes, census table, the c1 and c16 delta
tables from the best pair, the streaming delta table.

## 7. E2, the host tail breakdown, same session, on A

One server on main with `SGLANG_TORCH_PROFILER_WITH_STACK=1` in its environment, default
config. One profiler window at c1 of 12 requests and one at c16 of 192 requests, the same
`/start_profile` and `/stop_profile` calls as the census. Then:

```bash
python perfkit.py ingest $OUT/e2_c16/trace_tts_engine_pid*_rank0.trace.json.gz -o $OUT/e2_c16/tts_engine.pkl
python perfkit.py hosttail $OUT/e2_c16/tts_engine.pkl --rows 16 --top 40 --json $OUT/e2_c16/hosttail_rows16.json | tee $OUT/e2_c16/hosttail_rows16.md
python perfkit.py hosttail $OUT/e2_c1/tts_engine.pkl --rows 1 --top 40 --json $OUT/e2_c1/hosttail_rows1.json | tee $OUT/e2_c1/hosttail_rows1.md
```

`perfkit.py` is `tasks/qwen3_omni_0518_numerics/scripts/perfkit.py` on the analysis branch,
with the `hosttail` command added on 2026-09-11. The ingest line prints a `pyfuncs` count; if
it is zero the server ran without the stack flag and the report falls back to aten ops only.
The traces are large with stacks, keep the windows to the sizes above.

What to read: the tail p50 against doc 02's 1.1 to 1.4 ms, the owner table (how much of the
tail is omni owned), and the top frames. S4 takes only omni owned frames above the noise
floor of the (no frame) row.

First run, 2026-09-11: the traces were made and ingested (2.1 M and 6.5 M Python frames), but
the first version of `hosttail` measured the wrong window and misread the owners: it took the
tail from the last device span of the step, which is the next step's input staging copy just
before the launch, so it reported 37 to 48 us, and it matched owners on paths with a leading
slash the profiler does not write, so every frame read as python. The tool was corrected the
same day: the tail now starts at the end of the predictor replay's device work, C level frames
fold into their Python caller, a child span is clipped to its parent, and the report carries the
device idle time of the step. Rerun on the retained pickles, no new trace needed:

```bash
python perfkit.py hosttail $OUT/e2_c16/tts_engine.pkl --rows 16 --top 40 --json $OUT/e2_c16/hosttail_rows16.json | tee $OUT/e2_c16/hosttail_rows16.md
python perfkit.py hosttail $OUT/e2_c1/tts_engine.pkl --rows 1 --top 40 --json $OUT/e2_c1/hosttail_rows1.json | tee $OUT/e2_c1/hosttail_rows1.md
```
