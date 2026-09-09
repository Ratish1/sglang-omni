# Runbook for the predictor rope store slice (S2)

Branch `perf/qwen3-tts-predictor-rope-store`, head `79cdfe185`, the three S2 commits with
upstream main `4562a7ef3` merged in on 2026-09-09. Main now carries the predictor chain (#1971)
and the memory provisioning slice (#2042), so the c16 pair runs with the card free, which is the
condition doc 08 section 2 set for this run. A is upstream main `4562a7ef3`, B is `79cdfe185`.
The runtime diff is one file, `sglang_model.py`: the predictor's private cache becomes slot
major, sglang's rope kernel stores k and v into it in the same launch that rotates q and k,
and the attention reads the cache through a transposed view. The two cache copies per layer
are gone on the fused path, 160 kernels per replay over the 16 layers and 5 sub steps. Design
and gate in doc 04 section 4.1.

Protocol of plan 07: boots interleaved A B B A, two per arm and point, the one second sample of
every GPU on the host kept for each run, our own lanes never overlapping a full corpus point.
Serve, benchmark, scoring and census commands as runbook 06, `--seed 1234 --warmup 0` on the
full corpus.

## 1. Suites on B

```bash
git rev-parse HEAD
pytest tests/unit_test/qwen3_tts -q
pytest tests/ -v -m "not benchmark and not accelerator" -x
pytest tests/ -v -m "accelerator and not benchmark" -x
```

New tests to see pass, all in `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`: the
accelerator test `test_rope_store_writes_the_cache_the_copy_path_writes`, which is experiment
E0 of doc 04, a real `RotaryEmbedding` at head dim 128 with batch 1 and 16 over every slot, the
stored rows equal to the copy path and q and k equal to the plain rope call bit for bit. And the
gate test: the copy path for the fixture's identity rotary, the fused path for a CUDA rotary
without the fallback kernel. Every existing bit identity test of the predictor graph unchanged.

## 2. Startup, B, default config

One line decides the gate before any request. In the census trace of section 3 the rope kernel
name is `fused_rope_store_kernel`. If it reads `fused_rope_kernel` the gate took the copy path
and the run is not measuring S2, stop and read `_resolve_predictor_rope_store`.

Serve logs on B, same checks as runbook 06: no lazy capture, no fallback, no retract, no CUDA
error, and the startup line of #2042 with the pool within a few hundred tokens of A's, since S2
allocates nothing new.

## 3. Census, both arms, 1 and 16 rows

Same as runbook 06: one warmup, then a profiler window of 12 requests per concurrency unit at c1
and c16, `perfkit.py ingest`, `census --rows 1` and `--rows 16`, then `perfkit.py diff` of A
against B at each row count.

Pass:

- kernels per replay 1222 on A and 1062 on B at both row counts
- the `elementwise_kernel` family down by 160, the two cache writes per layer and sub step, and
  `fused_rope_store_kernel` in place of `fused_rope_kernel` with the same count, 80
- no family grown
- replay busy down by at least the removed durations, about 0.31 ms at 1 row and 0.38 ms at 16
  from the doc 03 section 12 per kernel times
- the attention kernel: same name as A, `sdpa_sm80_flash_fprop_wmma_f16_knob_2_64x32x128`, and
  its time unchanged. The cache is now read through a transposed view with different strides,
  so cuDNN may pick another kernel. A different name is not a failure by itself, the c1
  identity of section 4 decides, but it is recorded.

## 4. Full corpus, c1 and c16, two boots per arm and point

Order A c1, B c1, B c16, A c16, then the second pair the other way round.

Pass on c1: 1088 of 1088 WAVs byte identical between A and B. The hashes against the
`c1_wav_sha256.json` of the #2042 archive are recorded but not a gate: main took #2005, the
repetition penalty change, between that run and `4562a7ef3`, and whether it moved the
Qwen3-TTS c1 bytes is read from A's hashes, not assumed. WER and similarity equal. Median
latency and QPS: B better than or equal to A inside the paired spread, the expected gain is
smaller than S1's, the removed kernels are about 0.3 ms of a 4.4 ms replay.

Pass on c16: WER errors inside 114 to 135 and similarity inside 71.12 to 71.34, the eighteen
boot band of runbook 10. Median and p99 latency and QPS paired against A, no metric worse
beyond the run to run spread of the two A boots. Peak GPU memory from the one second samples
equal between arms within the sample noise, S2 changes no allocation.

If c1 is not byte identical the attention kernel changed under the transposed view, and the
slice is G2 of doc 04 section 6, not G1: report the number of differing WAVs and the c16
quality band, and hold the PR until the reason is read from the census names.

## 5. Streaming, one pair at c16

The predictor runs the same code in streaming, so the c1 identity above already covers the
codes streaming emits. What streaming adds is the timing side: a shorter predictor step moves
the chunk cadence. One pair, A then B, in the CI layout, the one the stream repro used: two
workers behind the router, `--vocoder.process vocoder`, `--vocoder.gpu_memory_fraction 0.10`,
caps 64, engine fraction 0.85, full corpus, c16, warmup 1, three passes per boot.

Pass: completed equal to routed on both workers in every pass, no range screen hit and no
traceback in either worker log, WER inside the c16 band above, TTFC mean and p99 and ITL p99
on B inside the band of A's three passes or better.

## 6. What to archive

The suites' output, both census directories with `census_diff_c1.md` and `census_diff_c16.md`,
the four serve logs per arm, the WAV hash files, the WER and similarity summaries, the speed
tables, the memory csv per run, and the streaming pass summaries. Readout goes in
`13_rope_store_readout_<date>.md`, PR body from it.
