# Runbook for the KV pool admission bound slice

Head note after the run, readout in `11_kv_pool_readout_20260906.md`. Three expectations below
were wrong and are corrected here rather than rewritten, so the archive's README still maps onto
the sections. Section 4: this arm is main plus the slice, whose single signature startup captures
20 predictor graphs at 128 running, the 39 is the chain branch's two signature set. Section 3:
the cached token equality holds at c1 only, at c16 the count moves with prefill timing and the
two A boots differ from each other. Section 3: the c16 quality band of 116 to 128 errors and
71.18 to 71.32 similarity was written from a subset, eighteen archived boots span 114 to 135 and
71.12 to 71.34 with upstream main at both ends, so c16 quality is reported against that range
and the gate is c1 byte identity plus the kernel census.

Branch `perf/qwen3-tts-kv-pool-admission-bound`, head `8807833ca`, three commits on upstream
main `50db4a550` (sglang pin 0.5.18): the token cap derived in `adjust_overrides`, the builder
fraction removed, and the startup line reporting the pool against the bound. A is
`50db4a550`, B is `8807833ca`. Protocol of plan 07: interleaved boots A B B A, two per arm and
point, the one second sample of every GPU kept for each run, our own lanes never overlapping
a full corpus point. Same serve, benchmark, scoring and census commands as runbook 06.

## 1. Suites on B

```bash
pytest tests/unit_test/qwen3_tts -q
pytest tests/ -v -m "not benchmark and not accelerator" -x
pytest tests/ -v -m "accelerator and not benchmark" -x
```

New tests to see pass: `test_qwen3_tts_engine_caps_the_kv_pool_at_the_admission_bound`
(three parameter sets), `test_qwen3_tts_engine_keeps_a_deployment_token_cap`,
`test_qwen3_tts_engine_adds_no_token_cap_under_a_stage_byte_budget`,
`test_qwen3_tts_engine_leaves_the_memory_fraction_to_sglang`,
`test_qwen3_tts_engine_reports_the_pool_against_the_admission_bound` (two parameter sets).

## 2. Startup, B, default config

Three lines in the serve log decide the slice before any request:

- `KV Cache is allocated ... #tokens: 131072`, against 589142 on A
- `Memory pool end. avail mem=` about 59 GB, against 11.79
- `Qwen3-TTS KV pool holds 131072 tokens against an admission bound of 131072 (16 running x 8192 context)`

And from the memory csv, memory at ready about 27 GB, against 75.2.

## 3. Full corpus, c1 and c16, two boots per arm and point

Pass on c1: 1088 of 1088 WAVs byte identical between A and B and to the earlier archives
(`c1_wav_sha256.json` of the d26 archive), WER and similarity equal, latency and qps inside
the paired spread. The prefix cache gate: sum the `#cached-token` and `#new-token` of every
`Prefill batch` line of each serve log, both arms read 13576 cached of 74096 at c1.

Pass on c16: no `CUDACachingAllocator` retry warning in either B boot (A logs one or two),
WER inside 116 to 128 errors and similarity inside 71.18 to 71.32, peak GPU memory about
32 GB on B against about 80.6 on A, the cached token share equal between arms.

## 4. The larger ladder on B

The 128 running server, both flags together as in runbook 06 §3, then the request level
subtalker top k 64 run of 64 requests. Pass: `Captured 39 ... predictor CUDA graphs`, six lazy
`Captured ... key=(<bucket>, 'sampled', 64, ...)` lines, 64 of 64 complete, no cuDNN error,
no allocator warning. This is the run that failed 2 of 64 at 6 MiB free on the chain head.

## 5. Retraction at c16 with the allocator peak

`SGLANG_TEST_RETRACT=1 SGLANG_TEST_RETRACT_INTERVAL=64 SGLANG_TORCH_PROFILER_PROFILE_MEMORY=1`
in the server's environment, both arms, 192 requests in a profiler window. Before reading the
trace, confirm the flag reached the stage process: `tr '\0' '\n' < /proc/<pid>/environ | grep
PROFILER`. Pass: 192 of 192 on both, every retraction followed by its re prefill, and the
trace's allocator events present. If the trace still carries no `[memory]` events with the
flag confirmed in the process, record that and move on: the peak is then read from the
one second samples.

## 6. Allocator snapshot, the composition measurement

On A and on B, one boot each with the profiling branch's hook merged in or applied on top:
`SGLANG_TORCH_PROFILER_MEMORY_SNAPSHOT=1` in the server's environment, one profiler window at
c16 of 192 requests, the snapshot written at profiler stop. Reduce it with
`perfkit.py snapshot` to allocations by call site at the window's end. This is the table
plan 05 section 1 needs for the reductions: the size of the second tokenizer copy, the three
vocoder graph holders, the predictor pool, the attention workspaces and the vocoder's
whole utterance decode transient.

## 7. After this slice

The rope store branch rebased on the chain head plus this branch, A becomes chain plus this
slice, its c16 pair runs with the card free.
