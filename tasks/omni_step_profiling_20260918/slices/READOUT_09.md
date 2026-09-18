# Run 09 readout: PF A/B (PR #2244)

moss RTX 4090 D, Qwen3-TTS-12Hz-1.7B-Base, seed-tts English full corpus (1,088), warmup 1.
A main `27b5b0d4f`, B `perf/qwen3-tts-base-prefill-graph` `2f60fc1a8`; both arms of a
point at the same time on separate cards. Runbook `scripts/run_pf_ab.sh`. Text outputs:
`artifacts/moss_omni_step_profiling/omni_step_profiling/run09`, `run09b`, `run09c` (tgz
md5 155d7dd9..., ab71f09d..., 6e330827..., 812d6590...). Every boot: head, import path
under its worktree, A logs the prefill backend disabled, B captures it.

## 1. Results (the PR tables)

- Stream c16 (run09, cards 1, 2): req/s 13.082 to 14.934 (+14.2%), RTF mean -12.3%,
  TTFC p50 133.1 to 115.1 ms, p99 760.4 to 789.9 ms (+3.9%, the one cell that moves the
  other way; unexplained, one boot per arm), inter-chunk p95 -16.8%. WER 1.06 / 1.02%,
  similarity 71.41 / 71.24. B replays 819 of 820 prefills; the one eager is 807 tokens.
- Stream c1 seeded (run09, cards 3, 4): TTFC -10 ms at every percentile (-13%), decode
  cadence unchanged (-0.7%). 667 of 1,088 WAVs byte identical (finding 5 of the plan);
  no A/A seeded identity measured, so the share owed to PF alone is not known.
- Buffered c16 at `mem_fraction_static` 0.80 on both arms (run09c, cards 5, 6): req/s
  +5.4%, RTF mean -8.2%, latency p50 -7.5%. Capture leaves 3.31 GB free at 0.80
  (2.33 GB at 0.85).
- Ladder top 512: 1 of 820 (stream) and 1 of 847 (buffered) prefills exceed it; median
  prefill 59 to 62 tokens. No ladder change.
- The mechanism accounts for the c16 gain: about 820 prefills per run, about 12 ms saved
  each (run 07: 26.1 to 13.7 ms), about 10 s of an 83 s run.

## 2. Findings outside the PR

1. Main's buffered c16 runs out of memory at the shipped 0.85 on 24 GB. run09: 115
   failed requests; run09b: 446 failed, the tts_engine scheduler thread crashed
   (`omni_scheduler.py:1423` run_batch, the eager Talker prefill rope
   `thinker_model.py:258` asking for 2 MiB with 1.69 MiB free), first failure in the
   preprocessing speaker encoder. Peak 24,082 MiB. B peaks at 24,060 MiB in the same
   point and completes: replayed prefills use the pool captured at startup, so it does
   not make the failing allocation; its headroom is the same. Not a PF effect; it is the
   deferred memory sizing, now reproduced at buffered c16 on 24 GB.
2. Unseeded sampling runs away to the 2,048-token cap (163.84 s) about once per 1,088
   buffered requests, in either arm (A run09 `common_voice_en_19770519`, B run09c
   `common_voice_en_19916471`). The WavLM similarity scorer cannot score a 163 s clip on
   24 GB (asks 16 GiB).
3. Runbook fix: `--skip-gpu-cleanup` let similarity start before the ASR server freed
   the card; the script now waits for the card (`3f408e348`).
