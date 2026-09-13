# 33. Runbook: the vocoder bootstrap through captured graphs (doc 32 item 1)

Two branches on upstream main 3060470a8.

- `perf/qwen3-tts-codec-precompile` at b64dd1f71, three commits: the decoder traces a
  compiled shape on the codes and state the graph runner warms and captures with, so
  each shape compiles once instead of twice (0ba531872). Bit exact, verified by doc 34's
  bench (12 graphs for 12 shapes, capture 100.7 to 55.3 s). The review pass (4f7d9a32a)
  dropped the Dynamo limit raise (the warm runner's four shapes fit the default of 8)
  and the grad mode check (the runner is the only caller and runs under inference mode).
  The ceiling of 8 traced shapes per function is noted at the compile call (b64dd1f71)
  rather than raised: the warm buckets trace 4 and no knob adds a compiled shape.
- `perf/qwen3-tts-bootstrap-graphs` at 8b66f41d4, twelve commits: the runner's window
  schedule and bucket queries (7e5e2aa48), the window runner and its knob (cb9cf8f46),
  the windowed decode path (1453ef538), no default until measured (fd54363d6), the
  precompile fix cherry-picked (c2212e459, 50830c2c3), the first bucket and compile
  choice (1498c2078), the measured default ladder 1 to 64 (05683d8a4), then the review
  pass: the windowed decode fills one owned waveform per cohort and the scheduler
  counters are gone (8ba94ef0c), the window runner takes the warm runners' buckets and
  compiles only the steady stride they already trace (988fdcdfd), the planner and
  docstrings trimmed (fa329e4f4), the ceiling note cherry-picked (8b66f41d4).

Review decisions, 2026-09-13. One bucket was tuned to the seed-tts corpus, where
reference lengths differ per request: a deployment with one fixed cloning voice makes
every bootstrap the same width, and at bucket 1 a cohort of n rows became n serial window
sequences where main ran one batched eager decode. The window runner now uses the same
buckets as the cold and warm runners (1, 2, 4, 8), the set the initial worker's cohorts
already split at. With four buckets, compiling widths 16, 32 and 64 costs 12 shapes,
about 50 s of boot by doc 34's table, and the reason recorded for that subset was wrong
(those rungs appear at most once per bootstrap, only 64 repeats). Compile of the window
rungs is therefore its own later slice with its own boot and first chunk reads. This
slice captures every window eager except width 8, which replays the compiled step the
warm runners trace anyway. Doc 34 section 3 gives the device times this arm should show:
uncompiled windows at cap 64 are 9.9, 14.2, 24.1 and 26.9 ms at 40, 80, 120 and 160
frames against eager 10.6, 13.8, 16.2 and 20.0 in isolation, with host time 0.1 to 0.2 ms
against 8.6 to 9.0 ms and about 860 launches.

Step 0 of the earlier version of this runbook is done: doc 34 holds the bench readout.
The steps below are the session that decides both PRs. Same session rules as doc 31:
plain server command, GPU 0, GPUs 1 to 3 recorded before every boot, dmon on every boot,
full corpus, warmup 1, no seed, two passes per arm, event recorder in pass 2 stopped
after 200 completions, decode log gap check on the first boot.

What the change does. A reference prefixed bootstrap (reference frames plus the first
chunk) used to be one eager decode of an uncaptured width, about 860 host launches and
about a thousand lock handoffs. The initial worker now owns a second graph runner, the
window runner, with widths 1, 2, 4, 8, 16, 32, 64 at the buckets 1, 2, 4, 8. Width 8
replays the compiled step the warm runners trace, every other width is captured eager.
A same width cohort whose width no runner captured is consumed as a sequence of those
widths against its arena slots, largest first, one replay per window, each replay's
output copied into one waveform the cohort owns. The knob is the vocoder factory
argument `incremental_codec_cuda_graph_window_frames`; an empty list turns windowing
off. The codec state line in the serve log carries a `window` runner entry next to
`cold` and `warm` with its captured keys, footprint, `replays` and `fallback_counts`.

## Step 1, unit tests on the box, full files, on 8b66f41d4

```bash
git fetch origin perf/qwen3-tts-bootstrap-graphs && git worktree add tmp/bw 8b66f41d4
cd tmp/bw
python -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py -q
python -m pytest tests/unit_test/qwen3_tts/test_incremental_codec_cuda_graph.py -q
python -m pytest tests/unit_test/qwen3_tts/test_pipeline.py -q
```

Three tests in the graph file run only with CUDA and run here. All must pass before a
boot. Archive the output. Every test on both branches is unrun until this step.

## Step 2, the pair that decides #2123 (2 boots)

A: upstream main 3060470a8 plus `tasks/qwen3_tts_e4_investigation_20260912/early_ids.patch`.
B: 8b66f41d4 plus the same patch (it touches model_runner.py only). Streaming c16.

Reads per arm from `first_chunk_anatomy.py` on the pass 2 events plus the client
summary: TTFC mean and p99, req/s, inter chunk, preprocessing p50, the first frame to
first audio segment at ahead 0 and its mean, the prefill by overlap table, the cadence,
dmon GR active during traffic. From B's serve log: boot time from launch to ready
against A, the window runner's captured keys and `graph_footprint_bytes`, the last
codec state line's window runner `replays` and `fallback_counts`, and the cold runner's
`uncaptured_fresh_frames`. On both arms grep the serve log for `recompile_limit` and
`disabled the`, expected empty: Dynamo admits 8 traced shapes per function, the warm
runner traces 4 after the precompile fix, and a hit disables the runner with only a
warning.

Expected on B, stated before the run: the bootstrap segment at ahead 0 from about 52 ms
to 10 to 27 ms by reference length (doc 34 section 3, uncompiled windows) and its mean
from 69 ms to a similar range; `uncaptured_fresh_frames` near zero on the cold runner
and `fallback_counts` empty on the window runner; window replays about 2 to 4 per
reference bootstrap (the binary split of its width). Boot time not longer than A by
more than the 28 window captures, since no new shape compiles and the precompile fix
returns about 15 s from the warm runner; footprint about twice doc 34's 1.68 GB for
buckets 1 and 4. Not expected on B: the preprocessing segment, about 42 ms above main
on early ids, which is doc 32's item 3. So B's first chunk should land roughly halfway
between early ids and main, not within 10 ms of main; that gate needs item 3 as well.

## Step 3, the pair for the default launch (2 boots)

A: upstream main 3060470a8. B: 8b66f41d4. Same reads. Expected on B: the bootstrap
segment at ahead 0 from about 30 ms to 10 to 27 ms, req/s not below A.

## Step 4, the origin check, Nsight on step 2's two arms (1 window each)

The doc 29 default backend protocol, full preconditioning pass, 20 s window in pass 2,
`--gpu-metrics-devices=0 --gpu-metrics-frequency=20000`, SQLite export, `nsys_threads.py`.
Read per thread: launches over decode done syncs on the initial worker (launches per
bootstrap, about 860 on A), lock wait per bootstrap, the preprocessing workers' and the
scheduler's lock wait, GR and SM active. The slice reached the origin only if the
initial worker's launches per bootstrap fell to tens and the other threads' lock waits
fell with them; if launches fell and waits did not, report it as a finding about what
the waits are on.

## Step 5, quality, on step 3's B, pass 2

WER and speaker similarity against the bands (WER about 1.0 percent, similarity about
71.2). A seeded streaming c1 pass on A and B, identity count reported not gated: window
boundaries change the reduction order inside the decoder's convolutions and attention.

Archive per boot: head, import path, server command, gpus before, dmon log, launch to
ready time, both passes' speed_results and client logs, every event file of pass 2,
serve.log, the Nsight SQLite and `nsys_threads.py` JSON for step 4, plus step 1's output.
