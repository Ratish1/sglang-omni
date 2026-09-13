# 33. Runbook: the vocoder bootstrap through captured graphs (doc 32 item 1)

Two branches on upstream main 3060470a8.

- `perf/qwen3-tts-codec-precompile` at 0ba531872, one commit: the decoder traces a
  compiled shape on the codes and state the graph runner warms and captures with, so
  each shape compiles once instead of twice, and raises Dynamo's cache entry limits the
  way sglang does before compiling per bucket. Bit exact; verified by doc 34's bench
  (12 graphs for 12 shapes, capture 100.7 to 55.3 s).
- `perf/qwen3-tts-bootstrap-graphs` at 05683d8a4, seven commits: the runner's window
  schedule and bucket queries (7e5e2aa48), the window runner and its knob (cb9cf8f46),
  the windowed decode path (1453ef538), no default until measured (fd54363d6), the
  precompile fix cherry-picked (c2212e459), the window runner compiling its rungs from
  the steady stride up at one bucket with a cohort counter (1498c2078), the measured
  default ladder 1 to 64 (05683d8a4).

Step 0 of the earlier version of this runbook is done: doc 34 holds the bench readout.
The steps below are the session that decides both PRs. Same session rules as doc 31:
plain server command, GPU 0, GPUs 1 to 3 recorded before every boot, dmon on every boot,
full corpus, warmup 1, no seed, two passes per arm, event recorder in pass 2 stopped
after 200 completions, decode log gap check on the first boot.

What the change does. A reference prefixed bootstrap (reference frames plus the first
chunk) used to be one eager decode of an uncaptured width, about 860 host launches and
about a thousand lock handoffs. The initial worker now owns a second graph runner, the
window runner, with widths 1, 2, 4, 8, 16, 32, 64 at one bucket, the rungs 8 and up
captured from the compiled decoder step. A same width cohort whose width no runner
captured is consumed as a sequence of those widths against its arena slots, largest
first, one replay per window. The knob is the vocoder factory argument
`incremental_codec_cuda_graph_window_frames`; an empty list turns windowing off. The
codec state line in the serve log carries `windowed_decodes: {cohorts, rows, replays}`
and a `window` runner entry next to `cold` and `warm`.

## Step 1, unit tests on the box, full files, on 05683d8a4

```bash
git fetch origin perf/qwen3-tts-bootstrap-graphs && git worktree add tmp/bw 05683d8a4
cd tmp/bw
python -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py -q
python -m pytest tests/unit_test/qwen3_tts/test_incremental_codec_cuda_graph.py -q
python -m pytest tests/unit_test/qwen3_tts/test_pipeline.py -q
```

Three tests in the graph file run only with CUDA and run here. All must pass before a
boot; archive the output. Every test on both branches is unrun until this step.

## Step 2, the pair that decides #2123 (2 boots)

A: upstream main 3060470a8 plus `tasks/qwen3_tts_e4_investigation_20260912/early_ids.patch`.
B: 05683d8a4 plus the same patch (it touches model_runner.py only). Streaming c16.

Reads per arm from `first_chunk_anatomy.py` on the pass 2 events plus the client
summary: TTFC mean and p99, req/s, inter chunk, preprocessing p50, the first frame to
first audio segment at ahead 0 and its mean, the prefill by overlap table, the cadence,
dmon GR active during traffic. From B's serve log: boot time from launch to ready
against A, the window runner's captured keys and footprint, the last codec state line's
`windowed_decodes` cohorts, rows and replays, the window runner's `replays`, and the
cold runner's `uncaptured_fresh_frames`.

Expected on B, stated before the run: the bootstrap segment at ahead 0 from about 52 ms
to 10 to 20 ms and its mean from 69 ms to a similar range; `uncaptured_fresh_frames`
near zero; cohorts equal to rows (one row per windowed cohort on this corpus); boot
time not longer than A by more than the window runner's capture (about 13 s of compile
for the three rungs above 8, minus the 15 s the precompile fix returns from the warm
runner). Not expected on B: the preprocessing segment, about 42 ms above main on early
ids, which is doc 32's item 3. So B's first chunk should land roughly halfway between
early ids and main, not within 10 ms of main; that gate needs item 3 as well.

## Step 3, the pair for the default launch (2 boots)

A: upstream main 3060470a8. B: 05683d8a4. Same reads. Expected on B: the bootstrap
segment at ahead 0 from about 30 ms to 10 to 20 ms, req/s not below A.

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
boundaries and the compiled kernels change reduction order inside the decoder.

Archive per boot: head, import path, server command, gpus before, dmon log, launch to
ready time, both passes' speed_results and client logs, every event file of pass 2,
serve.log, the Nsight SQLite and `nsys_threads.py` JSON for step 4, plus step 1's output.
