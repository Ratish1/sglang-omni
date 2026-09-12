# 31. Runbook: the reference prefix prime (P1 of doc 30)

Branch `perf/qwen3-tts-prefix-prime` at 651de81e3, two commits on upstream main d90d71c37.
One session, plain server command with no environment variable in front of it, physical
GPU 0, GPUs 1 to 3 recorded before every boot, `nvidia-smi dmon -i 0 -s pucv -d 1` logging
on every boot. Every pass is the full 1088 request corpus, warmup 1, no seed. Check the
decode log line gap on the first boot: about 0.6 s per 40 steps, or stop.

## Step 0, unit tests on the box

```bash
git fetch origin perf/qwen3-tts-prefix-prime && git worktree add tmp/p1 651de81e3
cd tmp/p1 && python -m pytest tests/unit_test/qwen3_tts/test_pipeline.py -k "prefix or prime or stateful_codec or stream_output or bootstrap" -q
python -m pytest tests/unit_test/pipeline/test_scheduler.py -k "enqueue_built_request or stream_output" -q
python -m pytest tests/unit_test/qwen3_tts/test_incremental_codec.py -q
```

All must pass before a boot. Archive the output.

## Step 1, the pair on main (2 boots)

A: upstream main d90d71c37 from a worktree. B: 651de81e3. Streaming c16, two passes each,
event recorder in pass 2 (`enable_torch` false, stopped after 200 completions), the
anatomy script on both event dirs. Read: first audio at zero requests ahead, code chunks
received before first audio, preprocessing segment, first chunk mean and p99, req/s,
inter chunk, and the dmon utilization mean during traffic. Expected on B: the zero ahead
bootstrap well under control's 27 ms, chunks before first audio back to 1, first chunk
mean below control's.

## Step 2, the pair on early ids (2 boots)

A: d90d71c37 plus `tasks/qwen3_tts_e4_investigation_20260912/early_ids.patch`. B: 651de81e3
plus the same patch (it applies to model_runner.py only, which P1 does not touch). Same
protocol and reads as step 1. This is the pair that decides #2123: the target is B's first
chunk mean within 10 ms of step 1's main control at the same throughput as early ids.

## Step 3, quality, on the step 1 B boot's pass 2 outputs

WER and speaker similarity on the c16 pass WAVs, against the bands in the #2123 body
(WER about 1.0 percent, similarity about 71.2). A seeded c1 pass on A and B of step 1 is
also run, but its identity count is reported, not gated: the bootstrap is now two decoder
partitions, so Base voice clone WAVs differ within the decoder's partition tolerance.

## Step 4, Nsight on step 2's B (1 boot, only if step 2 passes its gate)

The E5 default backend protocol: full preconditioning pass, 20 s window in pass 2 with
`--gpu-metrics-devices=0 --gpu-metrics-frequency=20000` on `nsys start`, SQLite export,
`nsys_threads.py` on it. Read the initial worker's lock wait per bootstrap (48 ms on early
ids), the cold graph runner's replay and miss counters in the serve log, and GR and SM
active.

Archive per boot: head, import path, server command, gpus before, dmon log, both passes'
speed_results and client logs, the event dir, serve.log; plus the test output of step 0.
