# 43. Runbook: reference encoder graphs, tests, census and the profiling script comparison

Branch `perf/qwen3-tts-reference-encoder-graphs`, head c52eef547, two commits on upstream
main 69ddc6baa (doc 42 is the plan). Worktree `tmp/re` for the branch, `tmp/main` at
69ddc6baa for the control, `tmp/an` for scripts. Same session rules as doc 33: module
launch, import path archived per boot, GPU 1, GPUs recorded before each boot, dmon on,
full corpus, warmup 1, no seed, zero failures.

## 1. Tests, before any boot

```bash
cd /sgl-workspace/sglang-omni && git fetch -q origin perf/qwen3-tts-reference-encoder-graphs
git worktree add tmp/re origin/perf/qwen3-tts-reference-encoder-graphs 2>/dev/null || (cd tmp/re && git checkout -q c52eef547)
cd tmp/re && PYTHONPATH=. python -m pytest tests/unit_test/qwen3_tts/test_reference_encoder_cuda_graph.py \
  tests/unit_test/qwen3_tts/test_pipeline.py -q 2>&1 | tail -15
```

Expected: all pass, including the two accelerator tests in the new file. A failure is
the finding; send the tail.

## 2. Census, default launch, main against the branch

Two boots, A from `tmp/main`, B from `tmp/re`, passes 1 and 2 at streaming c16 with the
event recorder on pass 2, then the seeded c1 pass, quality on B pass 2, exactly as doc
33 steps 1 to 3. The B gate, in `serve.log` before readiness:

```
Qwen3-TTS reference encoder graphs captured for [32, 48, 64, 96, 128, 192, 256] frames
```

A `disabled the runner` line on B voids the boot. Reads: req/s, TTFC mean, p50, p99,
the preprocessing segment p50 and p95 and the encode mode split from
`first_chunk_anatomy.py`, WER and speaker similarity on B. Then the same pair with
`early_ids.patch` applied on both arms, the #2123 gate: B's TTFC mean against A's
control from this session, within 10 ms at early ids throughput.

## 3. Nsight, one boot per arm at c16, and the two SM scripts on the same export

The teammate's `compute_sm_window.py` (branch `profie_cosy_workload`) reads the same
four GPU metric rows ours does and cuts its window from the client log's
`Benchmarking N requests` and `Results saved to` lines through
`TARGET_INFO_SESSION_START_TIME`. Ours averages every sample the export holds. To
compare them, one capture has to hold the whole pass:

```bash
# boot B under nsys launch as in doc 33 step 4, then, before the c16 pass starts:
nsys start --gpu-metrics-devices=1 --gpu-metrics-frequency=20000 -o /sgl-workspace/sglang-omni/tmp/re-full
# run the pass 2 client with its log to $S/B/pass2/client.log
nsys stop
nsys export --type sqlite -o /sgl-workspace/sglang-omni/tmp/re-full.sqlite /sgl-workspace/sglang-omni/tmp/re-full.nsys-rep
python tmp/an/tasks/perf_backlog/scripts/nsys_gpu_metrics.py /sgl-workspace/sglang-omni/tmp/re-full.sqlite | tee $S/B/gpu_metrics_ours.txt
python <profie_cosy_workload>/.claude/skills/model-profiling/cosyvoice3_default_nsys/compute_sm_window.py \
  /sgl-workspace/sglang-omni/tmp/re-full.sqlite $S/B/pass2/client.log | tee $S/B/gpu_metrics_theirs.txt
```

Reads: SM Issue, SMs Active, GR Active and Tensor Active from both. With the capture
spanning the pass, the two windows differ only by the samples before `Benchmarking`
and after `Results saved`, so the means should sit within a few tenths of a point. A
larger gap points at the timestamp conversion (their script subtracts the session's
UTC start from the log's local clock; on a host whose clock is not UTC the window
lands elsewhere) or at the metric ids (theirs are fixed at 2 to 5, ours are looked up
by name; check `TARGET_INFO_GPU_METRICS` if the labels disagree). Run their script a
second time on a 100 Hz capture of the same pass if the sampling rate is in question:
both are unweighted means, so the expectation is the same and only the noise differs.

Archive: per boot as in doc 33, plus `re-full.sqlite`, both metric text files and the
test tail.
