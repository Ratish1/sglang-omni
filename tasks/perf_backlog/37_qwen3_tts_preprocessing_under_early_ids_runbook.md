# 37. Runbook: the preprocessing segment under early ids (doc 32 item 3, the reads)

Run 2026-09-14, archive `pp-session-results.zip`, readout in doc 38. The lock held
fraction this runbook asked for is not in a `--gil` record (doc 38 section 3); the
`gil_share.py` call below no longer takes `--rate` and `--duration`.

Doc 36 left one regression: with PR #2151, early ids still costs 23.5 ms of TTFC mean
over the PR alone (117.0 to 140.5 ms) and the whole of it is the preprocessing segment,
36.3 to 57.8 ms p50, 100.6 to 143.2 p95. Two mechanisms can produce that and the session
2 data cannot separate them: the interpreter lock (the scheduler no longer parks, the
preprocessing worker launches about 580 kernels per request, each launch a lock handoff)
and load (early ids runs 14 percent more requests through the same pool, so part of the
segment is queueing). This session measures both directly, with two boots, no Nsight.

Both arms are the PR head, so this is not an A/B of the PR. P is dbef8d539 as is. PE is
dbef8d539 plus `tasks/qwen3_tts_e4_investigation_20260912/early_ids.patch` (applies
cleanly on dbef8d539, checked). Same session rules as doc 33: `python -m
sglang_omni.cli serve` from the worktree, `import_path.txt` before the boot, the window
gate on both boots (both run the branch), one GPU for both, GPUs recorded before each
boot, dmon on, full corpus, warmup 1, no seed, streaming c16, decode log gap check on the
first boot. Scripts from the analysis worktree (`tmp/an`, pull it first for
`scripts/gil_share.py`). `py-spy` must be installed in the venv (`pip install py-spy`),
and it needs the stage process pid, printed at boot as `spawned 1 process(es) (pids=[N])`.

## Boot P, the PR alone, three passes

1. Pass 1, unprofiled.
2. Pass 2, event recorder on (`enable_torch` false), stopped after 200 completions. The
   anatomy read.
3. Pass 3, the lock read: the same client, and 15 s after it starts:

```bash
py-spy record --pid $STAGE_PID --gil --threads --nonblocking --rate 250 --duration 60 \
  --format raw -o $S/$ARM/gil_raw.txt
python tmp/an/tasks/perf_backlog/scripts/gil_share.py $S/$ARM/gil_raw.txt \
  | tee $S/$ARM/gil_share.txt
```

`--gil` keeps only samples of the thread holding the lock, so each thread's sample count
is its share of lock ownership; the threads are named (`scheduler-tts_engine`,
`qwen3-tts-vocoder-initial`, `qwen3-tts-vocoder-followup-N`, `qwen3-tts-ref-code`, the
reference encoder batch worker, the preprocessing workers). `--nonblocking` samples
without pausing the process. Pass 3 is not quoted for latency; the sampler has a cost.

## Boot PE, the PR plus early ids, four passes

Passes 1 to 3 as on P. Then:

4. Pass 4, the load matched read: the same client with `--request-rate 15.8`, the PR
   alone's pass 2 throughput, so the arrivals are Poisson at that rate under the same c16
   cap. Event recorder on with its own run id and event dir, stopped after 200
   completions.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts --generate-only --use-existing-server --stream \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --meta zhaochenyang20/seed-tts-eval-arrow \
  --ref-format references --lang en --warmup 1 --concurrency 16 --request-rate 15.8 \
  --port 31001 --output-dir $S/PE/pass4/bench 2>&1 | tee $S/PE/pass4/client.log
```

## Reads

- Anatomy on P pass 2, PE pass 2 and PE pass 4 (`first_chunk_anatomy.py`), the
  preprocessing segment p50 and p95, TTFC mean, req/s.
- `gil_share.txt` on P and PE: the scheduler's share, the preprocessing workers' and the
  reference encoder's shares, the vocoder workers' shares, and the lock held fraction.

Expected, stated before the run. If the lock is the mechanism: PE pass 4 keeps most of
the preprocessing delta at matched load (the segment stays above about 50 ms p50), the
scheduler's lock share on PE rises by roughly the fraction its park used to leave free,
and the preprocessing workers' shares fall while their segment grows. If load is the
mechanism: PE pass 4's preprocessing segment returns near P's 36 ms, and the lock shares
of P and PE look alike. Either way the next slice is fewer launches on the preprocessing
thread; these reads say how much of the 21 ms that can recover and whether the
scheduler's own launches (doc 32 item 2) have to be part of it.

Archive per boot as in doc 33, plus `gil_raw.txt`, `gil_share.txt` and, on PE, the
pass 4 bench, client log and events.
