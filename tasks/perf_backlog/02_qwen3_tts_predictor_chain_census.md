# 02. Qwen3-TTS predictor chain: kernel census (T22)

Status: profiling step, no code. The tool is written and its attribution was
checked once against a trace's own step ledger. Every number for this item
comes from a fresh H100 trace on the profiling branch, nothing older is
used.

## 1. What this measures

Doc 22 section 2, from the step ledger: at c1 the decode step spends 7.1 ms
on the GPU, 2.2 ms of it in the backbone forward and 5.0 ms in the
predictor replay, the sampling and the collect. The predictor chain is 16
sequential sub-steps of a five layer forward at batch 1 to 16. At those
sizes the chain is bound by the number and latency of its kernels, not by
FLOPs, so the census asks: how many kernels per replay and per sub-step,
how long each family runs, how much of the replay is gaps between kernels,
and which kernels the fusions should target.

## 2. Branch

`perf/qwen3-tts-profiling`, pushed. It is the startup capture branch (PR
#1947, head `fff86552c`) plus the six ledger commits of `perf/step-ledger`
(the per step ledger, the p50 column, the eager prefill shape counter).
The capture phase timing commit of the ledger branch is left out, the
capture code it timed no longer exists. Serve from this branch so the
ledger's `forward_ms` and `gpu_span_ms` come with the trace.

## 3. Tool

`tasks/qwen3_omni_0518_numerics/scripts/perfkit.py`, standard library only.

```
python perfkit.py ingest TRACE.json.gz -o TRACE.pkl
python perfkit.py steps  TRACE.pkl [--rows N] [--json steps.json]
python perfkit.py census TRACE.pkl [--rows N] [--top 40] [--json census.json]
python perfkit.py diff   census_A.json census_B.json
```

- `ingest` streams the chrome trace once and keeps kernels, memcpys and
  the CUDA runtime calls with their correlation ids. Tens of seconds per
  GB.
- `steps` finds the scheduler thread (the one launching CUDA graphs),
  attributes every kernel to its launch by correlation id, classifies each
  graph launch as backbone or predictor by the kernels it contains, and
  builds one step per backbone launch. Per row count it prints the step
  wall, the backbone busy time, the eager sampling before the predictor,
  the predictor replay busy and wall, the kernels per replay, the staging
  and collect after it, and the idle time inside the step and inside the
  replay. Rows come from the int32 token staging copy after the replay.
- `census` takes the replays of one row count, splits each into sub-steps
  at the sampler kernel, and prints: kernels per replay, busy and wall,
  the kernel duration and inter kernel gap distributions, busy per kernel
  family, kernels per sub-step, the full kernel sequence of one middle
  sub-step with median durations, and the top kernels per replay by time.
- `diff` compares two census JSONs per family, for the A/B of a fusion.

The markers that recognise the predictor graph and the sub-step boundary
are the names of omni's Triton kernels (`_gather_codec_embedding_and_add_kernel`,
`_seeded_top_k_top_p_sample_kernel`, `_seeded_gumbel_sample_sorted_kernel`)
and can be overridden with `--predictor-marker` and `--substep-marker`.

The profiler's op observer is thread local and the scheduler thread does
not start it, so the trace has no Python frames for the step. The tool
does not need them: kernels launched by a graph replay carry the
correlation id of the `cudaGraphLaunch` call and the graph id, and every
graph id is one (bucket, signature).

## 4. Box protocol

One server, on GPU 2, from the profiling branch, default flags plus the
profiler directory. Do not set `SGLANG_TORCH_PROFILER_WITH_STACK`, the
default trace is tens of MB per window and the census needs no stacks.

```
git fetch origin && git checkout perf/qwen3-tts-profiling && git reset --hard origin/perf/qwen3-tts-profiling
export SGLANG_TORCH_PROFILER_DIR=$OUT/prof
CUDA_VISIBLE_DEVICES=2 sgl-omni serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml --port 31001 > $OUT/logs/serve.log 2>&1 &
# wait for /health, then one warmup request so the first window has no captures in it
```

Two windows, c1 then c16, each its own run id so each has its own trace:

```
for C in 1 16; do
  curl -s -X POST http://127.0.0.1:31001/start_profile -H 'Content-Type: application/json' \
    -d "{\"run_id\":\"census_c$C\",\"event_dir\":\"$OUT/census_c$C/events\",\"enable_torch\":true,\"trace_path_template\":\"$OUT/census_c$C/trace_{stage}\"}"
  python benchmarks/eval/benchmark_tts_seedtts.py --use-existing-server --meta zhaochenyang20/seed-tts-eval-arrow \
    --ref-format references --lang en --max-concurrency $C --max-samples $((C * 12)) --seed 1234 --warmup 0 \
    --generate-only --output-dir $OUT/census_c$C/bench
  curl -s -X POST http://127.0.0.1:31001/stop_profile -H 'Content-Type: application/json' -d "{\"run_id\":\"census_c$C\"}"
  sleep 20
done
```

12 samples per row of concurrency keeps each trace under a few hundred
MB and gives a few hundred decode steps per window. Then, per window, on
the tts_engine stage trace:

```
python tasks/qwen3_omni_0518_numerics/scripts/perfkit.py ingest $OUT/census_c1/trace_tts_engine_pid*_rank0.trace.json.gz -o $OUT/census_c1/tts_engine.pkl
python tasks/qwen3_omni_0518_numerics/scripts/perfkit.py steps  $OUT/census_c1/tts_engine.pkl --json $OUT/census_c1/steps.json | tee $OUT/census_c1/steps.md
python tasks/qwen3_omni_0518_numerics/scripts/perfkit.py census $OUT/census_c1/tts_engine.pkl --rows 1  --json $OUT/census_c1/census_rows1.json | tee $OUT/census_c1/census_rows1.md
python tasks/qwen3_omni_0518_numerics/scripts/perfkit.py census $OUT/census_c16/tts_engine.pkl --rows 16 --json $OUT/census_c16/census_rows16.json | tee $OUT/census_c16/census_rows16.md
```

Send back: the two `steps.md`, the two census `.md` and `.json`, the
`step_ledger_tts_engine_*.json` from each events directory, and
`serve.log`. The traces themselves can stay on the box.

## 5. What the census decides

- Kernels per sub-step by family, with the sequence of one sub-step: this
  names the fusion candidates and the launch count each one removes.
- The inter kernel gap and the per kernel duration distributions: if the
  median kernel runs a few microseconds and the gaps are near zero, the
  replay is latency bound and the gain of a fusion is close to the kernel
  count it removes times the median kernel time.
- The predictor busy time against the ledger's `gpu_span_ms - forward_ms`
  for the same rows: the two must agree within the sampling and collect
  share, which validates the attribution on this run.
- Rows 1 against rows 16: whether the chain's time is flat in the batch
  (latency bound) or grows (throughput bound), which decides whether the
  first slice targets kernel count or GEMM efficiency.

The plan for the fusions is written after this census, from its numbers.
