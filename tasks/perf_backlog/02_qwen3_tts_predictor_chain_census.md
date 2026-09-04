# 02. Qwen3-TTS predictor chain: kernel census (T22)

Status: census and timeline done (sections 6 and 8), the plan is
`03_qwen3_tts_predictor_chain_plan.md`. Every number here comes
from the H100 run of 2026-09-04 on the profiling branch.

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

## 6. Results, run of 2026-09-04

Source: `artifacts/qwen3-tts-predictor-census-results.tar.gz`, branch head
`4970be647`. c1 window 12 samples, 649 decode steps. c16 window 192
samples, 803 decode steps, 47 at 16 rows. The tool's step wall equals the
ledger's cycle p50 within 10 us at both row counts (8.962 against 8.952
ms at 1 row, 9.434 against 9.425 at 16), so the attribution is right on
this run.

| | 1 row | 16 rows |
| --- | ---: | ---: |
| step wall p50 | 8.962 ms | 9.434 ms |
| device busy in the step | 5.843 ms | 6.620 ms |
| device idle in the step | 3.119 ms (35%) | 2.815 ms (30%) |
| backbone replay busy | 1.793 ms | 2.014 ms |
| eager sampling before the predictor | 0.074 ms | 0.083 ms |
| predictor replay busy | 3.932 ms | 4.464 ms |
| predictor replay wall | 4.740 ms | 5.282 ms |
| kernels per replay | 1371 | 1371 |
| kernel duration p50 | 1.70 us | 1.95 us |
| gap between kernels p50 | 0.45 us | 0.45 us |
| staging and collect | 0.044 ms | 0.059 ms |

The profiler itself costs about 1 ms of host time per step: the ledger run
without it (doc 22 section 2) had a 7.84 ms cycle at 1 row.

### 6.1 The replay is bound by kernel count

Busy time grows 13.5% from 1 row to 16 while the work grows 16 times, and
the replay wall is the sum of 1371 kernel durations plus 1370 gaps of
0.45 us (0.62 ms of the 0.81 ms difference between wall and busy). There
is no host launch inside a replay, the whole chain is one
`cudaGraphLaunch`, so the only way down is fewer kernels and cheaper
kernels.

Families per replay at 1 row: GEMM 432 kernels 2.073 ms (53%), attention
256 kernels 0.517 ms (176 of those are the RMSNorm Triton kernel the
family regex misread, fixed in the tool), elementwise 321 kernels 0.503
ms, sampling 30 kernels 0.366 ms, embedding 76 kernels 0.130 ms, norm 80
kernels 0.115 ms, rope 80 kernels 0.109 ms, activation 80 kernels 0.104
ms.

### 6.2 One sub-step, 86 kernels

Sixteen sub-steps: the first has 162 kernels (the talker hidden enters
through the projection and flash attention at cache length 0), fourteen
have 86, the last has 5. The 86 of a middle sub-step, in order:

- 6 to enter: the argmax reduce of the previous sub-step's logits (4.1 us
  at 1 row, 9.9 us at 16), the `where` that selects sampled against argmax
  rows, a 32 byte copy of the codes, the fused codec embedding gather and
  add, the projection GEMM with bias from the talker width to the
  predictor width (5.9 us).
- 5 layers of 16: RMSNorm, qkv GEMM (5.2 us), fused qk norm, fused rope,
  two elementwise kernels writing k and v into the private cache (1.9 and
  2.1 us), the cuDNN attention kernel (2.9 us), the o_proj GEMM with the
  residual in its epilogue (5.4 us), RMSNorm, gate_up GEMM (6.5 us),
  act_and_mul, down GEMM as split K (5.1 us) plus its reduce (1.6 us),
  the residual add.
- 9 to leave: final RMSNorm, head GEMM (4.4 us), four `index_select` of
  the sampling parameters by row and three elementwise kernels for the
  sub positions (the sampler's prologue, 8 kernels, about 12 us), the
  fused seeded top k top p sampler (20.6 us, the largest kernel in the
  chain).

### 6.3 Candidates, sized from the counts

Each removed kernel saves its duration plus the 0.45 us gap. Per replay
at 1 row:

| Candidate | Kernels removed | Busy removed | Note |
| --- | ---: | ---: | --- |
| Sampler prologue: the four index selects are identity gathers on the graph path, the sub positions are one table per step | 128 | about 0.20 ms | omni code only |
| All rows sample: skip the argmax, the where and the copy when the signature says so | 48 | about 0.16 ms | a signature term, omni code only |
| Rope, qk norm and the two cache writes as one kernel per layer | 240 | about 0.30 ms | one Triton kernel, omni owned |
| Residual add fused into the next RMSNorm | 80 | about 0.10 ms | sglang's fused_add_rmsnorm exists |
| Down projection without split K reduce | 80 | about 0.13 ms | algorithm choice, may cost GEMM time, measure |
| The sampler kernel itself, 20.6 us for a 2048 wide row | 0 | up to 0.20 ms | kernel work |

About 1.0 ms of the 4.74 ms replay wall from kernel count alone, 21%,
before any GEMM work. The GEMMs are 432 calls at 5 to 6.5 us each moving
4 to 12 MB of weights per call, 40% of H100 bandwidth, 2.07 ms against a
0.75 ms bandwidth floor per replay: the second half of the item, a GEMV
path for M up to 16, sized after the first half lands.

### 6.4 The host side is a second item

The GPU is idle 3.1 ms of every 9.0 ms step at 1 row and 2.8 ms of 9.4 at
16, about 2 ms of it after the profiler's own cost is taken out. The
ledger agrees: gpu span 8.49 ms against 5.84 ms of kernel time. Where in
the step the host is late is what section 7 measures.

## 7. Next box step: the timeline of one step

On the existing pickles, seconds each, no new profile:

```
python tasks/qwen3_omni_0518_numerics/scripts/perfkit.py timeline $OUT/census_c1/tts_engine.pkl --rows 1 | tee $OUT/census_c1/timeline_rows1.md
python tasks/qwen3_omni_0518_numerics/scripts/perfkit.py timeline $OUT/census_c16/tts_engine.pkl --rows 16 | tee $OUT/census_c16/timeline_rows16.md
```

It prints, for the median step of that row count, every host call on the
scheduler thread with its time from the step start, the device span its
kernels occupied, and the device idle before each span. That is the line
of code launches against the line of kernels, and it names the host work
the GPU waits on. The fusion plan and the host plan are written from
these two tables.

## 8. The timeline, read

Source: `timeline_rows1.md` and `timeline_rows16.md` of the same run, the
median wall step of each row count. Times from the step's backbone launch.

| | 1 row | 16 rows |
| --- | ---: | ---: |
| backbone launch call on the host | 0.518 ms | 0.381 ms |
| device idle until the backbone kernels start | 0.521 ms | 0.384 ms |
| backbone kernels | 0.521 to 2.520 | 0.384 to 2.603 |
| eager sampling and predictor input copies, about 30 launches | host 0.60 to 1.31, device 2.53 to 2.64 | host 0.47 to 1.21, device 2.61 to 2.73 |
| predictor launch call on the host | 1.801 ms, from 1.313 | 1.222 ms, from 1.215 |
| device idle until the predictor kernels start | 0.482 ms | 0.005 ms |
| predictor kernels | 3.118 to 7.857 | 2.735 to 8.031 |
| host silent, waiting on the token copy | 3.23 to 7.90 | 2.55 to 8.08 |
| tail: 35 to 45 launches and copies, device busy 40 to 60 us | 7.86 to 8.96, 1.10 ms | 8.03 to 9.43, 1.40 ms |

Three readings.

- The eager sampling is not on the critical path. The host issues it
  while the backbone runs and the device executes it in 0.11 ms right
  after.
- The two graph launch calls cost the host 0.5 and 1.8 ms at 1 row, and
  the device idles for exactly the time the call takes to submit the
  nodes. The profiler instruments every node it submits, and the ledger
  run without the profiler had a 7.84 ms cycle against 8.95 here, so most
  of that 1.0 ms of idle is the profiler's. How much a 1371 node launch
  costs without it is not known from this run: measured by B3 of doc 03.
  Whatever it is, it scales with the node count that the fusions cut.
- The tail after the predictor is real and host bound: 1.1 ms at 1 row
  and 1.4 ms at 16, for 40 to 60 us of device work, as 35 to 45 launches
  each separated by 10 to 170 us of host time. From the code on that path:
  the token staging copy and its wait, the two output clones of
  `post_process_outputs`, the next step's `prepare_decode_buffers` with
  six `torch.tensor(list)` uploads (sglang_model.py, one per sampling
  parameter), `_write_feedback_buffers` with one launch per row inside a
  Python loop (the run of 16 launches 12 us apart at 16 rows,
  model_runner.py), the forward batch build and the graph input copies of
  sglang's runner, then the backbone launch. The per row loop alone is
  0.19 ms at 16 rows and grows with the batch.
