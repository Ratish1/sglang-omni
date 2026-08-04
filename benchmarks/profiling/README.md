# H100 CPU-saturation profiling runbook

This runbook collects causal evidence for the Fun-ASR throughput collapse
without using the router. Run every condition on the same H100 host, model
revision, corpus, server arguments, CPU policy, and dependency environment.

The harness performs target-concurrency warmup until throughput and mean
latency are stable, rejects incomplete ASR work by default, records the
environment, and validates acknowledged profiler artifacts.

## 1. Prepare the checkout and dataset

```bash
git fetch origin debug/problem
git switch debug/problem
uv pip install -e .
python -m benchmarks.dataset.prepare --dataset seedtts
```

Record tool capabilities before changing placement:

```bash
lscpu -e=CPU,CORE,SOCKET,NODE,ONLINE,MAXMHZ,MINMHZ
nvidia-smi topo -m
cat /sys/devices/system/cpu/smt/control
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/cpuset.cpus.effective
perf --version
turbostat --version
nsys --version
```

Do not copy CPU lists from another machine. Derive server physical cores,
their SMT siblings, the GPU-local NUMA node, and interferer cores from this
host's topology.

## 2. Start one direct Fun-ASR server

Use a new server process for every trial:

```bash
export SGLANG_TORCH_PROFILER_DIR="$PWD/artifacts/cpu-saturation/server"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

sgl-omni serve \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
  --host 127.0.0.1 \
  --port 8000 \
  --stages.asr.factory_args.pre_lm_cache_max_entries=0 \
  --stages.asr.factory_args.pre_lm_cache_size_bytes=0
```

Wait for model loading, compile warmup, and CUDA graph capture to finish. The
benchmark performs a second workload-level stability warmup.

The cache overrides are part of the experimental contract. Without them,
warmup or the adjacent comparison can populate the audio-embedding cache and
silently turn the captured pass into a cache-hit workload. Profiled modes
reject `pre_lm_cache_hit` events unless `--allow-cache-hits` is explicitly
set.

For baseline `perf` runs, identify the `asr` stage process rather than the HTTP
coordinator:

```bash
ps -eLo pid,ppid,tid,psr,comm,args | grep -E 'sgl|scheduler-asr|fun-asr'
```

Torch/event/Nsight runs can obtain the stage PID from the acknowledged start
manifest for measured collectors. Still pass `--server-pid ASR_STAGE_PID` in
every mode so warmup stability also includes stage CPU-ms/request, not only
QPS and p50 latency.

## 3. Quiet unprofiled baseline

Run one harness invocation per fresh server. Use at least five restarts:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode baseline \
  --run-id quiet-r1 \
  --host 127.0.0.1 \
  --port 8000 \
  --concurrency 32 \
  --max-samples 1088 \
  --server-pid ASR_STAGE_PID \
  --collectors psi,perf-stat,turbostat \
  --turbostat-cpus SERVER_CPU_LIST
```

Restart the server and repeat as `quiet-r2` through `quiet-r5`. Alternate
conditions as A/B/B/A instead of running every quiet trial first.

For a service-capacity curve, repeat with finite open-loop offered rates:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode baseline \
  --run-id open-120-r1 \
  --concurrency 64 \
  --request-rate 120 \
  --max-samples 1088 \
  --server-pid ASR_STAGE_PID \
  --collectors psi,perf-stat,turbostat
```

Sweep rates below and above the quiet completion rate. Offered rate, completed
rate, errors, queue phases, and in-service phases must be interpreted
together.

Collect detailed PMU events in a separate restart so multiplexing does not
contaminate the low-overhead pass:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode baseline \
  --run-id pmu-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --server-pid ASR_STAGE_PID \
  --collectors psi,perf-stat \
  --perf-events \
    cycles,instructions,cache-references,cache-misses,branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend
```

Check `perf_stat.csv` and its ignored rows: unsupported or heavily multiplexed
events are not evidence. Use model-specific raw events only after recording
the CPU model and `perf list` output.

## 4. Event-only semantic timeline

This is the lowest-overhead attribution run:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode events \
  --run-id events-quiet-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --profile-samples 256 \
  --server-pid ASR_STAGE_PID \
  --collectors psi,perf-stat,turbostat
```

The output directory contains:

- `manifest.json`: software, topology, cgroup, policy, corpus, and command;
- `warmup.json`: stability windows;
- `adjacent_baseline.json`: unprofiled comparison immediately before capture;
- `events/*.jsonl`: request/build/encoder/scheduler phase events with PID/TID;
- `perf_stat.csv`, `turbostat.txt`, and `system.json`;
- `profile_start.json` and `profile_stop.json`: acknowledgements;
- `result.json`: performance, perturbation, and integrity result.

Reject a run with failed requests, dropped events, missing stage
acknowledgements, unstable warmup, or unexpected encoder-cache hits. The
profiled and adjacent passes use the same duration-stratified sample subset;
`--profile-samples 0` opts into the full corpus.

## 5. Bounded PyTorch operator trace

Run torch profiling separately from Nsight and detailed PMU passes:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode torch \
  --run-id torch-quiet-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --profile-samples 256 \
  --server-pid ASR_STAGE_PID \
  --collectors psi \
  --torch-wait-steps 1 \
  --torch-warmup-steps 1 \
  --torch-active-steps 20
```

The profiler starts, steps, and stops on `scheduler-asr`. The stop endpoint
flushes an in-flight async decode step before export. The harness rejects a
trace missing the scheduler-owner canary, required CUDA activity, scheduled
steps, rank identity, event files, or finalization acknowledgement.

Do not use the profiled run as the throughput baseline. `result.json` reports
the adjacent-baseline perturbation. If the QPS loss exceeds 5%, use the trace
only for operator attribution.

Enable expensive flags only in a second short run:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode torch \
  --run-id torch-stacks-r1 \
  --concurrency 32 \
  --max-samples 256 \
  --collectors psi \
  --torch-active-steps 8 \
  --with-stack \
  --record-shapes
```

## 6. Nsight Systems joint CPU/CUDA trace

Nsight must launch the server so it follows the worker process tree. The
server emits one scheduler-owned NVTX range named
`sglang_omni.capture_window`; the harness opens it only after stability
warmup and closes it after the measured requests complete.

```bash
export SGLANG_OMNI_NVTX_RANGES=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --capture-range=nvtx \
  --nvtx-capture=sglang_omni.capture_window \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$PWD/artifacts/cpu-saturation/nsys-quiet-r1" \
  sgl-omni serve \
    --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
    --host 127.0.0.1 \
    --port 8000 \
    --stages.asr.factory_args.pre_lm_cache_max_entries=0 \
    --stages.asr.factory_args.pre_lm_cache_size_bytes=0
```

From another shell:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode nsys \
  --run-id nsys-quiet-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --profile-samples 256 \
  --server-pid ASR_STAGE_PID \
  --collectors psi \
  --nsys-report \
    "$PWD/artifacts/cpu-saturation/nsys-quiet-r1.nsys-rep"
```

Inspect CPU scheduling, CUDA API launches, kernels, streams, and GPU feed gaps
inside the capture range. A CPU/NVTX range represents host enqueue time; do not
infer device completion from visual nesting.

## 7. Scheduler-delay run

`perf sched` is intentionally separate from torch and Nsight:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode events \
  --run-id sched-load-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --server-pid ASR_STAGE_PID \
  --collectors psi,perf-sched

perf sched timehist \
  -i artifacts/cpu-saturation/sched-load-r1/perf_sched.data \
  > artifacts/cpu-saturation/sched-load-r1/perf_sched_timehist.txt
```

Filter the report using native TIDs recorded in `events/*.jsonl` for:

- `scheduler-asr`;
- active `omni-request-build_*` workers;
- `fun-asr-audio-encode`.

The relevant signal is runnable-to-running scheduler delay, not aggregate CPU
utilization alone.

## 8. Mechanism-isolating conditions

Repeat quiet baseline, event-only, `perf stat`, actual frequency, and scheduler
delay for these placements. Use the same corpus and randomized A/B/B/A order:

| ID | Server | Interferer | Isolates |
|---|---|---|---|
| A | unpinned | none | quiet reference |
| B | pinned physical cores | none | pinning alone |
| C | unpinned | compute loops | reported loss |
| D | pinned | server SMT siblings | SMT contention |
| E | pinned, siblings idle | other cores, same socket | package/cache/frequency |
| F | pinned, siblings idle | other socket | socket/NUMA |
| G | pinned, siblings idle | memory bandwidth load | cache/memory pressure |
| H | one logical CPU/core | matched load | SMT validation |
| I | controlled governor/boost | matched load | DVFS causality |
| J | CPU and memory NUMA-bound | matched load | remote-memory effect |
| K | OMP/MKL/PyTorch thread sweep | none and load | oversubscription |

Use `taskset` or cgroup cpusets only after mapping physical cores and siblings.
For NUMA conditions bind both CPU and memory placement. Record IRQ placement and
the interferer's command, affinity, start/stop timestamps, and counters.

## 9. Evidence required for a conclusion

- **DVFS:** stable instructions/request and IPC, negligible scheduler delay,
  actual busy MHz falling in proportion to lost service rate, and a controlled
  frequency condition reproducing/removing the gap.
- **SMT/cache:** sibling placement hurts at comparable actual MHz while IPC,
  backend stalls, or cache behavior worsens; sibling isolation recovers it.
- **Scheduler delay:** critical TIDs show materially increased
  runnable-to-running delay, migrations, or preemption.
- **Extra work:** instructions/request or semantic call/batch counts increase
  for identical inputs.
- **Serial host dispatch:** request-build/pre-LM/scheduler duration predicts
  GPU feed gaps even after isolation and frequency control.
- **Queueing:** offered rate exceeds completion and queue wait grows while
  in-service phase cost remains stable.
- **Request-build head-of-line blocking:** the
  `scheduler_request_build_hol_blocked` event shows completed later futures
  held behind the first incomplete future for material time.

Only after one mechanism has positive evidence and alternatives have rejection
evidence should a runtime change be implemented. CPU isolation remains a valid
CI control even if it is not the application root cause.

## 10. H100 code gate before experiments

Run the profiler-specific tests inside the CUDA container before collecting
evidence:

```bash
python -m pytest -q \
  tests/unit_test/profiler/test_event_recorder.py \
  tests/unit_test/profiler/test_integrity.py \
  tests/unit_test/profiler/test_profiler_control_client.py \
  tests/unit_test/profiler/test_profiler_protocol.py \
  tests/unit_test/profiler/test_scheduler_profiler_control.py \
  tests/unit_test/profiler/test_torch_profiler_schedule.py \
  tests/unit_test/benchmarks/test_benchmark_runner_rate.py \
  tests/unit_test/benchmarks/test_system_collectors.py \
  tests/unit_test/fun_asr/test_encoder_service.py \
  tests/unit_test/fun_asr/test_request_builders.py
```

Then run one 32-request torch smoke capture (`--profile-samples 32`,
`--torch-active-steps 4`) and require a passing `result.json` integrity gate
before starting the multi-restart matrix.
