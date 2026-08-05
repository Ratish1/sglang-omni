# H100 CPU-saturation profiling runbook

This runbook collects causal evidence for the Fun-ASR throughput collapse
without using the router. Run every condition on the same H100 host, model
revision, corpus, server arguments, CPU policy, and dependency environment.

The direct server is intentional: it isolates the model-serving critical path.
It cannot reproduce router overhead or prove that the router is innocent.
Only add a router replay after the direct path is localized; do not use router
traffic as the first trace.

The harness performs a full corpus/shape warmup followed by independent
stability windows, discovers the ASR stage PID through an acknowledged
start/stop handshake, rejects incomplete ASR work, records request identities
and immutable inputs, and validates the event state machine and finalized
artifacts. Every profiled pass is bracketed by matched unprofiled passes, and
CPU PSI is captured around all three windows. An exited command is not
automatically an accepted capture.

## One-shot H100 campaign

This is the first command to run after the code gate. It owns every server
restart, alternates quiet and 64-process CPU-stress trials in A/B/B/A order,
and executes exactly five fresh-server trials per condition.

Before each server starts, the campaign requires two consecutive five-second
execution-cgroup CPU PSI windows at or below 2%. It waits for up to five
minutes and persists both cgroup and host-wide PSI in `host_preflight.json`.
The cgroup is the acceptance signal because the server inherits it; host-wide
PSI is evidence but is not a hard gate on a shared machine where unrelated
tenants may keep it elevated. This preflight runs before the campaign's own
server or CPU interferer exists, so local stress is not mistaken for ambient
contamination.

The checked campaign uses client concurrency 32 independently of the 64 CPU
interferer processes. Fun-ASR has 16 pending request-build slots plus a
16-request backlog under its default server configuration, and its documented
single-worker results complete without shedding through concurrency 32.
Concurrency 64 can overflow that admission boundary—especially under the
stressed condition—and turns the comparison into a rejection experiment.
Do not raise `max_queued_requests` in this primary campaign: a queue-lifted
concurrency-64 run is a separate admission/queueing experiment and must not
replace the matched zero-failure attribution matrix.

```bash
MODEL_REVISION="$(
  python - <<'PY'
from huggingface_hub import model_info
print(model_info("FunAudioLLM/Fun-ASR-Nano-2512-hf").sha)
PY
)"
test -n "$MODEL_REVISION"

cp benchmarks/profiling/campaign.events.h100.example.json \
  /tmp/campaign.events.h100.json
python - "$MODEL_REVISION" /tmp/campaign.events.h100.json <<'PY'
import json
import sys
from pathlib import Path

revision, raw_path = sys.argv[1:]
path = Path(raw_path)
config = json.loads(path.read_text())
index = config["harness_args"].index("REPLACE_WITH_IMMUTABLE_HF_COMMIT_SHA")
config["harness_args"][index] = revision
path.write_text(json.dumps(config, indent=2) + "\n")
PY

python -m benchmarks.profiling.run_cpu_saturation_campaign \
  --config /tmp/campaign.events.h100.json \
  --output-dir "$PWD/artifacts/cpu-saturation/events-ab-$(date -u +%Y%m%dT%H%M%SZ)"
```

Before running it, inspect the JSON and change only machine facts that are
actually different (GPU index, port, corpus size, or server flags). Do not
insert CPU binding into this first campaign: its purpose is to reproduce and
localize the unbound quiet-versus-64-loop delta. The harness resolves the ASR
worker PID; never substitute the coordinator PID.

Here, “64-loop” means the 64-process CPU interferer, not client concurrency.
If concurrency 32 does not complete 100% on a different model or server
configuration, first run an unprofiled concurrency sweep and use the highest
shared zero-rejection level that still reaches the throughput plateau. Record
that deviation and use the same level for every condition.

The campaign is accepted only when all ten trials complete, every
`result.json` has `accepted=true`, `integrity.valid=true`,
`request_lifecycle_integrity.valid=true`, and
`system_integrity.valid=true`, the two unprofiled controls drift by no more
than 2%, and the profiled QPS differs from their midpoint by no more than 2%.
If control drift exceeds 2%, the run is classified as inconclusive rather
than blaming the event recorder. Campaign throughput and latency are taken
from the midpoint of the unprofiled controls; event-enabled results are used
only for causal phase metrics. `summary.json` reports per-condition values,
medians, MAD, and seeded bootstrap 95% intervals. Preserve the entire campaign
directory.

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
export SGLANG_OMNI_EVENT_DEFER_WRITES=1
export SGLANG_OMNI_EVENT_QUEUE_CAPACITY=131072
export SGLANG_OMNI_EVENT_FINALIZE_TIMEOUT_S=180

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

Deferred event writes are also part of the bounded 256-request attribution
contract. Producers enqueue timestamped records during measurement; JSON
serialization, filesystem writes, fsync, rename, and checksumming occur at
stop. If 131,072 records are insufficient, the run is rejected for drops
rather than silently switching to a partial trace. Do not use deferred mode
for an unbounded production capture.

For manual inspection, distinguish the `asr` stage process from the HTTP
coordinator:

```bash
ps -eLo pid,ppid,tid,psr,comm,args | grep -E 'sgl|scheduler-asr|fun-asr'
```

When `--server-pid` is omitted, every mode performs a short acknowledged
event-recorder start/stop before warmup and requires exactly one target PID for
the requested stage. That discovered PID is used for warmup CPU-ms/request and
all measured collectors. You may pass `--server-pid` manually, but profiled
modes reject it if it disagrees with the later stage acknowledgement.

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
  --model-revision "$MODEL_REVISION" \
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
  --model-revision "$MODEL_REVISION" \
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
  --model-revision "$MODEL_REVISION" \
  --collectors psi,perf-stat \
  --perf-events \
    cycles,instructions,cache-references,cache-misses,branches,branch-misses,stalled-cycles-frontend,stalled-cycles-backend
```

Check `perf_stat.csv` and its ignored rows: unsupported or heavily multiplexed
events are not evidence. Use model-specific raw events only after recording
the CPU model and `perf list` output. If the default `perf` wrapper does not
match the host kernel, pass the working executable explicitly, for example
`--perf-binary /usr/lib/linux-tools/$(uname -r)/perf`; the exact path is
recorded in the manifest and collector result.

## 4. Event-only semantic timeline

This is the lowest-overhead attribution run:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode events \
  --run-id events-quiet-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --profile-samples 256 \
  --model-revision "$MODEL_REVISION" \
  --collectors psi,cgroup-psi,thread-snapshot,gpu-dmon \
  --gpu-index 0
```

The output directory contains:

- `manifest.json`: software, topology, cgroup, policy, corpus, and command;
- `warmup/`: every raw shape and stability pass;
- `warmup.json`: the accepted stability-window index;
- `adjacent_baseline.json`: matched unprofiled control immediately before;
- `adjacent_baseline_after.json`: matched unprofiled control after finalization;
- `adjacent_baseline_system.json`: PSI windows around both controls;
- `events/*.jsonl`: request/build/encoder/scheduler phase events with PID/TID;
- `measurement/`: every raw measured request result;
- `thread_snapshots.jsonl`, `gpu_dmon.txt`, and `system.json`;
- `profile_start.json` and `profile_stop.json`: acknowledgements;
- `event_report.json`: joined high-level phase and batching summary;
- `artifact_index.json`: byte size and SHA-256 of every artifact;
- `result.json`: performance, perturbation, and integrity result.

Reject a run with failed requests, dropped events, missing stage
acknowledgements, unstable warmup, or unexpected encoder-cache hits. The
profiled and both adjacent passes use the same duration-stratified sample
subset and request concurrency; `--profile-samples 0` opts into the full
corpus. Pressure limits apply independently to both controls and the profiled
window when explicitly configured. The checked attribution campaign does not
set a loaded-window PSI ceiling: PSI is a measured outcome and forcing it low
would censor a possible CPU-saturation mechanism. It gates only ambient
cgroup PSI before startup and records host-wide and cgroup pressure throughout.

## 5. Two bounded PyTorch operator traces

Run torch profiling separately from Nsight and detailed PMU passes:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode torch \
  --run-id torch-quiet-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --profile-samples 256 \
  --model-revision "$MODEL_REVISION" \
  --collectors psi \
  --torch-wait-steps 1 \
  --torch-warmup-steps 1 \
  --torch-active-steps 20
```

This scheduler-owned trace answers whether Python request construction,
scheduler execution, or CUDA launch dispatch dominates. Run a separate
encoder-owned trace to see the preprocessing thread and encoder submission
path that the scheduler-owned trace cannot faithfully observe:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode torch \
  --run-id torch-encoder-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --profile-samples 256 \
  --model-revision "$MODEL_REVISION" \
  --collectors psi \
  --torch-owner pre_lm_encoder \
  --torch-wait-steps 1 \
  --torch-warmup-steps 1 \
  --torch-active-steps 20
```

The two owners are deliberately mutually exclusive. The stop endpoint flushes
in-flight async work before export. The harness rejects a trace missing the
selected owner canary, CUDA launch-to-kernel correlation, scheduled steps,
rank identity, event files, or finalization acknowledgement.

Do not use the profiled run as the throughput baseline. `result.json` reports
the profiled change from the midpoint of the two unprofiled controls and their
temporal drift. If the controls drift by more than 2%, the probe-effect
comparison is inconclusive. Campaign performance always comes from the
unprofiled controls.

Enable expensive flags only in a second short run:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode torch \
  --run-id torch-stacks-r1 \
  --concurrency 32 \
  --max-samples 256 \
  --model-revision "$MODEL_REVISION" \
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
  --model-revision "$MODEL_REVISION" \
  --collectors psi \
  --profile-timeout-s 600 \
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
  --model-revision "$MODEL_REVISION" \
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
- **Request-build head-of-line blocking:** matched
  `scheduler_request_build_hol_start/end` intervals show completed later
  futures held behind the first incomplete future for material time.
- **Admission leak or abort race:** each `request_builder_submitted` has exactly
  one `request_build_capacity_release`; the lifecycle gate rejects imbalance.
- **Client-side artifact:** `client_queue_ns` or scheduled-arrival lag rises
  while server receive-to-terminal phases do not. The load generator reads
  warmed audio from cache and performs any first load outside its event loop.

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
  tests/unit_test/benchmarks/test_cpu_saturation_campaign.py \
  tests/unit_test/benchmarks/test_profile_cpu_saturation.py \
  tests/unit_test/benchmarks/test_system_collectors.py \
  tests/unit_test/serve/test_openai_api.py \
  tests/unit_test/fun_asr/test_encoder_service.py \
  tests/unit_test/fun_asr/test_request_builders.py
```

Then run one 32-request torch smoke capture (`--profile-samples 32`,
`--torch-active-steps 4`) and require a passing `result.json` integrity gate
before starting the multi-restart matrix.

## 11. Decision sequence after the campaign

Do not collect every expensive profiler at once.

1. Use the event campaign to identify the first phase whose per-request
   service time or wait time diverges between quiet and stress.
2. If runnable delay diverges, run `perf-sched` alone and join by the native
   TIDs in the events. If busy MHz diverges without runnable delay, run
   `turbostat` plus the low-overhead default `perf-stat`.
3. If request build or pre-LM CPU work diverges, run the matching bounded Torch
   owner (`scheduler` or `pre_lm_encoder`).
4. If CPU phase duration predicts GPU idle/feed gaps, run the one Nsight joint
   trace. Require the NVTX capture token, CUDA API rows, kernels, OS-runtime
   rows, and the copied `.nsys-rep`.
5. Only then repeat the relevant narrow capture with server-core pinning,
   sibling isolation, and a matched stress placement. This is a causal
   intervention, not the default fix.
6. Use the router only as a final boundary test. If direct serving reproduces
   the mechanism, the root cause is below the router. If direct serving is
   clean but router serving fails under the same offered-load contract, add
   router ingress/egress timing and profile that separate boundary.
