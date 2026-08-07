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

### Unprofiled stability characterization

If either condition cannot pass warmup stability, stop repeating the event
campaign. Use the dedicated stability configuration instead:

```bash
cp benchmarks/profiling/campaign.stability.h100.example.json \
  /tmp/campaign.stability.h100.json
```

Replace only the immutable model revision and machine-specific GPU index, then
run:

```bash
python -m benchmarks.profiling.run_cpu_saturation_campaign \
  --config /tmp/campaign.stability.h100.json \
  --output-dir \
  "$PWD/artifacts/cpu-saturation/stability-ab-$(date -u +%Y%m%dT%H%M%SZ)"
```

This is an unprofiled workload experiment: it never starts the request event
recorder, Torch profiler, or Nsight. The campaign resolves the real stage PID
from the authoritative stage-spawn startup line and verifies that it belongs
to the launched server process tree.

Collectors start before the corpus-shape pass and remain active across 20
fixed 256-request windows. Instability does not stop the run. Every window
atomically preserves its request result and a matched system record containing
host/cgroup PSI, stage CPU time, native-thread CPU and runnable-delay deltas,
migrations, and request accounting. Continuous thread snapshots,
`nvidia-smi dmon`, and the built-in sysfs/procfs `cpu-frequency` collector
provide process scheduling, GPU utilization/clock, and sampled
`scaling_cur_freq`. The frequency result is explicitly labeled as sampled
scaling frequency, not APERF/MPERF-derived effective MHz. Without
`--cpu-frequency-cpus`, it is host-wide and CPU interferers contribute to its
busy weighting. Placement experiments must pass the exact server CPU list.
The server log remains the source for ordinary scheduler batching and
occupancy lines; event recording stays off. Add `perf-stat` or `turbostat`
only when matching kernel tools are actually available.

On Python versions that do not propagate `threading.Thread.name` to Linux
`comm`, the scheduler, request-build executor, and pre-LM encoder set bounded
native names when their worker threads start. Procfs captures therefore expose
`sched-<stage>`, `omni-request-bu` (the Linux 15-byte truncation), and the
model-specific pre-LM worker prefix without enabling request events. The
Fun-ASR example requires these labels with `--required-thread-comms`, so a
capture cannot be accepted if native naming silently fails.

Each workload window records monotonic and wall-clock boundaries. Its
observable native-thread CPU total must reconcile with stage process CPU time
within `--max-thread-cpu-accounting-error`. A thread born during the window is
included from its lifetime counters when it remains observable at the ending
boundary. A thread that exits before that boundary makes exact per-thread
attribution invalid. Per-window frequency summaries include the nearest sample
on each side of the workload interval and require that both boundaries are
bracketed. The campaign records the server, stage, and interferer process trees
with per-thread affinity and cgroup membership.
Incomplete requests, missing requested collector samples, incomplete pressure
windows, or failed CPU reconciliation preserve the artifacts but set
`accepted=false`; rejected trials never enter campaign comparisons.

Run one fresh server for `quiet` and one for `unbound-cpu64` first. Inspect the
window distributions and telemetry before increasing restart count. Loaded
PSI is evidence, not an acceptance gate. Preserve results even when none of
the rolling three-window groups meets the 5% stability criterion.

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

## 6. Non-injected Nsight CPU attribution

Use this path when `nsys profile ... sgl-omni serve` kills a CUDA worker before
weight loading. It does not launch, attach to, preload, or inject a library
into the server. The server starts normally and the harness creates a separate,
named, system-wide Nsight session around only the measured request window:

```text
normal server -> shape/stability warmup -> unprofiled control
              -> system-wide CPU sample + context-switch window
              -> unprofiled control -> finalize/export/validate
```

This mode deliberately configures no API trace domain. (`nsys start` does not
accept the `--trace=none` spelling on the tested 2026.2.1 installation.) It
collects CPU IP/backtrace samples and scheduler context switches, not CUDA,
NVTX, OS-runtime interposition, Python sampling, or GIL tracing. Every Nsight
control/export command runs with inherited `DEBUGINFOD_URLS` removed so
system-wide finalization cannot stall downloading symbols for unrelated host
processes. Independent `gpu-dmon`, procfs thread snapshots, CPU frequency, and
PSI collectors remain active. This is the correct next experiment on a host
where every injected Nsight launch destabilizes the worker.

Run the existing focused unit gate, then confirm the installed Nsight CPU
environment:

```bash
pytest -q \
  tests/unit_test/benchmarks/test_system_collectors.py \
  tests/unit_test/benchmarks/test_profile_cpu_saturation.py

nsys --version
nsys status --environment
nsys sessions list
```

Start a fresh Fun-ASR server normally, using the exact server command in
section 2. Do not prefix it with `nsys`, and do not set an Nsight injection or
`LD_PRELOAD` variable. After the server is healthy, run the quiet arm:

```bash
export GPU=1
export MODEL_REVISION=e296c62e32a328b1d649a8f02701ac54c2fac9f0

python -m benchmarks.profiling.profile_cpu_saturation \
  --mode nsys-system-wide \
  --run-id nsys-system-wide-quiet-r1 \
  --concurrency 32 \
  --max-samples 256 \
  --shape-warmup-samples 64 \
  --warmup-samples 64 \
  --stability-windows 2 \
  --stability-tolerance 0.10 \
  --max-warmup-windows 4 \
  --max-adjacent-baseline-drift 0.05 \
  --profile-samples 128 \
  --model-revision "$MODEL_REVISION" \
  --collectors psi,cgroup-psi,thread-snapshot,gpu-dmon,cpu-frequency \
  --thread-sample-interval-ms 250 \
  --required-thread-comms sched-asr,omni-request-bu,fun-asr-audio-e \
  --nsys-required-thread-comms sched-asr,omni-request-bu,fun-asr-audio-e \
  --nsys-samples-per-backtrace 4 \
  --gpu-index "$GPU" \
  --profile-timeout-s 600 \
  --nsys-finalize-timeout-s 600
```

This is a narrow attribution run, not the earlier full campaign. Per arm it
uses one 64-request shape pass, two to four 64-request stability windows, one
128-request control before the capture, exactly one captured 128-request pass,
and one 128-request control after it. Thus only 128 requests are inside Nsight,
and the complete arm is bounded to 576–704 requests. Four CPU samples per
native backtrace reduces unwind and artifact cost while retaining every
scheduling transition and periodic CPU samples. The two short controls may
drift by at most 5% for a valid QPS/latency comparison. The 600-second values
are failure timeouts; they do not extend the capture.

Warmup stability is diagnostic in this mode, not a capture gate. If the server
oscillates between fast and slow regimes, `warmup.json` records
`reached_stability=false` and the observed QPS range, then proceeds to the
bounded capture. Likewise, excessive adjacent-control drift sets
`performance_comparison_integrity.valid=false`: it forbids interpreting
profiled-versus-control QPS, but it does not discard complete scheduling
evidence or prevent the CPU64 arm. Stop only when `accepted=false`,
`capture_complete=false`, or request/system integrity fails.

The harness discovers the target process read-only from `/proc` using the
startup-created `sched-asr` thread. The full three-thread evidence contract is
checked after warmup, when lazy request-builder workers have been created. If
more than one matching server exists, stop the unrelated server or pass the
exact ASR stage `--server-pid`. This mode never uses the event-recorder control
plane for PID discovery.

For the CPU64 arm, stop the quiet server and start another fresh server with
identical flags. In a separate shell, start the same unbound interferer used by
the stability experiment:

```bash
python -m benchmarks.profiling.cpu_interferer --workers 64 \
  |& tee /tmp/nsys-system-wide-cpu64-interferer.log
```

Then rerun the harness with only the run ID changed:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode nsys-system-wide \
  --run-id nsys-system-wide-cpu64-r1 \
  --concurrency 32 \
  --max-samples 256 \
  --shape-warmup-samples 64 \
  --warmup-samples 64 \
  --stability-windows 2 \
  --stability-tolerance 0.10 \
  --max-warmup-windows 4 \
  --max-adjacent-baseline-drift 0.05 \
  --profile-samples 128 \
  --model-revision "$MODEL_REVISION" \
  --collectors psi,cgroup-psi,thread-snapshot,gpu-dmon,cpu-frequency \
  --thread-sample-interval-ms 250 \
  --required-thread-comms sched-asr,omni-request-bu,fun-asr-audio-e \
  --nsys-required-thread-comms sched-asr,omni-request-bu,fun-asr-audio-e \
  --nsys-samples-per-backtrace 4 \
  --gpu-index "$GPU" \
  --profile-timeout-s 600 \
  --nsys-finalize-timeout-s 600
```

Stop the interferer with `Ctrl-C` after the harness exits. Do not compare an
arm from this pair with an earlier campaign: the useful comparison is this
fresh, adjacent quiet/CPU64 pair with identical workload and server settings.

An accepted result requires all of the following:

- `result.json`: `capture_complete=true`, `accepted=true`,
  `request_integrity.valid=true`, and `system_integrity.valid=true`;
- `system.json`: complete request-window monotonic and wall-clock boundaries;
- `nsight-system-wide/system-wide-cpu.nsys-rep`: finalized and nonempty;
- `nsight-system-wide/system-wide-cpu.sqlite`: successful non-lazy export;
- `nsight-system-wide/summary.md`: generated temporal and hotspot summary;
- nonempty `SCHED_EVENTS`, `COMPOSITE_EVENTS`, and `cpuCycles=1` samples;
- stable `/proc` TID/start-time identities across the measured window;
- scheduling events, real CPU samples, and sampled leaf symbols for
  `sched-asr`, `omni-request-bu`, and `fun-asr-audio-e`.

`accepted` covers capture, request, and system integrity. It intentionally does
not assert that the host remained in one throughput regime. QPS and latency may
be compared between arms only when both results also report
`performance_comparison_integrity.valid=true`. Thread state, blocking reasons,
on-CPU intervals, and native stacks remain valid attribution evidence when
that comparison field is false.

The filtered evidence is in
`system.json["nsys_system_wide_cpu"]["evidence"]`. For each semantic thread
class it reports TIDs, scheduling counts, true CPU-sample counts, and the top
leaf and inclusive native symbols. It also reconstructs the timestamped
sched-in/sched-out intervals and reports:

- total on-CPU service and class-active wall time;
- runnable-but-off-CPU, blocked, and unknown off-CPU distributions;
- blocking-reason time, CPU changes, and non-alternating-event warnings;
- pairwise and all-class simultaneous on-CPU overlap;
- trace timestamps for the longest runnable, blocked, and overlap intervals,
  which are the exact regions to inspect if the GUI is needed.

The same decision-oriented subset is written to
`nsight-system-wide/summary.md`, so opening the GUI is not mandatory for the
first diagnosis. The raw `.nsys-rep` and SQLite database remain available for
visual verification or follow-up SQL. `cpuCycles=0` call stacks are excluded
from hotspot attribution because those are context-switch stacks, not periodic
CPU samples.

Interpret the pair in this order:

1. If runnable off-CPU time rises while CPU-sample share is flat, the thread is
   being starved by the OS scheduler. Compare migrations and placement.
2. If on-CPU time, sample share, and a native stack family grow while runnable
   delay stays low, investigate additional service cost or cache/SMT
   interference in that owner.
3. If blocked time rises, use the reported block reason and owner to select the
   dependency boundary (request builder, pre-LM encoder, or scheduler wakeup).
4. Use overlap to decide whether one serial owner or several simultaneously
   active stages create the critical path. Falling `gpu-dmon` SM utilization
   while those host stages are delayed confirms GPU-feed starvation.
5. If the scheduler remains runnable and inexpensive but ready work is still
   admitted late, Nsight has reached its semantic boundary. Measure only
   request-build head readiness, later-ready count, pre-LM publish, and drain
   time; do not broaden the profiler.
6. Only after an owner and mechanism are selected should a narrow in-process
   Torch or Python profiler be applied.

This capture cannot prove GIL ownership or map interpreter samples to Python
lines, attach scheduling intervals to request IDs, prove request-build
head-of-line ordering, or join CUDA launches to kernels. Those are intentional
limitations of avoiding injection. It can select the owner and mechanism class
without risking another model startup under Nsight injection.

For optional manual inspection, copy the finalized `.nsys-rep` from the remote
container and open it with the same or a newer Nsight Systems GUI. Select the
target PID and expand the `sched-asr`, `omni-request-bu`, and
`fun-asr-audio-e` OS-thread rows. The useful visual check is whether a slow
interval is runnable but not scheduled, blocked, or on CPU in a sampled native
stack. Do not infer request ordering or CUDA causality from this CPU-only
timeline; those lanes were intentionally not collected.

### Scheduler architecture boundary

Do not choose a scheduler repair merely because the scheduler thread appears
in the distributed critical path. Fun-ASR has modality-specific work before LM
admission: audio loading/feature construction, a request-build pool, and one
batched pre-LM encoder service. Once a request is LM-ready, the token/KV
scheduler's core invariants are shared with text.

The design question is therefore where resource readiness and backpressure
belong, not whether every modality needs a separate token scheduler. Evidence
of ordered-drain head-of-line blocking or pre-LM queue amplification supports a
speech-specific admission policy. Evidence of expensive or starved LM
scheduling after readiness supports a core scheduler change. TTS may require a
different stage policy for streaming cadence and downstream audio generation,
while still sharing the LM scheduler where its token/KV invariants match.

### Injected joint CPU/CUDA/Python trace

Nsight must launch the server so it follows the worker process tree. The
server emits one scheduler-owned NVTX range named
`sglang_omni.capture_window`; the harness opens it only after stability
warmup and closes it after the measured requests complete.

For the Fun-ASR CPU-interference investigation, CUDA/NVTX/OS-runtime tracing
alone is insufficient. Require all of the following in the same short
capture:

- process-tree CPU sampling, to locate native and Python CPU hotspots;
- process-tree context switches, to distinguish running, runnable, and
  blocked intervals for the named critical threads;
- Python sampling, to attribute work inside the request builder, scheduler,
  and pre-LM service;
- Python GIL tracing, to distinguish GIL ownership/waiting from CFS delay;
- CUDA API and GPU activity, to join host stalls to launch gaps and kernels.

Nsight Systems 2024.2 or newer provides the Python sampling and GIL switches
used below. Check the installed CLI before starting the server; do not silently
drop an unsupported signal and later treat the trace as complete:

```bash
export SGLANG_OMNI_NVTX_RANGES=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

nsys --version
nsys profile --help | grep -E \
  -- '--cpuctxsw|--python-sampling|--python-sampling-frequency|cuda-sw|python-gil'

nsys profile \
  --trace=cuda-sw,nvtx,osrt,python-gil \
  --cuda-trace-scope=process-tree \
  --cuda-event-trace=false \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --python-sampling=true \
  --python-sampling-frequency=250 \
  --resolve-symbols=false \
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

Use `cuda-sw` deliberately on H100. Recent Nsight releases prefer CUDA's
hardware event trace when they consider it available, but NVIDIA documents
that path as beginning with Blackwell and recommends the legacy software trace
for unsupported GPUs or incompatible driver/CUPTI combinations. Forcing the
software path also avoids treating an automatic hardware-trace fallback as an
accepted capture. `--resolve-symbols=false` prevents report finalization from
blocking on remote debug-symbol downloads; the Python samples, native thread
names, NVTX ranges, CUDA API names, and kernel names required by this
investigation remain in the report.

Before loading model weights, validate the exact selected GPU and driver path
with a disposable CUDA-context capture:

```bash
CUDA_VISIBLE_DEVICES="$GPU" nsys profile \
  --trace=cuda-sw \
  --cuda-trace-scope=process-tree \
  --cuda-event-trace=false \
  --sample=none \
  --cpuctxsw=none \
  --resolve-symbols=false \
  --force-overwrite=true \
  --output="/tmp/nsys-cuda-sw-smoke-gpu${GPU}" \
  python - <<'PY'
import torch

assert torch.cuda.is_available()
value = torch.ones(1, device="cuda")
torch.cuda.synchronize()
assert value.item() == 1.0
PY
```

Require a finalized `.nsys-rep` and exit status zero. If this minimal
`cuda-sw` capture produces the same NVRM reference-state error, stop: the
remaining blocker is the installed Nsight/CUPTI/driver combination or a stale
external profiler session, not SGLang-Omni. Do not spend another model startup
on it. Check for profiler processes owned by the same user and use a
toolkit-bundled Nsight version compatible with the installed driver; a driver
reset or host reboot is an operator action, not part of this benchmark.

### CPU-only fallback when CUDA tracing destabilizes the driver

If either hardware or software CUDA tracing emits an NVRM reference-state
error, do not launch the model under any CUDA-enabled Nsight trace. Use the
CPU-only evidence contract instead. It still captures the evidence needed to
decide between scheduler/GIL starvation, request-preprocessing work, pre-LM
dependency waits, CFS scheduling delay, and request-build head-of-line
blocking. GPU SM utilization, power, and clocks come from the independent
`gpu-dmon` collector rather than CUPTI, so this mode cannot correlate an
individual CUDA API call to a kernel.

Launch each fresh server with no CUDA trace domain or CUDA-specific Nsight
switches:

```bash
export SGLANG_OMNI_NVTX_RANGES=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

nsys profile \
  --trace=nvtx,osrt,python-gil \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --python-sampling=true \
  --python-sampling-frequency=250 \
  --resolve-symbols=false \
  --capture-range=nvtx \
  --nvtx-capture=sglang_omni.capture_window \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$PWD/artifacts/cpu-saturation/nsys-cpu-quiet-r1" \
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
  --nsys-cpu-only \
  --run-id nsys-cpu-quiet-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --profile-samples 256 \
  --model-revision "$MODEL_REVISION" \
  --collectors psi,gpu-dmon \
  --gpu-index "$GPU" \
  --profile-timeout-s 600 \
  --nsys-report \
    "$PWD/artifacts/cpu-saturation/nsys-cpu-quiet-r1.nsys-rep"
```

Repeat once under the same unbound-CPU64 condition, changing only the run ID
and output paths. An accepted CPU-only result explicitly records
`nsys_report.capture_contract="cpu-only"` and `cuda_required=false`. It must
still pass the automated gates for the capture window, every required semantic
NVTX range, OS-runtime activity, a finalized report, complete requests, and
valid independent GPU telemetry. Before interpreting an accepted report, open
it in Nsight and require non-empty Python Samples, Python GIL, and thread
Scheduling lanes for `sched-asr`, `omni-request-bu`, and
`fun-asr-audio-e`. Reject the capture if any lane is absent. Those lane
encodings are not exposed by a stable cross-version `nsys stats` report, so
the harness does not pretend to validate them from incidental report text.

If that CPU-only bundle still kills a worker before the capture window, the
failure has not implicated the NVTX annotations. That command also enables
OS-runtime interception, Python/GIL instrumentation, CPU sampling, and
process-tree context-switch collection during worker startup. Do not repeat
the same bundle. Reduce the experiment to NVTX alone:

```bash
export SGLANG_OMNI_NVTX_RANGES=1
export NSYS_NVTX_PROFILER_REGISTER_ONLY=0

nsys profile \
  --trace=nvtx \
  --sample=none \
  --cpuctxsw=none \
  --python-sampling=false \
  --resolve-symbols=false \
  --capture-range=nvtx \
  --nvtx-capture=sglang_omni.capture_window \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output="$PWD/artifacts/cpu-saturation/nsys-nvtx-quiet-r1" \
  sgl-omni serve \
    --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
    --host 127.0.0.1 \
    --port 8000 \
    --stages.asr.factory_args.pre_lm_cache_max_entries=0 \
    --stages.asr.factory_args.pre_lm_cache_size_bytes=0
```

Run the same stable workload from another shell:

```bash
python -m benchmarks.profiling.profile_cpu_saturation \
  --mode nsys \
  --nsys-nvtx-only \
  --run-id nsys-nvtx-quiet-r1 \
  --concurrency 32 \
  --max-samples 1088 \
  --profile-samples 256 \
  --model-revision "$MODEL_REVISION" \
  --collectors psi,thread-snapshot,gpu-dmon \
  --gpu-index "$GPU" \
  --profile-timeout-s 600 \
  --nsys-report \
    "$PWD/artifacts/cpu-saturation/nsys-nvtx-quiet-r1.nsys-rep"
```

This is a semantic phase trace, not a CPU stack or scheduler trace. Acceptance
requires the capture window and all request-build, pre-LM, and scheduler ranges,
plus complete requests and independent GPU telemetry. Interpret it together
with the already-qualified native-thread `schedstat`/migration evidence; do
not infer GIL ownership, native hotspots, CUDA launch correlation, or kernel
timing from this report. Run the CPU64 arm only after the quiet NVTX-only arm
is accepted. If the worker also dies under this exact minimal command, the
remaining failure is NVTX/Nsight injection itself and this host cannot produce
a valid Nsight semantic trace.

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

Run one fresh-server quiet capture and one fresh-server unbound-CPU64 capture
with identical requests and launch flags. These are diagnostic traces, not
throughput baselines; retain the unprofiled matched pair for performance
numbers. Keep the active window narrow (normally 128-256 requests) because
Python sampling, GIL tracing, CUDA tracing, and context-switch collection are
deliberately more invasive than the stability campaign.

The trace must contain all of these semantic ranges:

- `request_build.total`, plus `request_build.audio_load`,
  `request_build.feature_extract`, `request_build.tokenize_and_pack`, and
  `request_build.pre_lm_wait`, on `omni-request-bu`;
- `pre_lm.encode`, `pre_lm.synchronize`, and
  `pre_lm.split_embeddings`, on `fun-asr-audio-e`;
- `scheduler.recv_and_admit`, `scheduler.select_batch`,
  `scheduler.model_launch`/`scheduler.model_execute`,
  `scheduler.model_resolve`, and `scheduler.result_process`, on `sched-asr`;
- `request_build.head_of_line` whenever later-ready request-builder futures
  are held behind the oldest incomplete future;
- `sglang_omni.capture_window`, Python samples, Python GIL states, and
  scheduling/context-switch records;
- CUDA API calls and GPU kernels for the joint contract only. CPU-only
  captures instead require complete independent `gpu-dmon` telemetry.

Inspect the two captures in this order:

1. Compare `request_build.total` and its child ranges. Time outside the child
   ranges is request construction/bookkeeping; time in `pre_lm_wait` is a
   dependency stall, not request-builder CPU.
2. Split `pre_lm_wait` into encoder queueing, `pre_lm.encode`, CUDA launch,
   `pre_lm.synchronize`, and split/clone time. A long synchronize with busy GPU
   is device work; a delayed encode launch with idle GPU is host starvation.
3. Check whether `request_build.head_of_line` overlaps later-ready workers and
   GPU feed gaps. Its presence alone is not a bug; it must be material on the
   critical path.
4. On each semantic thread, distinguish CFS runnable delay, GIL wait, blocked
   futex/condition wait, and on-CPU Python/native samples. These mechanisms
   require different fixes.
5. Join the last scheduler/pre-LM CUDA API call before each GPU idle interval.
   The CPU range containing that launch gap identifies the first host phase
   that stopped feeding the H100.

A CPU/NVTX range represents host enqueue or wall time; do not infer device
completion from visual nesting. Do not enable the JSONL event recorder or
Torch profiler in this capture. Their signals are already represented by NVTX,
and stacking profilers would make scheduler/GIL attribution uninterpretable.

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
  tests/unit_test/profiler/test_trace_ranges.py \
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
