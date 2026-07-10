# Experimental same-GPU data parallelism with CUDA MPS

Same-GPU data parallelism runs several complete SGLang Omni replicas on one
physical GPU. CUDA Multi-Process Service (MPS) can let kernels from those CUDA
processes execute concurrently, filling gaps left by host dispatch or small GPU
workloads.

This is an **experimental deployment technique**. It is useful only when it
beats a tuned single replica under the latency, output, quality, and reliability
requirements of the workload. It is not the default DP placement strategy.

Historical Higgs TTS experiments in [issue
#907](https://github.com/sgl-project/sglang-omni/issues/907), [PR
#912](https://github.com/sgl-project/sglang-omni/pull/912), and [PR
#986](https://github.com/sgl-project/sglang-omni/pull/986) found substantial
MPS scaling over particular same-GPU configurations. The first independent
review briefly found tuned DP1 faster, but that DP run used concurrency 16 per
replica and under-filled a 64-request generation cap. Its corrected sweep found
DP2 and DP3 wins for host-bound Higgs, while compute-bound MOSS-TTS-Local showed
almost no peak gain. These results establish the mechanism and its sensitivity,
not portable performance ratios: checkpoint, generation length, CPU placement,
batching, and client behavior all matter.

## What is replicated

The Higgs V1 configuration is one fused pipeline process containing
preprocessing, reference audio encoding, autoregressive generation, and the
vocoder. Same-GPU DP launches that complete process more than once. Each replica
has its own Python interpreter and CUDA context, which can reduce Python/GIL and
host-dispatch serialization.

It also duplicates the AR weights and CUDA graphs, codec/vocoder state, KV and
request pools, reference caches, scheduler, and HTTP state. Memory cost and
startup time therefore grow with the replica count. MPS schedules CUDA work
between processes; it does not share model weights or turn the replicas into one
engine.

Without MPS, independent CUDA contexts normally time-slice. With MPS, kernels
from different clients may overlap on the same SMs. Scaling stops when GPU
compute, memory bandwidth, VRAM, host dispatch, the client, or a router becomes
the next bottleneck. NVIDIA's [MPS deployment
guide](https://docs.nvidia.com/deploy/mps/) describes its scheduling and control
interfaces.

MPS provides concurrency, not strong isolation. On Volta and newer GPUs, a
fatal GPU fault is reported to all MPS clients sharing the affected GPU, and the
MPS server remains unavailable to affected clients until they exit. Test this
blast radius and restart behavior before production use.

## Memory is a measured contract

`--mem-fraction-static` is an SGLang memory-planning input, not a fraction that
can safely be multiplied by the replica count. Higgs loads other GPU components
before the AR engine profiles its KV headroom. The second and later replicas see
less free memory because complete earlier replicas are already resident.
Consequently:

- equal fractions do not guarantee equal `max_total_num_tokens`;
- launch order may change per-replica KV capacity and batching headroom;
- sequential startup makes the order reproducible, but not the capacities equal;
- a point that fits on one driver/CUDA/checkpoint combination may OOM on another.

Record the resolved KV token capacity for every replica. Publication-quality
comparisons should pin an equal capacity when the backend exposes a stable
setting, or fail the run when the resolved values differ. The validation harness
warns when it cannot extract this value from the installed SGLang log format.

## CPU and traffic placement

Give every replica a dedicated subset of one fixed, NUMA-local server CPU
budget. Give every direct benchmark client a separate dedicated subset, disjoint
from all server cores. DP1 receives the union of the server cores used by DP2–4;
DP does not receive additional host resources merely because it has more
processes.

Shared affinity such as assigning `0-31` to every replica does not isolate CPU
dispatch and can hide severe scheduling contention. Record `nvidia-smi topo -m`,
CPU sets, memory binding, and client placement with every result.

Measure direct traffic first: one concurrent canonical benchmark client per
worker. This establishes the replica pool ceiling and per-worker balance. Add
the [Omni Router](omni_router.md) in a separate, otherwise identical condition.
Router throughput is a different result because its connection limits, request
selection, event loop, and CPU placement can become bottlenecks.

Saturate every replica before comparing peaks. Sweep per-worker client
concurrency through and beyond the effective generation batch cap; concurrency
16 can substantially under-drive a replica configured for 64 running requests.
Use the best point under the same latency/RTF SLO for DP1 and each DPk condition.

## Safe MPS lifecycle

Use a GPU UUID rather than a physical ordinal when scoping `CUDA_VISIBLE_DEVICES`
for the MPS daemon and every replica. Inside that visibility boundary the
pipeline correctly addresses the selected GPU as `cuda:0`, without depending on
host ordinal remapping.

Each benchmark condition should use unique `CUDA_MPS_PIPE_DIRECTORY` and
`CUDA_MPS_LOG_DIRECTORY` paths. A valid lifecycle is:

1. Confirm the selected GPU is otherwise idle and in compute mode `Default`.
2. Start a per-user MPS daemon with only that GPU UUID visible.
3. Launch one tracked replica process group and wait for bounded `/health`
   readiness before launching the next.
4. Run MPS `ps` (or enumerate server/client lists) and prove that a CUDA client
   belonging to every tracked replica is attached. A live control daemon or the
   absence of `MpsRpc` errors is not proof of attachment.
5. On teardown, send `SIGTERM` only to tracked replica groups and wait.
6. For a stuck CUDA client, use MPS `terminate_client` with the enumerated server
   and client PIDs before any final process-group kill.
7. Send `quit` only to the MPS daemon owned by this condition.

Do not use broad `pkill` patterns. Do not reuse or stop another service's MPS
daemon. An interrupted launch, readiness failure, client failure, and normal
completion must all follow the same owned cleanup path.

For an MPS-off control, explicitly set `CUDA_MPS_PIPE_DIRECTORY` to a unique
nonexistent path. NVIDIA documents this as the MPS bypass; merely omitting the
variable can attach a client to a daemon at the default pipe and invalidate the
comparison. Use one UUID-scoped MPS control/server pair per GPU for multi-GPU
experiments, with a distinct pipe directory for each pair.

MPS active-thread percentage and pinned device-memory limits are useful
experiments for provisioning or containing a noisy replica. They are not strong
SM, bandwidth, or fault isolation. Validate their exact CUDA-version syntax and
measure both unconstrained and approximately `100 / DP` active-thread settings.

## H200 validation procedure

Use the same-GPU DP benchmark harness in
`benchmarks/same_gpu_dp/README.md`. It drives the canonical
`benchmark_tts_seedtts` `/v1/audio/speech` path, records manifests, validates CPU
placement, owns MPS and child lifecycles, and aggregates the existing
`speed_results.json` schema. Run its dry-run mode and inspect every command before
using the H200.

First tune DP1 across at least:

- client concurrency `8, 16, 32, 48, 64, 96, 128`;
- `max_running_requests` and CUDA graph batch size `64/64` and `128/128`;
- viable memory fractions;
- the complete NUMA-local server CPU budget not reserved for clients.

Choose the Pareto-best DP1 point under a declared p95 latency/RTF SLO. Then run:

| Variable | Required points |
|---|---|
| Replicas | DP1, DP2, DP3, DP4 (record OOM/not-fit points) |
| MPS | off and on |
| MPS provisioning | unconstrained and approximately `100 / DP` active threads |
| Traffic | direct first, router separately |
| Workload | short/long text, warm/cold reference, streaming/non-streaming |
| Load | per-worker concurrency sweep through the generation batch cap |
| Repetitions | at least five, with randomized condition order |
| CPU | one fixed total server and client budget, dedicated subdivisions |
| Memory | equal recorded KV-token capacity or an explicit failed fairness gate |

Do not report request QPS alone. Preserve and compare:

- p50, p95, and p99 latency;
- mean and tail real-time factor (RTF);
- generated audio seconds per wall second;
- output codec tokens per second and total/mean output tokens;
- generated duration and sample success/failure distribution;
- per-worker throughput and balance;
- WER plus the relevant similarity/naturalness checks after generation.

Use a separate ASR phase so quality measurement does not contend with the TTS
servers. Device-level Nsight Systems SM/DRAM/Tensor metrics can explain idle
headroom, but they do not replace end-to-end metrics.

A reasonable acceptance gate is:

- more than 10% throughput over tuned DP1 with non-overlapping 95% confidence
  intervals;
- comparable p95/p99 latency and RTF under the declared SLO;
- no additional failures or material output-duration/token/quality drift;
- controlled/equal KV capacity and balanced direct-worker traffic;
- a 30–60 minute soak and repeated clean start/stop cycles;
- router throughput close to direct aggregate throughput, or an explicit claim
  that excludes router deployment.

## MIG alternative

NVIDIA Multi-Instance GPU (MIG) is an alternative when stronger hardware
partitioning and fault isolation matter more than flexible pooling. H200 offers
several MIG profiles; available combinations depend on the exact H200 product
and driver. Confirm them with NVIDIA's [supported MIG profile
table](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html)
and the target host.

MIG gives each instance dedicated resources but may strand capacity when
replicas have uneven demand. MPS shares the GPU more flexibly but shares the
failure domain. MPS may also be used within a MIG instance. Benchmark MIG as a
separate placement condition; do not compare a differently sized MIG workload
to full-GPU MPS and call the difference a scheduling result.
