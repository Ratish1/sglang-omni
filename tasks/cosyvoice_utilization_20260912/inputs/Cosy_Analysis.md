**For this specific model, I would start with an explicit eager-versus-compiled DiT comparison, verify the actual batch sizes reaching each stage, and capture a short, warmed Nsight Systems trace at concurrency 16. I would not start with custom kernels, tensor parallelism, or MPS.**

There is unusually relevant evidence here: an SGLang-Omni CosyVoice3 profiling change reported CPU-bound eager DiT dispatch at concurrency 16, with its author’s H200 experiment showing about **53% higher audio throughput after compilation**. That is a strong reason to test this path—not evidence that your deployment has the same bottleneck or will obtain the same improvement. ([github.com][1])

Also, **30% SM utilization is not a production-quality threshold**. The objective should be more correct audio per GPU-second while meeting first-audio latency, streaming deadlines, and quality requirements. NVIDIA likewise emphasizes absolute workload duration over utilization percentages when evaluating optimization progress. ([NVIDIA Docs][2])

I prepared a detailed reference and a small diagnostic toolkit:

[**Download the complete research playbook**](sandbox:/mnt/data/SGLang_Omni_CosyVoice_Utilization_Playbook.md) · [**Download the diagnostic scripts and experiment template**](sandbox:/mnt/data/SGLang_Omni_CosyVoice_Diagnostics.zip)

The toolkit includes environment/source inspection, a native benchmark sweep, and an overlap-aware Nsight SQLite analyzer. Its Python tests passed on synthetic data; no benchmark has been run on your deployment.

# 1. First establish what “2%” actually means

Before changing anything, record the **exact metric name, tool, GPU UUID, measurement window, and aggregation**.

These measurements answer different questions:

| Measurement                  | What it tells you                                    | What it does not tell you                                                     |
| ---------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------------------- |
| `nvidia-smi` GPU utilization | How much of a sampling interval had kernel execution | How much of the GPU’s compute capacity was used                               |
| DCGM `SM_ACTIVE`             | Active-warp time averaged across SMs                 | Whether low activity comes from idle periods or insufficient spatial coverage |
| SM occupancy                 | Resident warps relative to supported capacity        | Whether those warps are issuing useful instructions efficiently               |
| Tensor/DRAM activity         | Activity in particular GPU subsystems                | Whether the whole serving pipeline is efficient                               |

In particular, an active warp can be waiting for memory. High occupancy is not equivalent to high useful throughput. ([NVIDIA Docs][3])

## The two fundamentally different causes of low SM activity

Think of two dimensions:

**Temporal underutilization:** the GPU receives work intermittently.

```text
GPU:  [work] ................. [work] ................. [work]
CPU:        preparing / waiting / dispatching / synchronizing
```

**Spatial underutilization:** work is present, but individual launches do not occupy much of the device.

```text
GPU:  [small work][small work][small work][small work][small work]
      Most SMs have little or nothing to do during each launch.
```

As an illustration, 10% temporal activity with 20% spatial coverage gives 2% average activity. So does 50% temporal activity with 4% coverage. These are conceptual examples, not a formula for combining differently sampled tools.

**The interventions differ.** Increasing batch size will not repair reference-download stalls. Faster preprocessing will not repair an inefficient device kernel that already receives continuous work.

Also verify that the metric represents the correct physical resource. CUDA-visible ordinals, host GPU ordinals, and MIG instances can differ; unsupported counters must not be interpreted as zero. ([NVIDIA Docs][3])

# 2. What matters specifically for CosyVoice3

Treat the service as several workloads, not “one small 0.5B model”:

```text
Request / reference audio
        ↓
Reference and text preprocessing
        ↓
Autoregressive speech-token generation
        ↓
Flow / DiT acoustic generation
        ↓
HiFT waveform generation
        ↓
Copy / encode / stream / client
```

The cookbook describes a separate 22-layer DiT, so the `0.5B` label is not an adequate description of the entire pipeline’s execution cost. 

Several exact-model implementation details materially affect your investigation:

| Detail                                                              | Practical consequence                                                |
| ------------------------------------------------------------------- | -------------------------------------------------------------------- |
| An earlier BF16 autocast bug could leave acoustic execution in FP32 | Verify the effective execution path, not just the configured dtype.  |
| Flow batching and concurrent preprocessing have been implemented    | First check that your checkout includes and actually exercises them. |
| The retrieved factory has DiT compilation disabled by default       | Use explicit A/B settings rather than relying on defaults.           |
| HiFT has its own precision setting and currently defaults to FP32   | Do not indiscriminately force every acoustic component to BF16.      |

These points come from the implementation and its associated changes. ([GitHub][4])

There is a particularly important version discrepancy: a default-on compilation change merged on **September 7, 2026**, but the factory source retrieved for this review says `False`. Your installed source and resolved configuration—not a historical PR title—must determine the baseline. ([GitHub][1])

**My initial hypothesis ranking would therefore be:**

1. CPU dispatch overhead in the acoustic path.
2. Insufficient *actual* stage batching despite HTTP concurrency 16.
3. Upstream preprocessing/readiness starvation.
4. Streaming waits, copies, synchronization, or output backpressure.
5. Device-kernel inefficiency after those possibilities are separated.

That is a testing priority, not a diagnosis.

# 3. Define the result you are trying to improve

For production TTS, I would optimize:

> **SLO-qualified audio seconds produced per wall second per GPU.**

Here, *SLO-qualified* means that the output passes your latency, continuity, error, and quality policy.

Record the following together:

| Metric                                    | Why it matters                                                    |
| ----------------------------------------- | ----------------------------------------------------------------- |
| Successful requests/second                | Useful operationally, but sensitive to utterance length           |
| Generated audio seconds/second            | Normalizes for output duration better than request throughput     |
| p50/p95/p99 first-audio latency           | Measures interactive responsiveness                               |
| Completion latency and request RTF        | Measures whole-request behavior                                   |
| Streaming deadline misses or underruns    | Detects a service that starts quickly but cannot sustain playback |
| Errors, truncated output, quality results | Prevents incomplete or degraded audio from appearing “faster”     |
| GPU memory, CPU usage, queue growth       | Establishes whether the operating point is sustainable            |

Use actual sample count and sample rate to calculate audio duration. Do not count a WAV header as first audio or assume an arbitrary HTTP fragment corresponds to one acoustic chunk.

Keep two RTF definitions separate:

```text
Per-request RTF = request latency / that request’s audio duration

Aggregate RTF = measurement wall time / total generated audio duration
```

The inverse of average request RTF is **not** generally aggregate audio throughput.

For streaming, implement a simple playback-buffer measurement: add received playable audio duration, subtract elapsed time, and record when the buffer would fall below zero. Specify the initial playback buffer.

**A successful optimization can lower GPU utilization at unchanged offered traffic because it finishes the work sooner.** Do not reject that improvement because the dashboard percentage went down. ([NVIDIA Docs][2])

# 4. Freeze the environment and run the first A/B test

## 4.1 Capture the actual installation

Run the included collector **inside the serving environment, with the server’s Python executable**:

```bash
python /path/to/cosyvoice_diagnostics/collect_env.py \
  --output results/environment.json
```

It records package versions, source paths and hashes, factory signatures, GPU topology, selected environment settings, and visible CPU/cgroup information.

Separately save the **resolved server configuration and launch command**. Source defaults do not prove live settings.

Before interpreting performance, establish:

* The package imported by the server is the checkout you are editing.
* The observed GPU UUID is the one serving requests.
* No debug synchronization or unrelated workload contaminates the baseline.
* Precision, accelerator selection, concurrency caps, and cache policy are known.

For example, `CUDA_LAUNCH_BLOCKING=1` forces synchronous CUDA execution and is inappropriate for an ordinary performance baseline. ([PyTorch Docs][5])

## 4.2 Compare explicit eager and compiled execution

Preserve your existing launch configuration. The relevant overrides for an eager baseline are:

```bash
--vocoder.factory.enable_dit_torch_compile false \
--vocoder.factory.enable_flow_estimator_trt false
```

For the compiled candidate:

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.enable_dit_torch_compile true \
  --vocoder.factory.enable_flow_estimator_trt false
```

These options are documented for the inspected integration. Run the candidates **sequentially**, not as two competing servers on the same GPU. 

Warm representative lengths and batch sizes before measuring. Record compilation/recompilation behavior separately from steady-state performance.

A useful initial benchmark is:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --lang en \
  --max-concurrency 16 \
  --max-samples 256 \
  --warmup 32 \
  --use-existing-server \
  --generate-only \
  --stream \
  --output-dir results/cosy_compiled_c16_stream
```

The checked runner supports these arguments. Its streaming mode requests PCM; `--generate-only` keeps later ASR evaluation out of the generation-performance measurement. ([GitHub][6])

Run the identical workload at **c1 and c16** first. Then sweep **1, 2, 4, 8, 16**, with 32 only as an approved diagnostic point.

The screening run above is not a p99 qualification. Use larger independent runs for tails.

## 4.3 Separate workload dimensions that can conceal the bottleneck

Test short, medium, and long utterances separately, followed by a mixed workload. Also separate:

**Repeated versus unique references.** A warm reference cache can make preprocessing disappear from a benchmark even when it dominates real traffic.

**Streaming versus buffered-client output.** Verify the internal execution path as well; client response mode alone does not establish which acoustic path executed.

**Closed-loop versus rate-driven arrivals.** Closed-loop c16 is useful for a capacity curve. Production overload testing must also preserve intended arrival times, rejections, and queueing rather than hiding them behind a client semaphore.

Use fixed inputs and generation settings. For precise stage comparisons, replay fixed intermediate tokens/conditioning so that different generated output lengths cannot explain the gain.

# 5. Capture the timeline before going deeper into kernels

## 5.1 Collect counters in the benchmark window

On a host with DCGM configured, discover supported fields:

```bash
dcgmi profile --list --entity-id gpu:0

dcgmi dmon --entity-id gpu:0 \
  --field-id 1001,1002,1003,1004,1005 \
  --delay 1000
```

Match `gpu:0` to the **physical GPU**, not blindly to CUDA ordinal 0. Check compatible metric groups and locally supported CLI options. ([NVIDIA Docs][7])

Record clocks and power alongside utilization:

```bash
nvidia-smi \
  --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm \
  --format=csv \
  --loop-ms=1000
```

Retain raw samples. A dashboard average that includes startup and drain can hide the actual steady-state behavior. The queried utilization remains a time-based metric, not SM capacity utilization. ([NVIDIA Docs][3])

Hardware-counter collection can conflict across profilers. Coordinate any DCGM profiling pause with the monitoring owner: its pause/resume scope is host-engine-wide, not limited to the selected GPU. ([NVIDIA Docs][7])

## 5.2 Use Nsight Systems for the first detailed trace

Launch the **actual server** under Nsight, including its worker descendants:

```bash
mkdir -p traces

nsys launch --trace=cuda,nvtx,osrt --sample=none \
  sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.enable_dit_torch_compile false \
  --vocoder.factory.enable_flow_estimator_trt false
```

After readiness and representative warmup, start a sufficiently long c16 benchmark in another terminal. Capture a steady-state interval from a control terminal in the same user/container session:

```bash
nsys start --stop-on-exit=false -o traces/cosy_eager_c16
sleep 20
nsys stop
```

Use one interactive session at a time. Repeat for the compiled candidate. This launch-then-start collection pattern avoids requiring an arbitrary fixed startup delay. ([NVIDIA Docs][8])

Export summaries:

```bash
nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  traces/cosy_eager_c16.nsys-rep

nsys export --type sqlite \
  -o traces/cosy_eager_c16.sqlite \
  traces/cosy_eager_c16.nsys-rep
```

The supplied analyzer calculates the **union of captured kernel intervals**, not the sum of durations. It requires an explicit steady-state window and reports the largest gaps. Nsight also provides built-in gap and synchronization analysis. ([NVIDIA Docs][8])

## 5.3 Examine the ten largest gaps

For every major gap, answer:

> **Was eligible work available? If so, what prevented its next launch?**

Record the eligible queue, next request/batch, worker/thread state, blocking operation, and event that ended the gap.

This leads to a useful first decision table:

| Trace observation                            | First interpretation to test                                 | Next experiment                                       |
| -------------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------- |
| Long gaps; no eligible work                  | Upstream or client starvation                                | Replay real prepared inputs into the next stage       |
| Long gaps; eligible queue nonempty           | Scheduling, lock, wait, dispatch, or synchronization problem | Correlate queue state with the responsible CPU thread |
| Many tiny kernels separated by CPU gaps      | Launch/dispatch overhead                                     | Explicit compile A/B and graph-break inspection       |
| Continuous kernels with small actual batches | Spatial underfilling                                         | Fixed-shape batch sweep                               |
| Device continuously busy but throughput poor | Device execution or memory limitation                        | Targeted Nsight Compute                               |
| GPU becomes idle while sending audio         | Output backpressure or encoding                              | Controlled fast-sink versus real-client comparison    |

**Do not start with Nsight Compute across the entire server.** Kernel replay/serialization can change concurrency behavior and timing. Use it after Nsight Systems identifies the device work that matters. ([NVIDIA Docs][2])

# 6. Instrument the difference between concurrency and useful work

This is probably the most important serving-system distinction:

> **16 in-flight HTTP requests does not mean an AR batch of 16, a Flow batch of 16, or a HiFT batch of 16.**

Some requests may be waiting for references, some for speech tokens, some for acoustic decoding, and some for client delivery.

Record these independently:

| Stage             | Required observations                                                             |
| ----------------- | --------------------------------------------------------------------------------- |
| Ingress/admission | In-flight, admitted, queued, rejected                                             |
| Preprocessing     | Queue time, execution time, cache hit/miss, reference length                      |
| AR                | Ready/running requests, actual decode batch, graph/padded batch, context lengths  |
| Flow              | Eligible requests, readiness threshold, actual tensor shape, useful/padded frames |
| HiFT              | Its own batch, mel lengths, precision, output samples                             |
| Output            | D2H time/bytes, encoding time, pending bytes/audio seconds, blocked sends         |

For each dispatch, distinguish:

```text
queue waiting
host preparation / launch time
device completion
downstream delivery
```

CUDA work is asynchronous. A Python timer around `model(...)` generally measures submission rather than complete execution. CUDA events must cover the relevant stream dependencies. ([PyTorch Docs][5])

For live serving, sample event pairs and query completed pairs later. **Do not add `torch.cuda.synchronize()` after every request** to make timing convenient.

Add NVTX labels to the actual worker’s dispatch blocks, with stage and batch IDs. Do not wrap an `await` with a thread-local range and assume everything inside belongs to one request.

Use PyTorch Profiler only for a short, narrowed operator-attribution run. Shape, stack, and memory profiling can add overhead and affect tensor lifetimes; keep profiler-off results as the performance numbers. ([PyTorch Docs][9])

# 7. Bisect with faithful stage replays

When the trace identifies a suspicious region, remove adjacent stages one at a time while preserving real inputs.

| Replay                    | What you keep                                          | What you remove                                       | What it tells you                               |
| ------------------------- | ------------------------------------------------------ | ----------------------------------------------------- | ----------------------------------------------- |
| Prepared-reference replay | Real text and conditioning                             | Download/decode/reference extraction                  | Whether reference work limits downstream supply |
| AR-only                   | Real tokenized inputs and conditioning                 | Acoustic generation                                   | AR capacity and dispatch behavior               |
| Flow-only                 | Real speech tokens, masks, prompts, speaker embeddings | AR production delays                                  | Acoustic capacity when supplied work            |
| HiFT-only                 | Real mels and relevant state                           | AR and Flow                                           | Waveform-generation cost                        |
| Fast output sink          | Real completed output and necessary compute            | Normal encoding/network drain in a controlled variant | Output-path contribution                        |
| Full pipeline restored    | Original production-like workload                      | Nothing                                               | Whether the isolated gain survives contention   |

For every stage, sweep useful batch sizes **1, 2, 4, 8, 16**, separately across representative lengths.

Run two forms:

**Saturated replay:** all required inputs are ready immediately. This measures an upper bound.

**Cadence-preserving replay:** inputs become ready according to the real upstream timing. This tests scheduling and overlap under realistic dependencies.

If Flow is fast under saturated replay but mostly idle end-to-end, further Flow kernel work is probably not the first intervention.

Preserve masks, conditioning, cache state, prompt lengths, chunk position, and CFG pairing. Do not substitute zero tensors and assume the same control flow or cost.

For streaming, replay first, follow-up, and final chunks separately. Reset state correctly between trials.

**Never sum isolated stage capacities or multiply their speedups.** Recombine the pipeline and measure the outcome; the stages share resources and dependencies.

# 8. Apply the optimization that matches the evidence

## 8.1 CPU dispatch is limiting: compile, then investigate larger graph regions

For this model, explicit DiT compilation is the first low-friction candidate.

During a bounded diagnostic run, inspect:

```bash
TORCH_LOGS=graph_breaks,recompiles
```

Record whether compilation actually runs, which shapes recompile, and whether steady-state dispatch gaps shrink. Tensor scalar extraction and data-dependent control flow can break compiler graphs. ([PyTorch Docs][10])

The existing DiT compilation work targets the estimator; it does not establish that the entire acoustic loop or every other stage is compiled. ([GitHub][11])

If estimator compilation helps but substantial loop/dispatch overhead remains, evaluate a larger graph-safe region using real fixtures.

CUDA graphs require stable execution structure and memory addresses. Plan shape buckets, buffer ownership, capture safety, masks, output lifetimes, and fallbacks. AR graph support does not automatically graph Flow or HiFT. ([PyTorch Docs][5])

**Acceptance:** higher profiler-off goodput, fewer critical-path gaps, bounded warmup/recompile costs, and unchanged required semantics.

## 8.2 Actual batches are too small: tune the limiting stage, not just HTTP concurrency

The retrieved configuration includes a vocoder batch cap of 16 and a 30 ms wait setting, alongside frame-based admission policies. Those settings do not guarantee a batch of 16. ([GitHub][12])

Use a controlled screening grid:

| Setting                | Suggested experimental values                             |
| ---------------------- | --------------------------------------------------------- |
| Maximum acoustic batch | 1, 2, 4, 8, 16                                            |
| Peer/batch wait        | 0, 1, 2, 5, 10, 30 ms                                     |
| Frame budget           | Half, current, double—subject to memory safety            |
| Length-grouping policy | Current policy versus a controlled tighter/looser variant |

These are test values, not recommended production defaults.

For every batch, log why eligible requests were excluded: request cap, frame budget, length spread, padding budget, readiness, deadline, or another constraint.

Track:

```text
useful frame slots
allocated frame slots
repeated streaming-context work
batch waiting
first-audio and playback slack
```

Larger batches are not automatically better when padding or waiting dominates. Flow batching must also preserve masks and classifier-free-guidance pairing. ([GitHub][13])

**Acceptance:** more useful work per dispatch and better goodput without unacceptable latency, waste, or fairness regressions.

## 8.3 Preprocessing is starving the GPU: tune concurrency and native threads together

Separate reference fetching, audio decoding, resampling, tokenization, speaker extraction, and speech-token extraction.

Test preprocessing concurrency **1, 2, 4, 8**, then test ONNX intra-op threads **1, 2, 4** against the current setting.

Do not increase both aggressively at once. ONNX Runtime provides its own threading and spinning controls; those need to match actual CPU resources. ([Onnx Runtime][14])

Inspect the **actual ONNX session’s execution providers**, not merely which providers are globally available.

Test cached and uncached reference preparation separately. The inspected reference builder deliberately avoids using a mutable HTTP URL alone as a cache identity. 

Preserve thread-safety boundaries. The preprocessing concurrency change parallelizes independent preparation while retaining necessary finalization coordination. ([GitHub][15])

**Acceptance:** shorter preparation queues, earlier eligible GPU work, less CPU throttling, and improved full-pipeline goodput.

## 8.4 Streaming creates fragmentation: measure first-hop and follow-up behavior separately

The current streaming scheduler already supports grouping eligible chunks and ingesting new work between decode steps. Do not assume the path is permanently batch one. First-flush/prompt alignment handling also changed, so inspect your installed readiness logic. ([GitHub][16])

Measure:

```text
request admitted
→ enough speech tokens available
→ acoustic chunk eligible
→ peer wait ends
→ Flow starts/ends
→ HiFT starts/ends
→ first playable audio arrives
```

Keep the trained hop/lookahead behavior unchanged for the first optimization pass.

Then test existing hop-growth choices and deadline-aware waits. Smaller chunks may improve first audio but introduce more launches, repeated context, copies, and output operations. Larger chunks consume latency slack.

**Acceptance:** better audio goodput with acceptable TTFA and playback continuity—not merely faster final completion.

## 8.5 Copies and synchronization dominate: remove unnecessary host dependencies

Investigate hot-path occurrences of `.item()`, `.tolist()`, `.cpu()`, CUDA tensor printing, explicit synchronization, repeated device/dtype conversion, and small repeated allocations.

An occurrence is not automatically a bug. Ask whether its result is required **immediately** or could be materialized later or in a group.

For asynchronous copies, verify pinned-memory requirements, stream ordering, buffer lifetime, and consumer readiness. `non_blocking=True` is not a complete correctness or overlap solution by itself. ([NVIDIA Docs][17])

For ONNX CUDA stages that can consume and produce device-resident buffers, evaluate I/O binding to avoid implicit round trips. ([Onnx Runtime][18])

Also inspect the operations around IPC. A transport described as zero-copy can still be surrounded by conversions and copies.

**Acceptance:** less blocking on the actual critical path, correct ownership, bounded memory, and improvement with real clients.

## 8.6 TensorRT is promising: verify the physical execution batch

Treat TensorRT as a separate candidate, with DiT compilation disabled.

The important model-specific trap is that the inspected wrapper can split a larger CFG batch into supported request pairs and fall back to PyTorch for unsupported frame lengths. Its ONNX attention behavior also differs from the dynamic PyTorch streaming path. ([GitHub][19])

Record **actual enqueue shapes, enqueue count, profile misses, and fallback time**.

A scheduler reporting `batch=16` is not enough evidence that the accelerated backend executes one large device batch.

**Acceptance:** full-pipeline improvement across the real length distribution, including streaming-quality validation.

## 8.7 Device execution dominates: use Nsight Compute on a selected replay

Select a repeated kernel and representative shape from the system trace. Profile only a small number of warmed launches.

Inspect launch geometry/waves, occupancy constraints, eligible warps, memory traffic, throughput, register/shared-memory pressure, spills, and instruction mix. Use the installed tool’s section and metric catalogue. Kernel filters and launch counts make this targeted approach possible. ([NVIDIA Docs][20])

The likely interventions now become specific: better batching/packing, less padding, fusion, improved layout, a better kernel implementation, or validated reduced precision.

For a small dense AR workload, a useful rough model is:

```text
BF16 weight bytes ≈ 2P
matrix-multiply work for batch B ≈ 2PB FLOPs

weight-only arithmetic intensity ≈ B FLOPs/byte
```

This ignores KV, activations, cache effects, and other operations. It illustrates why batch 16 does not automatically make a small decoder compute-bound.

**Acceptance:** lower absolute execution time for representative work, followed by improved unprofiled serving goodput. NCU replay results alone are not a production-capacity measurement. ([NVIDIA Docs][2])

## 8.8 The efficient single replica still leaves capacity: test replicas/MPS

Only now compare one, two, and four same-GPU replicas.

At fixed total concurrency 16, compare:

```text
1 replica:   16 requests
2 replicas:   8 + 8
4 replicas:   4 + 4 + 4 + 4
```

Then separately test each arrangement’s maximum SLO-qualified capacity. Keep CPU allocation and memory budgets explicit.

MPS is relevant to overlap among suitable independent processes. It does not repair a serial queue or create work inside one underfed pipeline. ([NVIDIA Docs][21])

**Do not set `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=30` to “target 30% utilization.”** That variable constrains available client resources; it is not a desired utilization setting. ([NVIDIA Docs][22])

Account for duplicated weights, graph reserves, CPU pools, reduced per-replica batch sizes, router fairness, and tail interference.

For a small model, tensor parallelism or stage disaggregation should also be evidence-driven placement experiments—not automatic responses to a low SM reading.

# 9. Secondary opportunities and production gates

Once the main limiter is established, consider quantization, attention-kernel changes, conditioning/prefix reuse, speculative speech-token decoding, fewer Flow steps, or distillation.

Treat these as separate projects with explicit backend-support and quality gates. Do not assume every generic SGLang optimization is supported by this checkpoint, or that a model-quality change is equivalent to an implementation optimization.

For deployment qualification, include:

| Area                | Required checks                                                                                    |
| ------------------- | -------------------------------------------------------------------------------------------------- |
| Audio correctness   | WER/CER where appropriate, speaker similarity, listening, clipping, silence, repetitions, duration |
| Streaming integrity | First/follow-up/final chunks, offsets, overlap, no missing or duplicated samples                   |
| Batch correctness   | Different lengths and references together, masks, CFG pairing, output ordering                     |
| Scheduling          | Mixed-length fairness, cancellation, overload rejection, bounded queues                            |
| Resource stability  | Long-run memory/cache/graph growth, cleanup, CPU throttling                                        |
| Operations          | Cold readiness, engine rebuild, replacement, slow clients, noisy neighbors                         |

Run repeated A/A tests first to estimate noise, then alternate A/B/B/A using the same workload and cache policy. Do not accept an apparent speedup explained by shorter audio, extra rejection, or changed output quality.

If the service meets its requirements but genuinely needs little GPU capacity, compare smaller hardware or safe colocation on **cost and qualified output**, not the percentage produced by a smaller denominator.

# 10. The order I would follow on your deployment

| Priority | Action                                                                 | Result needed before proceeding                                     |
| -------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 1        | Identify the exact “2%” metric and installed source/configuration      | The measurement and execution path are trustworthy                  |
| 2        | Explicit eager/compiled DiT comparison at c1 and c16                   | A measured result, with actual backend use verified                 |
| 3        | Warmed c16 Nsight Systems trace plus actual stage-batch counters       | Starvation, dispatch, underfilling, or device limitation identified |
| 4        | Faithful replay of the limiting stage                                  | The trace’s suspected cause is reproduced or rejected               |
| 5        | Targeted batching, preprocessing, copy, or graph changes               | Better unprofiled goodput with latency/quality preserved            |
| 6        | TensorRT, kernel work, or same-GPU replicas where justified            | Improvement survives full-pipeline contention                       |
| 7        | Rate-driven, mixed-workload, quality, and sustained-load qualification | A defensible production operating point                             |

**The core principle is: find out why eligible useful work is not executing, fix that cause, and remeasure.** For this CosyVoice3 case, DiT dispatch and real acoustic batching deserve early attention; MPS and custom kernels belong later unless the trace specifically points there.

The most useful evidence packet for turning this into concrete code changes is your resolved launch configuration, GPU model/UUID, exact utilization metric, c1/c16 results, and a warmed c16 Nsight Systems trace.

[1]: https://github.com/sgl-project/sglang-omni/pull/1969 "[CosyVoice3] Compile the DiT estimator by default by db-ol · Pull Request #1969 · sgl-project/sglang-omni · GitHub"
[2]: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html "https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html"
[3]: https://docs.nvidia.com/deploy/nvidia-smi/index.html "docs.nvidia.com"
[4]: https://github.com/sgl-project/sglang-omni/pull/1715 "[Perf][Fun-CosyVoice3] Fix vocoder autocast so bf16 config actually applies by YijunWang1121 · Pull Request #1715 · sgl-project/sglang-omni · GitHub"
[5]: https://docs.pytorch.org/docs/2.9/notes/cuda.html "https://docs.pytorch.org/docs/2.9/notes/cuda.html"
[6]: https://raw.githubusercontent.com/sgl-project/sglang-omni/main/benchmarks/eval/benchmark_tts_seedtts.py "https://raw.githubusercontent.com/sgl-project/sglang-omni/main/benchmarks/eval/benchmark_tts_seedtts.py"
[7]: https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html "https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html"
[8]: https://docs.nvidia.com/nsight-systems/UserGuide/index.html "https://docs.nvidia.com/nsight-systems/UserGuide/index.html"
[9]: https://docs.pytorch.org/docs/2.9/profiler.html "https://docs.pytorch.org/docs/2.9/profiler.html"
[10]: https://docs.pytorch.org/docs/2.9/compile/programming_model.common_graph_breaks.html "https://docs.pytorch.org/docs/2.9/compile/programming_model.common_graph_breaks.html"
[11]: https://github.com/sgl-project/sglang-omni/pull/1670 "[Fun-CosyVoice3] torch.compile the DiT (flow decoder) backbone by CAICAIIs · Pull Request #1670 · sgl-project/sglang-omni · GitHub"
[12]: https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/config.py "https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/config.py"
[13]: https://github.com/sgl-project/sglang-omni/pull/1663 "[Perf][Fun-CosyVoice3] Batch inference for Flow decoder by nagisa-kunhah · Pull Request #1663 · sgl-project/sglang-omni · GitHub"
[14]: https://onnxruntime.ai/docs/performance/tune-performance/threading.html "Thread management | onnxruntime"
[15]: https://github.com/sgl-project/sglang-omni/pull/1755 "[Perf] [CosyVoice3]: parallelize reference preprocessing by charliechenye · Pull Request #1755 · sgl-project/sglang-omni · GitHub"
[16]: https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py "https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py"
[17]: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html"
[18]: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html "https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html"
[19]: https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py "https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py"
[20]: https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html "https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html"
[21]: https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html "https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html"
[22]: https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html "https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html"
