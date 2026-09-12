# Diagnosing Low GPU Utilization in SGLang-Omni TTS Serving

The objective is more correct, deadline-compliant audio per GPU-second—not an arbitrary SM percentage. A reported 2% is a symptom until the metric, measurement window, ready-work supply, and execution path are identified. This playbook assumes Linux and an NVIDIA CUDA GPU. It does not assume a particular GPU, installed commit, service-level objective, or root cause. No measurement of the target deployment has been performed.

The public sources were inspected on September 12, 2026. Repository `main` is mutable; the installed source and effective configuration take precedence over this snapshot. Recommendations, example experiment grids, and acceptance criteria below are proposed engineering procedures, not measured performance claims.

## 1. The shortest evidence-based route

Begin with an explicit eager-versus-compiled DiT comparison at concurrency 1 and 16, while checking the effective Flow dtype and keeping HiFT precision unchanged. Next, capture a warmed 20-second CUDA timeline at concurrency 16. Determine whether the GPU is waiting for work, receiving many tiny launches, executing undersized batches, or spending time in a genuinely limiting device operation. Optimize the category the trace establishes.

This is unusually relevant to this checkpoint: the exact-model default-on compilation PR reported CPU-bound eager DiT dispatch at concurrency 16. Its author's paired H200 experiment reported roughly 53% more audio throughput. That result motivates a test; it neither diagnoses a different installation nor predicts a 2%-to-30% improvement.[^compile-default]

There is a source-version trap. That default-on change merged on September 7, but the factory source retrieved for this review has `enable_dit_torch_compile=False`. Explicitly select the path and verify that compiled kernels execute. Never infer effective behavior from a merged PR title.[^factory]

## 2. Establish what the percentage measures

| Measurement | Interpretation | What it does not establish |
|---|---|---|
| `nvidia-smi` GPU utilization | Fraction of the sampling period with GPU kernel execution | Fraction of all SMs occupied, achieved FLOPs, or useful work |
| DCGM `SM_ACTIVE` | Time with an active warp, averaged across SMs | Whether time gaps or spatial underfilling caused a low value |
| SM occupancy | Resident warps relative to supported capacity | Instruction issue efficiency or application speed |
| Tensor activity | Tensor-pipeline activity | Whether the complete TTS service is efficient |
| DRAM activity | Memory-traffic activity | Allocated memory, or a standalone proof of bandwidth saturation |
| CUDA kernel union coverage | Time covered by at least one captured CUDA kernel | SM activity or activity in untraced processes |

These distinctions follow NVIDIA's measurement definitions. A warp can remain active while waiting for memory; resident work is not identical to productive instruction issue.[^smi][^dcgm-fields]

For intuition, imagine an average SM-active fraction as the product of how often work is present and how much of the device it covers while present. Ten percent temporal activity with twenty percent spatial coverage gives two percent overall. So can fifty percent temporal activity with four percent coverage. These are illustrative decompositions, not identities between differently sampled tools.

Do not add overlapping kernel durations and divide by wall time. Use the union of intervals on the same physical device. The included `analyze_nsys_sqlite.py` does this, requires explicit window boundaries, and labels the result as captured-kernel coverage rather than SM utilization.

Check the denominator. Device ordinals inside `CUDA_VISIBLE_DEVICES`, DCGM host ordinals, MIG instances, MPS-restricted contexts, and physical GPU UUIDs need not identify the same resource. Unsupported or blank counters are not zero. Capture the actual metric name, unit, normalization, interval, aggregation expression, GPU UUID, and process coverage with every report.

## 3. Define a production objective before optimizing

Use a constrained capacity objective:

`maximize correct audio seconds completed per wall second per GPU`

subject to acceptable first-audio latency, completion latency, streaming deadlines, error rate, memory safety, and audio quality. Add actual infrastructure cost when comparing different hardware or replica counts.

A useful request result contains output samples, sample rate, submission time, first playable-audio time, completion time, error status, and generated speech-token count. Derive audio duration from sample count and sample rate—not HTTP byte count unless the raw PCM format is known. A WAV header is not audio; a network fragment is not necessarily one model chunk.

Keep these quantities separate:

| Quantity | Definition and use |
|---|---|
| Request throughput | Successful requests / elapsed measurement time; sensitive to utterance-length mix |
| Audio throughput | Sum of successful output durations / elapsed measurement time |
| Request RTF | End-to-end request latency / that request's audio duration; includes queueing if measured at submission |
| Aggregate RTF | Wall time / total generated audio duration; inverse of aggregate audio throughput, not inverse of mean request RTF |
| TTFA | Submission to first playable audio; also record server-side first audio separately |
| Streaming stalls | Playback-buffer underruns or missed chunk deadlines after playback begins |
| SLO-qualified goodput | Only useful outputs meeting the explicitly stated quality and latency acceptance policy |

Record p50, p95, and p99, but do not describe a few hundred requests as a reliable extreme-tail characterization. Use a short run for diagnosis, then a larger independent run for production tails. Report sample counts and failures alongside percentiles.

For streaming continuity, maintain a client-side buffer model: initialize it to the chosen playback startup buffer, subtract elapsed real time, and add received playable audio duration. Log every negative-buffer interval. Specify whether a stream with one missed deadline counts as failed goodput or contributes only its timely audio.

If a faster implementation finishes the same offered traffic with less GPU activity, that can be success. Increasing padding, redundant work, batch waiting, or replica count solely to raise a gauge is not.

## 4. Model-specific facts that change the investigation

The pipeline has reference/text preprocessing, an autoregressive speech-token engine, and acoustic decoding through Flow/DiT and HiFT. The `0.5B` model name must not be used as the size or cost of the entire service. The cookbook describes a separate 22-layer DiT and 25 Hz speech tokens; characterize the acoustic stages independently.[^cookbook]

Historical exact-model work provides useful starting hypotheses:

| Change | Why it matters to this investigation |
|---|---|
| BF16 autocast correction, PR #1715 | An old implementation could silently run the vocoder path in FP32 despite BF16 configuration. Audit actual operator execution, not just parameter dtype. |
| Flow batching, PR #1663 | Request batching must preserve per-request conditioning, masks, and classifier-free-guidance pairing. |
| DiT compilation, PR #1670 | Launch/dispatch overhead in the repeatedly executed estimator is an explicit optimization target. |
| Preprocessing concurrency, PR #1755 | Serial reference conditioning can starve the downstream engine. |
| TensorRT estimator, PR #1858 | There is an alternative backend, but its batching and shape behavior must be measured. |
| Streaming path, PR #1656 | First-hop readiness and inter-stage overlap need separate analysis from final-response latency. |

These are source-supported historical changes, not a claim that an arbitrary environment includes them.[^dtype][^flowbatch][^compile][^preprocess][^trt-pr][^stream-pr]

The retrieved configuration sets preprocessing concurrency to 8 and a vocoder batch cap of 16. It also includes a 30 ms batch-wait setting and frame/padding-based Flow admission. Those are ceilings and policies, not evidence that a run contains 16 useful requests in every device batch.[^config]

The current factory keeps HiFT precision separately configurable and defaults it to FP32. Do not “fix” a historical BF16 problem by forcing every acoustic operation to BF16. First identify the effective Flow precision and retain the validated HiFT setting while isolating other changes.[^factory]

One additional backend trap: the inspected TensorRT wrapper handles larger classifier-free-guidance batches by splitting them into supported request pairs and falls back to PyTorch for out-of-profile frame lengths. A scheduler batch of 16 does not prove a TensorRT enqueue of 32 CFG rows.[^trt-source]

## 5. Freeze a reproducible baseline

Run the environment collector with the Python environment and container used by the server:

```bash
python /path/to/cosyvoice_diagnostics/collect_env.py \
  --output results/environment.json
```

The only path to adapt is the extracted diagnostic-kit directory. The collector records package versions, source hashes and factory signatures, Git state, GPU topology, CPU information, selected non-secret environment variables, and visible cgroup limits. It does not load a model or mutate the server.

Attach the effective server launch command and fully resolved stage/engine configuration. Source defaults are not resolved values. Record the checkpoint revision or local weight hashes, reference-input hashes, dataset order, generation settings, output format, warmup sequence, streaming mode, and result directory.

Audit the following before benchmarking:

| Area | Concrete check | Interpretation |
|---|---|---|
| Installation identity | Compare imported package path and its Git SHA with the checkout being edited | An editable install or worktree mismatch invalidates source assumptions |
| GPU identity | Match `nvidia-smi -L`, process GPU UUIDs, and container visibility | Avoid profiling an idle physical GPU while serving on another |
| Isolation | Inspect other GPU processes, CPU quota, memory pressure, and shared storage | Separate noisy-neighbor effects from code changes |
| Debug settings | Check `CUDA_LAUNCH_BLOCKING`, anomaly/debug paths, verbose token logs | Remove diagnostic serialization from the performance baseline |
| Precision and placement | Log actual input/output device and dtype for AR, Flow, and HiFT during one sampled request | A top-level dtype flag does not prove every operator's execution precision |
| Engine path | Record backend, graph capture/replay coverage, compilation status, and fallback counts | “Enabled” does not guarantee the measured shapes use it |
| Limits | Record ingress, engine, scheduler, batch, frame, and output-buffer limits | Find the narrowest cap, not just HTTP concurrency |
| Host resources | Inspect server-process affinity and cgroup CPU-throttling deltas | Collector-process limits may differ if run outside the service container |
| Hardware state | Record clocks, power/thermal limiting, memory use, and errors before/during load | Do not attribute hardware instability to a scheduler change |

Do not change shared-host power limits, clocks, MIG layout, MPS configuration, or monitoring globally without authorization. A read-only baseline is enough to identify these conditions.

## 6. Run explicit backend and concurrency comparisons

Run candidates sequentially on the same approved test GPU. Do not keep two candidate servers resident unless colocation is the experiment.

### A: explicit eager estimator

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.enable_dit_torch_compile false \
  --vocoder.factory.enable_flow_estimator_trt false
```

### B: explicit compiled estimator

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.enable_dit_torch_compile true \
  --vocoder.factory.enable_flow_estimator_trt false
```

These options are documented for the inspected integration. An older checkout may not expose them; check that checkout's CLI help and factory source rather than replacing unknown flags with invented ones.[^cookbook]

Exercise representative input lengths and batch sizes until compilation/cache misses no longer contaminate the steady-state comparison. Do not assume one warmup request covers every shape. Keep startup and cold-cache costs as separate deployment results.

Use the existing SeedTTS runner for a first streaming measurement:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 --lang en \
  --max-concurrency 16 --max-samples 256 --warmup 32 \
  --use-existing-server --generate-only --stream \
  --output-dir results/cosy_compiled_c16_stream
```

The checked runner supports these arguments, and streaming requests use PCM. Omit `--stream` for the corresponding buffered-client experiment. `--generate-only` separates generation performance from later ASR-based evaluation.[^benchmark]

Use `run_sweep.sh` to repeat the same procedure at concurrency 1, 2, 4, 8, and 16. Add 32 only as an approved diagnostic point. The script refuses to reuse an existing result directory and does not start or stop the server.

```bash
# From the SGLang-Omni checkout, after starting candidate B:
CONCURRENCIES="1 2 4 8 16" REPEATS=3 \
  bash /path/to/cosyvoice_diagnostics/run_sweep.sh compiled streaming
```

The 256-request default is a screening size, not a p99 certification. Repeat the leading candidates with the full evaluation set and a representative production distribution. Preserve fixed-token/conditioning replays separately from stochastic end-to-end generation so that shorter generated audio cannot masquerade as a speedup.

Run short, medium, and long utterances separately, then a mixed workload. Also separate repeated-reference and unique-reference traffic. Test at least one cold-reference run rather than allowing a repeated benchmark dataset to silently define the production cache-hit rate.

A client concurrency cap is not a constant arrival rate. First use closed-loop load to expose capacity curves; then use a rate-driven test that records intended arrival, actual send, admission, rejection, and completion. Ensure client semaphores do not hide offered-load queueing. Increase offered rate until latency or backlog fails the acceptance policy; do not call the highest throughput before an unbounded queue “production capacity.”

## 7. Collect low-overhead counters in the same window

On a host with DCGM configured and authorized, discover supported fields first:

```bash
dcgmi profile --list --entity-id gpu:0

dcgmi dmon --entity-id gpu:0 \
  --field-id 1001,1002,1003,1004,1005 --delay 1000
```

The GPU entity must match the server's physical GPU, not simply its CUDA-visible ordinal. These are current DCGM CLI forms. Older versions provide the corresponding `-i`, `-e`, and `-d` options; consult the locally installed help rather than guessing unsupported field names.[^dcgm-profiling]

Capture power and clocks alongside ordinary utilization:

```bash
nvidia-smi \
  --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,power.draw,clocks.sm \
  --format=csv --loop-ms=1000
```

Compare counters only within the workload's steady-state window. Check field compatibility and sampling/multiplexing before interpreting near-zero readings. Retain raw samples, not only a dashboard average that includes startup, drain, and idle time.[^dcgm-fields][^smi]

Do not run every hardware-counter profiler simultaneously. DCGM profiling can conflict with developer profilers. Its pause/resume operations affect the host engine, not only one GPU; coordinate any pause with monitoring owners and restore it afterward.[^dcgm-profiling]

CPU evidence matters equally. For the actual worker PID, collect thread CPU utilization, run-queue/context-switch information, and cgroup throttling during the same interval. For example, on a Linux system with `pidstat` installed:

```bash
# Replace 12345 with the observed worker PID.
pidstat -t -u -w -p 12345 1
```

If one thread is pegged while the GPU receives tiny bursts, investigate dispatch or Python work. If many runnable native threads exceed the CPU quota, investigate oversubscription. These are hypotheses to confirm with stack attribution, not automatic diagnoses from CPU percentage alone.

## 8. Capture a warmed Nsight Systems timeline

Use a dedicated test instance and one interactive Nsight session per user/container. Launch the actual server under the profiler so its worker descendants are in scope; profiling only a router or client is insufficient.

```bash
mkdir -p traces

nsys launch --trace=cuda,nvtx,osrt --sample=none \
  sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.enable_dit_torch_compile false \
  --vocoder.factory.enable_flow_estimator_trt false
```

After readiness and representative warmup, start a sufficiently long concurrency-16 benchmark in a second terminal. When the load has reached its measured steady phase, capture 20 seconds from a control terminal in the same user/container session:

```bash
nsys start --stop-on-exit=false -o traces/cosy_eager_c16
sleep 20
nsys stop
```

If a complete benchmark lasts less than the capture interval, increase its sample count or choose a shorter explicit steady-state interval. Do not count benchmark warmup or request drain as GPU starvation. Repeat the same capture for candidate B. Interactive launch/start/stop and CUDA/NVTX tracing are documented Nsight workflows.[^nsys]

Use the output report path printed by the tool. For the example basename:

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,osrt_sum \
  traces/cosy_eager_c16.nsys-rep

nsys export --type sqlite \
  -o traces/cosy_eager_c16.sqlite \
  traces/cosy_eager_c16.nsys-rep

python /path/to/cosyvoice_diagnostics/analyze_nsys_sqlite.py \
  traces/cosy_eager_c16.sqlite --describe
```

Select explicit nanosecond boundaries from the exported steady-state timeline, and the device ID returned by `--describe`. The following numbers are only an example of a 20-second SQLite-clock interval, not timestamps to copy blindly:

```bash
python /path/to/cosyvoice_diagnostics/analyze_nsys_sqlite.py \
  traces/cosy_eager_c16.sqlite --device 0 \
  --start-ns 10000000000 --end-ns 30000000000 \
  > traces/cosy_eager_c16.coverage.json
```

The analyzer merges overlapping kernels and lists the largest intervals with no captured kernel. It cannot establish why a gap exists. Examine those intervals in the GUI beside CUDA API calls, OS-runtime waits, NVTX ranges, and worker CPU activity. Nsight's analysis reports can also identify GPU gaps and synchronization; select the tools available in the installed release.[^nsys-analysis]

For each of the ten largest steady-state gaps, record: start/end, whether ready work existed, the oldest eligible request, the worker/thread that would launch next, its current CPU/wait operation, the next CUDA launch, and the state transition that ended the gap. Classify by this evidence rather than by visual appearance alone.

Use a second, shorter trace with Python sampling or stack capture only when the first trace identifies a CPU region needing attribution. Save the profiler-off benchmark as the performance result.

## 9. Instrument stages without serializing the service

Add stage and batch labels at the worker that performs the operation. Capture enqueue and eligibility separately: a request waiting for more speech tokens is not an eligible Flow item. Use sampled request IDs in traces, not unbounded request-ID labels in production metrics.

Recommended events and fields:

| Event | Required fields |
|---|---|
| Request accepted/admitted | Request ID, monotonic timestamp, input/reference lengths, streaming mode |
| Preprocessing queued/start/done | Queue and CPU times, cache hit/miss, reference bytes, provider/device |
| AR prefill/decode | Running/ready counts, actual batch, graph/padding batch, context lengths, speech tokens emitted |
| Flow eligibility/dispatch | Chunk index, first/follow-up/final flag, eligible count, admission rejection reason, useful/padded frames |
| Flow solve | Actual CFG tensor shape, timestep count, backend, fallback, CPU-submit time, device event span |
| HiFT decode | Independent batch and length distribution, dtype, device span, output samples |
| Output copy/encode/send | D2H bytes, encoder time, queued bytes/audio seconds, client backpressure |
| Cancellation/finalization | Stop reason, work discarded, buffer/KV/cache resources released |

Maintain separate measurements for queue waiting, host dispatch, and device completion. CUDA calls are asynchronous: a Python stopwatch around `model(...)` generally measures submission rather than completion. A device event bracket must cover the stream dependencies of the operation; an event on one stream does not automatically time work on an unrelated stream.[^cuda]

A simple dispatch marker is useful, provided it is not placed across an `await` that interleaves unrelated work on the same thread:

```python
from contextlib import contextmanager
import torch

@contextmanager
def dispatch_range(name: str):
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()

# In the actual GPU worker, around a synchronous host-dispatch block:
# with dispatch_range(f"flow/batch={batch_id}/requests={batch_size}"):
#     output = run_flow(batch)
```

This labels CPU dispatch; it is not a completed GPU duration. For sampled device timing, record start/end CUDA events around the actual stream work, retain a bounded queue of event pairs, and query completed pairs later. Do not synchronize after every request. For a microbenchmark, synchronization at the beginning and end of a whole timed trial is appropriate.

If the stage uses several streams, use its real completion event or join all relevant streams before recording the terminal timing event. Do not introduce extra joins into the serving path merely to simplify a metric. Measure the real critical path in the profiler instead.

Use PyTorch Profiler for operator/shape attribution in a short isolated worker run. Start without stack and memory tracing, then add them only for the narrowed problem; shape/stack collection adds overhead and can affect tensor lifetimes.[^torch-profiler]

## 10. Bisect the pipeline using faithful replays

Create a fixture set from real approved requests, recording actual intermediate tensors plus metadata. Include the short/medium/long length buckets and first/follow-up/final streaming chunks. Store dtype, shape, stride, length masks, prompt conditioning, speaker embeddings, token offsets, generation settings, and model revision. Protect voice/reference data as production-sensitive material.

| Experiment | Keep fixed | Remove or replace | Conclusion supported |
|---|---|---|---|
| Remote client vs localhost | Same requests, model, output mode, offered load | Network path and proxy placement | Size of transport/client contribution |
| Localhost vs internal submission | Same real frontend and stages | HTTP parsing/router envelope only | Size of ingress/router contribution |
| Cached conditioning replay | Same text and true conditioning | Reference download/decode/feature extraction | Whether reference preparation limits supply |
| AR-only fixture | Tokenized text and conditioning | Downstream acoustic service | AR capacity and per-step dispatch behavior |
| Flow replay | Real speech tokens, masks, speaker/prompt conditioning | AR production delays | Flow capacity when supplied work |
| HiFT replay | Real mel tensors and streaming state | AR and Flow | Waveform-generation cost and batch behavior |
| Output sink | Same completed waveforms and necessary copies | Real encoding/network drain, in a controlled variant | Output-path contribution |
| Full pipeline restored | Original workload | Nothing | Whether isolated improvement survives contention and dependencies |

For each isolated stage, sweep useful batch sizes 1, 2, 4, 8, and 16, subject to memory and supported shapes. Repeat per length bucket. Report per-item time, total trial wall time, actual device batch shapes, useful/padded work, and memory peak. Warm up first and pre-place fixture tensors when specifically testing device-resident capacity.

Measure two replay modes. A saturated replay tests an upper bound with work always ready. A cadence-preserving replay reproduces when upstream tokens and chunks become available. A large gap between them identifies scheduling/supply opportunity, not a guaranteed end-to-end speedup.

For a stateful stage, restore or reconstruct the initial state before each trial. Do not replay a final chunk through a first-chunk path, share mutable caches between unrelated requests, bypass conditioning, or replace valid masks with zeros. Keep the same correctness check before and after each transformation.

A useful standalone timing protocol is: initialize and place fixtures; run unmeasured warmup; synchronize once; record wall start; execute a fixed number of actual stage calls; synchronize once; record wall end. Use device events in a separate measurement or add a correctly scoped event pair. Confirm outputs are actually executed and not skipped by a cache or dead-code path.

Do not sum isolated-stage speedups or service capacities. Colocated stages compete for the same GPU, host threads, caches, memory, and launch resources. The end-to-end experiment remains the acceptance test.

## 11. Decision table: evidence to action

| Observed evidence | Discriminating test | Candidate intervention | Acceptance evidence |
|---|---|---|---|
| GPU gaps; no stage has eligible work | Replay real inputs directly into the next stage | Fix upstream preparation, readiness, or client supply | Eligible work arrives earlier; gaps and goodput improve |
| GPU gaps; ready queue is nonempty | Align scheduler wakeups, locks, and next launch | Remove excessive polling/waiting or blocking dispatch | Less ready-to-launch delay, same semantics |
| Dense tiny kernels with CPU gaps | Explicit eager/compiled A/B on fixed Flow shapes | Compile, fuse, then consider graphing stable regions | Fewer launches/gaps and faster unprofiled completion |
| HTTP c16; AR batch near one | Log running/eligible requests and engine caps | Admission/engine batching repair | Actual useful decode batch rises |
| Flow ready count high; Flow batches small | Log admission rejection reason and frame budget | Rebalance frame budget/length grouping/peer wait | Larger useful batches without tail or padding regression |
| Large padded batches | Track valid frame work and pad ratio per dispatch | Length-aware grouping, packed/masked kernels where supported | Lower wasted work and better goodput, not just higher activity |
| GPU continuously active; low device throughput | NCU on representative warmed kernels | Fix launch geometry, memory access, kernel choice, or precision | Absolute kernel and end-to-end time improve |
| High measured memory pressure | Roofline and actual bytes/bandwidth plus batch sweep | Weight reuse, lower bytes, validated quantization/fusion | Better throughput within numerical and memory constraints |
| One CPU thread saturated | Stack attribution at gap/dispatch region | Move/vectorize Python work; compile repeated GPU dispatch | Fewer host bottlenecks without adding hidden synchronization |
| Many native threads, CPU throttling | Thread-pool/concurrency grid | Reduce oversubscription and choose explicit CPU allocation | Less throttle/runqueue pressure, higher stage supply |
| Large D2H/sync footprint | Match copies/scalar extraction to callers | Defer/coalesce transfers and host materialization | Less blocking on critical path; valid output timing |
| GPU waits during output delivery | Fast-sink and normal-client comparison | Bounded async output, encoder/network repair | Goodput improves with real clients and bounded buffers |
| Performance drifts in repeated runs | Clocks, temperature, allocator/cache and queue telemetry | Fix instability, leaks, cache policy, or noisy neighbors | Stable long-run latency and resource usage |

No row is an instruction to change every corresponding parameter. Each is a hypothesis with a specific test and a falsifiable outcome.

## 12. Optimizations in priority order

### 12.1 Correct execution path and precision

First establish that the intended GPU backend runs, inference-only mode is active where appropriate, and the exact-model fixes are present. A parameter stored as BF16 does not guarantee every intermediate or library operator avoids FP32. Inspect a sampled stage's autocast context, inputs, outputs, and dominant kernels. Keep HiFT precision fixed initially; evaluate any later change with acoustic quality and chunk-boundary checks.[^dtype][^factory]

Do not mask a fallback by measuring only whole-service output. Count fallback invocations, frame lengths, and device time. A backend that handles easy short cases but falls back on the production tail can have a misleading average.

### 12.2 Compile repeated dispatch before writing custom kernels

The exact integration's DiT compilation work targets the estimator, not necessarily the complete Euler loop, AR path, or HiFT. Verify scope with trace labels and kernel changes. Dynamic lengths and data-dependent scalar/control operations can break compiler graphs; collect `TORCH_LOGS=graph_breaks,recompiles` during a bounded diagnostic run.[^compile][^torch-breaks]

Count recompilations after warmup. Separate cold compile latency, warm service latency, compiled-path hit rate, and memory growth. A throughput improvement that adds uncontrolled per-length compilation stalls may be unsuitable for interactive service.

Never remove a streaming/causal mask simply to obtain one graph. Replace data-dependent construction only with a mathematically equivalent implementation, test multiple prompt and hop lengths, and compare actual streaming audio.

### 12.3 Increase useful batching at the limiting stage

Record four numbers independently: HTTP in-flight requests, AR decode batch, ready Flow requests, and actual acoustic tensor batch. Keep HiFT batching independent as well. A configured max of 16 is not proof of any of these observed quantities.

Recommended small sweeps, not prescribed optimal settings:

| Knob | Screening values | What must remain controlled |
|---|---|---|
| Vocoder max batch | 1, 2, 4, 8, 16 | Length distribution, backend, precision |
| Batch wait | 0, 1, 2, 5, 10, 30 ms | TTFA and streaming deadlines |
| Frame budget | 0.5x, 1x, 2x the recorded baseline | GPU memory, padding, long-request fairness |
| Preprocessing concurrency | 1, 2, 4, 8 | ONNX/native-thread count and CPU quota |
| ONNX intra-op threads | 1, 2, 4, then recorded baseline | Preprocessing concurrency and affinity |

Change one dimension at a time for attribution. After finding individually promising settings, test combinations; interactions can reverse gains. At a fixed total concurrency of 16, adding replicas can reduce useful batch size per replica.

Track padding waste as `(allocated frame slots - valid frame slots) / allocated frame slots`, and account for prompt, generated, and overlapping context frames separately. Add work repeated for streaming context to the accounting even though it is semantically required rather than padding.

### 12.4 Remove preprocessing starvation and CPU oversubscription

Separate reference download, container decode, resampling, text normalization/tokenization, speaker extraction, and speech tokenization. Compare content-cached and unique-reference runs. Measure the real ONNX session's providers and profiling output; globally available providers do not prove a given session uses CUDA.

ONNX Runtime exposes intra-op threading and thread-pool/spinning controls. Tune them against the CPU allocation rather than the machine's advertised core count. Several concurrent requests each running a large native pool can spend more time competing than preparing work.[^ort-threads]

The exact-model preprocessing change parallelizes independent conditioning while preserving serialized finalization where required. Do not remove locks without demonstrating shared-model and cache thread safety.[^preprocess]

Reference caching is a serving optimization, not a free benchmark condition. Use an immutable content identity plus model/frontend configuration and correct tenant isolation. The inspected builder avoids treating a mutable HTTP URL alone as a safe cache identity.[^requests]

Moving all preprocessing to the GPU is not automatically helpful: test whether saved CPU time outweighs launch/copy overhead and interference with AR/Flow. Similarly, a separate CPU process can bypass interpreter contention but introduce IPC and memory-copy costs. Accept it only on full-pipeline measurements.

### 12.5 Streaming: optimize deadlines rather than only final throughput

Keep first-hop, follow-up, and final-hop paths separate. Measure the time for the first eligible acoustic chunk, its peer-batching wait, and the decode/copy/send sequence. A service can improve final request throughput while making TTFA or chunk jitter worse.

The inspected streaming scheduler can batch eligible chunks and interleave ingestion between decode steps; it is not simply a permanently batch-one decoder. Prompt alignment and first-flush handling also evolved, so instrument the installed readiness threshold rather than copying an older chunk formula.[^stream-source][^runner-source]

Do not assume changing client `stream=true` alone identifies the internal acoustic execution path. Log the path actually taken. Likewise, a non-streaming client may receive buffered results from an internally incremental pipeline.

For diagnosis, preserve the trained hop and lookahead settings, then compare existing hop-growth choices with the same texts and output checks. Smaller chunks can improve first audio but increase launch, overlap, transfer, and serialization work. Larger chunks can reduce overhead while consuming latency slack. Choose using a latency-goodput frontier, not an arbitrary hop size.

Evaluate deadline-aware grouping: cap peer waiting by the oldest eligible item's remaining playback slack, prioritize first hops only within a starvation-safe policy, and admit new work between decode steps. Use bounded queues measured in both bytes and audio seconds. A slow client must not create unbounded buffers or globally stall unrelated requests.

### 12.6 Copies, synchronization, allocation, and IPC

Search the hot path for per-token `.item()`, `.tolist()`, `.cpu()`, CUDA tensor printing, explicit synchronization, redundant device conversions, repeated concatenation/allocation, and expensive output encoding. A hit is an investigation site, not proof that the operation is removable.

Use trace attribution to decide whether a host materialization is required immediately or can be coalesced/deferred. Do not simply replace a blocking copy with `non_blocking=True`: buffer lifetimes, pinned storage, stream dependencies, and consumer readiness must all be correct. CUDA overlap depends on the memory and execution setup.[^cuda-best]

For ONNX CUDA stages whose inputs/outputs can remain device-resident, evaluate I/O binding rather than repeated implicit CPU transfers. It is useful only if the surrounding pipeline can consume those buffers safely.[^ort-io]

Inspect shared-memory, CUDA IPC, serialization, and ownership at actual stage boundaries. A “zero-copy” transport label does not prove no `.cpu()`, dtype conversion, or new allocation occurs before or after transport. Do not introduce a new process boundary until measuring its queue and transfer overhead.

Avoid per-request `empty_cache()` or broad allocator changes as speculative tuning. Profile allocations and memory growth first. Bound reference caches, in-flight tensors, graph caches, output buffers, and cancellation cleanup.

### 12.7 CUDA graphs after stable shape and dependency analysis

A CUDA graph reduces host dispatch for a graph-safe region by replaying fixed operations against stable addresses. That is distinct from general compiler fusion. Plan shape buckets, stable buffers, padding masks, warmup, capture safety, output lifetime, and fallback coverage. Replaying a graph for unsupported shapes or mutable state is a correctness problem, not merely a performance fallback.[^cuda]

AR graph support does not imply Flow or HiFT is graphed. If compiled DiT still leaves a significant Python/Euler dispatch footprint, evaluate a larger captured region using faithful fixtures before integration. Preserve noise/timestep semantics and classifier-free-guidance pairing.

### 12.8 TensorRT as a separate candidate

Run a separate candidate with compilation disabled and the estimator backend enabled:

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000 \
  --vocoder.factory.enable_dit_torch_compile false \
  --vocoder.factory.enable_flow_estimator_trt true
```

The integration rejects enabling both accelerators for the same estimator. Track engine identity, supported frame profile, actual enqueue shape/count, fallback rate, and warm/cold engine costs.[^config][^trt-pr]

Do not infer streaming numerical equivalence from buffered speed. The inspected wrapper notes frozen ONNX attention behavior, so streaming-quality validation is an explicit gate. Compare the complete path, not only TensorRT kernel time.[^trt-source]

### 12.9 Kernel work only after timeline localization

Select the dominant repeated kernel and its real representative shape from the system trace. Profile a deterministic stage replay, not the whole serving benchmark, with a small launch limit:

```bash
ncu --list-sets
ncu --list-sections

# Replace the regex with a kernel name observed in the trace and replay_flow.py
# with the faithful replay adapter built in section 10.
ncu --target-processes all --set detailed --clock-control none \
  --kernel-name 'regex:OBSERVED_KERNEL_PATTERN' \
  --launch-skip 20 --launch-count 5 \
  -o traces/flow_kernel python replay_flow.py
```

The replay adapter is intentionally model-specific and is not supplied as a fake universal benchmark. Its contract and fixture requirements are described in section 10. Adjust the skip count to exclude actual warmup launches of the selected kernel; it counts matching launches, not requests.[^ncu-cli]

Read launch geometry/waves, achieved occupancy, eligible warps, SM and memory throughput, register/shared-memory limits, spill traffic, and instruction mix. Inspect tensor-core use where supported by the actual operation and dtype. Use measured memory traffic and arithmetic work for a roofline assessment, not a utilization label alone.

Typical conclusions are: too few blocks, excessive padding, memory traffic, dependency stalls, low arithmetic intensity, poor kernel selection, or avoidable launch count. Choose fusion, layout changes, packing, validated reduced precision, or a custom kernel only for the measured limiter.

Nsight Compute can replay/serialize kernels and perturb clocks/cache behavior. Its throughput numbers are not end-to-end server capacity. Rerun the unprofiled serving benchmark after every kernel change.[^ncu-guide]

### 12.10 Same-GPU replicas, MPS, and hardware sizing

After a single replica is internally efficient but still leaves usable device capacity, compare one, two, and four independent replicas with explicit CPU and memory budgets. At fixed total concurrency 16, compare request distributions of 16, 8+8, and 4+4+4+4. Then separately test each layout's own SLO-qualified saturation capacity.

Compare MPS off/on as a separate factor, not bundled with unrelated scheduler changes. MPS can enable overlap across suitable independent processes. It does not manufacture ready work or repair serialization within one Python pipeline.[^mps]

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` restricts client resources; it is not a knob telling the GPU to reach a utilization target. Do not set it to 30 because the desired dashboard number is 30%.[^mps-env]

Account for duplicated weights, allocator/graph reserves, reference caches, KV capacity, CPU pools, router fairness, failure isolation, and noisy-neighbor tails. If one backend is already bandwidth-limited, overlapping more copies of it may merely intensify contention.

Do not use tensor parallelism just because the model is underutilizing a large GPU. Treat any split of a small stage across GPUs as a measured communication-versus-compute tradeoff. Stage disaggregation similarly adds transport and scheduling dependencies. First establish that placement, not local dispatch or batching, is the limiting factor.

MIG can provide isolation, and smaller GPUs or complementary co-tenants can improve economics. A higher percentage caused only by a smaller denominator is not proof of higher useful physical-GPU throughput. Compare cost and latency-qualified output directly.

### 12.11 Later algorithmic opportunities

Consider weight/KV quantization, fewer Flow steps, distillation, speculative speech-token decoding, prefix/conditioning reuse, and attention kernel changes only after the bottleneck and backend support are established. These can alter the model's numerical behavior, shape support, or speech quality. No claim is made that this checkpoint supports every generic SGLang optimization.

Reducing diffusion steps is a model-quality tradeoff. Speculative decoding requires a compatible proposal/verification strategy and measured acceptance/overhead. A faster AR engine cannot solve a Flow-dominated critical path; a faster vocoder cannot solve a reference-preparation bottleneck. Rank such work using end-to-end time contribution and engineering risk.

## 13. First-principles limits for a small autoregressive model

A simplified dense-weight estimate helps explain why a small decoder need not reach high compute utilization. If a layer's weights are read once for a batch, BF16 weight traffic scales roughly with `2P` bytes while the matrix multiplication work scales roughly with `2PB` FLOPs. Ignoring KV, activations, cache effects, and other operators, arithmetic intensity is approximately `B` FLOPs per weight byte. Batch 16 is therefore not automatically a compute-saturating workload.

This is a reasoning model, not a measured CosyVoice roofline or a FLOP count for the complete pipeline. Actual attention, acoustic decoding, cache behavior, and hardware balance must be measured.

With 25 speech tokens per second of generated audio, sixteen simultaneous real-time outputs would require about 400 aggregate speech tokens per wall second just to supply their autoregressive stream in an idealized steady state. That is a useful AR supply check, not a whole-service capacity guarantee: first audio, prompt work, burstiness, Flow, HiFT, and output deadlines remain additional constraints.[^cookbook]

For prioritization, use a simple critical-path bound. If the measured serial portion you are changing contributes 20% of a request's latency, making it infinitely fast cannot remove the other 80%. For streaming and shared-resource pipelines, derive this from the actual dependency timeline rather than adding stage durations that overlap.

## 14. Correctness and production qualification

An optimization passes only when faster useful execution survives the following gates:

| Gate | Required test |
|---|---|
| Text fidelity | Compare WER/CER and listen to representative failures; include punctuation, numbers, and language cases actually served |
| Speaker and prosody | Speaker similarity where available, human listening, stress cases with reference/style changes |
| Waveform validity | Sample rate, duration, NaN/Inf, clipping, silence, repetitions, missing segments |
| Streaming boundaries | First/follow-up/final chunks, overlap/crossfade, cache offsets, no duplicated or skipped samples |
| Batch correctness | Different lengths and references mixed together; correct masks, CFG pairing, output ordering |
| Numerical behavior | Fixed conditioning/token fixtures; tolerant signal checks, not only raw sample equality for stochastic paths |
| Scheduler behavior | Long/short fairness, prefill/decode interference, cancellations, deadlines, bounded queues |
| Resource stability | Sustained traffic, repeated failures, cache eviction, graph growth, allocator growth, output cleanup |
| Operational behavior | Cold startup, readiness, engine/cache rebuild, rolling replacement, overload rejection and recovery |
| Isolation | Slow clients, noisy neighbors, reference privacy, and failure containment |

Run acoustic/ASR evaluation separately from the generation-capacity window unless the deployed service genuinely colocates it. Avoid comparing a fast candidate that truncates more audio or rejects more requests with a complete-output baseline.

For A/B attribution, run repeated A/A first to estimate noise, then alternate A/B/B/A on the same GPU with the same workload and cache policy. Report both aggregate audio throughput and latency distributions. Do not multiply isolated speedup factors. Record which bottleneck becomes dominant after each successful change.

## 15. Recommended experiment order

| Order | Experiment | Proceed condition |
|---|---|---|
| 0 | Environment, metric identity, effective configuration, A/A repeat | The baseline is reproducible and measures the intended GPU/path |
| 1 | Explicit eager vs compiled DiT at c1 and c16 | Actual path verified; numerical outputs valid |
| 2 | Concurrency sweep and warmed Nsight timeline | A dominant starvation/dispatch/kernel/output category is identifiable |
| 3 | Faithful stage replay for that category | Isolated and cadence-preserving behavior explain the end-to-end trace |
| 4 | Stage-specific batch/wait/frame-policy sweep | More useful work without unacceptable padding or deadlines |
| 5 | Preprocessing concurrency/thread/cache variants when supply-limited | CPU and eligible-work measurements improve |
| 6 | Copy/sync/compile-break cleanup when visible on the critical path | Gaps or blocking work decrease in matched traces |
| 7 | TensorRT or larger graph region as separate candidates | Actual shapes/fallback and streaming quality pass |
| 8 | Targeted kernel changes only if device execution dominates | Unprofiled goodput improves |
| 9 | Same-GPU replicas/MPS or placement changes | Single-replica efficiency is understood and residual capacity is usable |
| 10 | Mixed-workload, open-loop, long-run, quality, and cold-start qualification | The accepted production goodput frontier moves outward |

The stopping criterion is not “30% SM utilization.” It is a defensible capacity point: no unexplained avoidable stalls with ready work; the limiting resource is understood; correct audio goodput is improved; latency and quality remain within policy; and the result is repeatable on the target deployment.

## 16. Evidence package for a concrete diagnosis

The most useful next investigation packet consists of the environment JSON, resolved launch/configuration, the exact metric behind “2%,” raw c1/c16 benchmark results, and one 20-second warmed c16 Nsight Systems report. Include first/follow-up/final-hop batch and ready-queue observations. With these, an investigation can move from hypotheses to specific source-level changes.

The supplied Python analyzer was tested on synthetic SQLite traces, including overlaps, clipping, empty windows, large timestamps, and read-only access. The shell runner was syntax-checked. Neither the GPU server nor profiler commands were executed against the target deployment; the bundle does not claim benchmark gains or compatibility with an uninspected older checkout.

## Sources

All implementation claims refer to the linked public sources. Mutable pages were inspected on September 12, 2026; pin local revisions for experiments. Versioned PyTorch documentation is used for stable timing/profiling principles, not as a statement of the installed version.

[^compile-default]: SGLang-Omni, PR #1969, DiT compilation default/performance discussion, merged September 7, 2026. https://github.com/sgl-project/sglang-omni/pull/1969
[^factory]: SGLang-Omni, `sglang_omni/models/fun_cosyvoice3/stages.py`, retrieved `main`, factory and acoustic execution. https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/stages.py
[^smi]: NVIDIA, *NVIDIA System Management Interface documentation*, utilization and query definitions. https://docs.nvidia.com/deploy/nvidia-smi/index.html
[^dcgm-fields]: NVIDIA, *DCGM Feature Overview*, profiling metric definitions and compatible groups. https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/feature-overview.html
[^cookbook]: SGLang-Omni, *Fun-CosyVoice3 cookbook*, model architecture and serving examples. https://raw.githubusercontent.com/sgl-project/sglang-omni/main/docs/cookbook/fun_cosyvoice3.md
[^dtype]: SGLang-Omni, PR #1715, BF16 autocast correction, merged August 29, 2026. https://github.com/sgl-project/sglang-omni/pull/1715
[^flowbatch]: SGLang-Omni, PR #1663, Flow batching, merged August 29, 2026. https://github.com/sgl-project/sglang-omni/pull/1663
[^compile]: SGLang-Omni, PR #1670, DiT compilation, merged August 30, 2026. https://github.com/sgl-project/sglang-omni/pull/1670
[^preprocess]: SGLang-Omni, PR #1755, preprocessing concurrency, merged August 30, 2026. https://github.com/sgl-project/sglang-omni/pull/1755
[^trt-pr]: SGLang-Omni, PR #1858, TensorRT Flow estimator, merged September 6, 2026. https://github.com/sgl-project/sglang-omni/pull/1858
[^stream-pr]: SGLang-Omni, PR #1656, streaming integration, merged September 7, 2026. https://github.com/sgl-project/sglang-omni/pull/1656
[^config]: SGLang-Omni, `fun_cosyvoice3/config.py`, retrieved `main`. https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/config.py
[^trt-source]: SGLang-Omni, `fun_cosyvoice3/flow_estimator_trt.py`, retrieved `main`, batching/profile fallback. https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/flow_estimator_trt.py
[^benchmark]: SGLang-Omni, `benchmarks/eval/benchmark_tts_seedtts.py`, retrieved `main`. https://raw.githubusercontent.com/sgl-project/sglang-omni/main/benchmarks/eval/benchmark_tts_seedtts.py
[^dcgm-profiling]: NVIDIA, *DCGM Profiling*, metric discovery, sampling, and profiler coexistence; page updated September 10, 2026. https://docs.nvidia.com/datacenter/dcgm/latest/learn/modules/profiling.html
[^nsys]: NVIDIA, *Nsight Systems User Guide*, interactive CLI, tracing, statistics, and export. https://docs.nvidia.com/nsight-systems/UserGuide/index.html
[^nsys-analysis]: NVIDIA, *Nsight Systems Post-Collection Analysis Guide*, gap and synchronization analysis. https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html
[^cuda]: PyTorch, *CUDA semantics*, version 2.9 documentation, asynchronous timing and CUDA graphs. https://docs.pytorch.org/docs/2.9/notes/cuda.html
[^torch-profiler]: PyTorch, *torch.profiler*, version 2.9 documentation, profiling activities and overhead. https://docs.pytorch.org/docs/2.9/profiler.html
[^torch-breaks]: PyTorch, *Common Graph Breaks*, version 2.9 documentation. https://docs.pytorch.org/docs/2.9/compile/programming_model.common_graph_breaks.html
[^ort-threads]: ONNX Runtime, *Thread management*. https://onnxruntime.ai/docs/performance/tune-performance/threading.html
[^requests]: SGLang-Omni, `fun_cosyvoice3/request_builders.py`, retrieved `main`, reference identity and preparation. https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/request_builders.py
[^stream-source]: SGLang-Omni, `fun_cosyvoice3/streaming_vocoder.py`, retrieved `main`, chunk scheduling and readiness. https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/streaming_vocoder.py
[^runner-source]: SGLang-Omni, `fun_cosyvoice3/model_runner.py`, retrieved `main`, AR streaming flush. https://raw.githubusercontent.com/sgl-project/sglang-omni/main/sglang_omni/models/fun_cosyvoice3/model_runner.py
[^cuda-best]: NVIDIA, *CUDA C++ Best Practices Guide*, asynchronous transfers, memory and execution overlap. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
[^ort-io]: ONNX Runtime, *I/O Binding*. https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
[^ncu-cli]: NVIDIA, *Nsight Compute CLI*, kernel selection and launch filtering. https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html
[^ncu-guide]: NVIDIA, *Nsight Compute Profiling Guide*, replay, serialization, clock control, and interpretation. https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
[^mps]: NVIDIA, *When to Use MPS*. https://docs.nvidia.com/deploy/mps/latest/when-to-use-mps.html
[^mps-env]: NVIDIA, *MPS Environment Variables*, active-thread limits. https://docs.nvidia.com/deploy/mps/latest/appendix-environment-variables.html
