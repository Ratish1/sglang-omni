# H100 Rust router CI-wide performance qualification

Status: **ready for H100 execution after remote preflight**

Record the tested repository revision and Rust binary hash at execution time.

Run this campaign only on the qualification H100 host. Its purpose is to find
the measured Rust-router operating point for every routed CI model and mode,
not merely to revisit metrics that moved during threshold calibration.

The campaign keeps model correctness, router performance, and worker capacity
as separate questions:

```text
current CI worker arguments
          |
          v
direct two-worker curve --------------------------+
          |                                       |
          v                                       v
Rust RR/LR concurrency curves -> select policy and K
                                          |
                                          v
                       one same-worker Python/Rust comparison
                                          |
                                          v
                       full existing CI correctness workload
                                          |
                                          v
                       update CI only from qualified results
```

## Scope

### Current two-H100 routed CI

Every row below currently runs through the Rust router. Ordinary DP2 rows use
two single-H100 workers. The MPS rows use the configured same-GPU process pool
(Higgs DP3 or MOSS DP2) and reserve the second H100 for later scoring. Both Rust
policies must be screened for every performance row, even when the existing
result improved.

| Family | Model or stage | Request modes | Correctness scope |
| --- | --- | --- | --- |
| ASR | Fun-ASR | SeedTTS EN non-stream; transcription SSE control | SeedTTS EN and ZH WER |
| ASR | Qwen3-ASR-1.7B | SeedTTS EN non-stream; transcription SSE control | SeedTTS EN and ZH WER |
| ASR | Whisper-large-v3 | SeedTTS EN non-stream; transcription SSE control | SeedTTS EN and ZH WER |
| ASR | MOSS-Transcribe-Diarize | movies800 non-stream and stream | movies800, AISHELL4 long, GoogleTime CER/cpCER/DER |
| TTS | Higgs TTS | WAV and PCM stream | full SeedTTS EN WER, similarity, UTMOS, stream consistency |
| TTS | MOSS-TTS Local | WAV and PCM stream | full SeedTTS EN WER, similarity, UTMOS, stream consistency |
| TTS | Qwen3-TTS 1.7B Base | WAV and PCM stream | full SeedTTS EN WER, similarity, UTMOS, stream consistency |
| TTS serving | Higgs mixed serving | REST WAV/PCM, batch, voice CRUD, WebSocket normal/stream, long generation | existing serving assertions and request accounting |
| MPS DP | Higgs and MOSS-TTS | WAV non-stream through the MPS worker pool | full SeedTTS EN WER and similarity |
| Qwen3-Omni | BF16 colocated DP2 TTS | audio output | SeedTTS-50 WER, UTMOS, request fields |
| Qwen3-Omni | FP8 colocated DP2 MMMU | text/image input, text output | MMMU accuracy |
| Qwen3-Omni | BF16 colocated DP2 MMSU | audio input, text output | MMSU accuracy |
| Qwen3-Omni | FP8 colocated DP2 Video-AMME | video input, text output | Video-AMME accuracy |

The Qwen3-ASR routers used later to score generated audio are evaluation
infrastructure, not additional generation workloads. Their performance is
covered by the Qwen3-ASR row; their output accounting remains part of every
audio-quality check.

This inventory comes from the current CI entry points and configuration:

- `tests/test_model/asr_ci_config.py` defines Fun-ASR, Qwen3-ASR, and Whisper;
- `tests/test_model/tts_ci_config.py` defines Higgs, MOSS, and Qwen3-TTS;
- `tests/test_model/test_asr_ci_multi_speaker.py` owns MOSS-TD;
- `tests/test_model/test_tts_serving_ci.py` owns the mixed Higgs stage;
- `tests/test_ci/test_tts_mps_dp2.py` owns Higgs/MOSS MPS validation;
- `tests/test_model/conftest.py` distinguishes routed Qwen3-Omni DP2 fixtures
  from direct disaggregated and TP2 fixtures;
- `.claude/skills/tune-ci-thresholds/models/{asr,tts,omni}/config.yaml` lists
  every calibrated model stage and result artifact.

### Qwen3-Omni stages that are direct in two-H100 CI

These jobs do not currently exercise a router. One worker consumes both CI
GPUs, so inserting a router cannot demonstrate multi-worker scheduling:

| Stage | Current worker topology |
| --- | --- |
| Thinker length | one BF16 TP2 worker |
| MMMU Talker | one BF16 disaggregated thinker/talker worker |
| MMSU Talker | one FP8 TP2 thinker/talker worker |
| Video-MME text and Talker | one BF16 disaggregated thinker/talker worker |
| Video-AMME Talker | one FP8 TP2 thinker/talker worker |
| Process replicas | integration test rather than a performance benchmark |

Keep these as their existing two-GPU direct CI correctness runs. They are not
part of the router performance matrix because the current topology has no
second worker for the router to select.

## Fixed controls

Use the following controls throughout:

- one candidate commit and one `cargo build --release --locked` binary hash;
- SGLang 0.5.19, the same container digest, dependency freeze, model revisions,
  datasets, CPU set, and GPU clocks or power policy;
- unseeded model generation, matching normal serving;
- the exact current CI worker arguments during router selection;
- one persistent worker set for all direct, RR, LR, and Python measurements of
  a model and mode;
- identical request bytes, sample order, warmup, and concurrency at each
  comparison point;
- a router restart when policy changes, followed by an untimed warmup;
- no local macOS performance evidence;
- no threshold, model argument, or CI concurrency changes during measurement.

Capture the full worker command line, environment, model revision, dataset
revision, container digest, binary hash, CPU affinity, GPU assignment, and
software versions in the run manifest.

## Automation

Do not execute this matrix as hand-maintained shell history. The runnable
package is `tasks/rust/router-e2e`; its README contains the direct H100
commands. The package provides:

- `manage_workers.py` owns persistent worker processes;
- `run_candidate.py` starts one temporary Rust or Python router, measures its
  process group, executes a benchmark command, captures diagnostics, and stops
  only the router;
- `run_direct_pair.py` starts benchmark clients against the workers at the same
  time and calculates throughput over their shared wall interval;
- `run_repo_benchmark.py` invokes the existing Omni benchmark entry points;
- `router_microbench.py` owns the real-socket proxy-only matrix;
- `run_matrix.py` generates current-schema Rust RR/LR configs and runs the
  requested direct, Python RR/LR, and Rust RR/LR points in order;
- `render_router_config.py` generates a current CI config for a manual router
  run.

One model and mode runs as:

```text
launch one persistent worker set
  -> direct curve
  -> Python RR curve
  -> Rust RR curve
  -> Python LR curve
  -> Rust LR curve
  -> select valid policy and concurrency
  -> one additional Python/Rust pair at that point
  -> full correctness/scoring
  -> zero-resource and cleanup audit
```

For example, after the two Qwen3-TTS workers are ready on ports 8011 and 8012:

```bash
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_matrix.py \
  --topology tts \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --worker-url http://127.0.0.1:8011 \
  --worker-url http://127.0.0.1:8012 \
  --concurrencies 8,16,32,64,96,128 \
  --rust-binary "$RUST_ROUTER_BIN" \
  --output-dir results/qwen3-tts-wav -- \
  "$OMNI_PYTHON" -m benchmarks.eval.benchmark_tts_seedtts \
    --use-existing-server --generate-only \
    --port '{router_port}' \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --ref-format references \
    --concurrencies '{target_concurrency}' \
    --output-dir '{output_dir}/audio' --disable-tqdm
```

The benchmark template uses the existing benchmark interfaces. Full-corpus
selection remains benchmark-specific: ASR receives `--max-samples 0`, while
TTS omits `--max-samples`; zero means an empty TTS dataset and must never be
passed there.

- ASR: `benchmarks.eval.benchmark_asr_seedtts --concurrencies ...`;
- standalone TTS: `benchmarks.eval.benchmark_tts_seedtts --generate-only
  --use-existing-server --concurrencies ...`;
- Omni and video benchmarks: invoke their existing one-concurrency entry point
  once per `K`;
- mixed serving: `benchmarks.eval.benchmark_tts_serving` with the selected
  schedule;
- MPS: the existing `tests.utils.tts_mps_runtime` lifecycle and benchmark
  functions.

`run_matrix.py --print-plan` prints every resolved command without starting a
process. A failed child stops the matrix immediately and leaves the completed
and failed trial evidence intact. Model paths, revisions, worker arguments, and
correctness thresholds continue to come from the current CI presets and
fixtures rather than the runner.

## Measurement method

For every model and request mode:

1. Measure the direct two-worker curve. Split aggregate concurrency `K` across
   both workers and calculate aggregate throughput over their shared wall
   interval.
2. Measure Rust `round_robin` and Rust `least_requests` at every declared `K`.
3. Select the highest-throughput valid Rust point. When throughput differs by
   at most 2%, select by p95 and then p99 latency. Record the complete curve;
   never report only the selected point.
4. At that exact `K`, run one Python `least_request` measurement and one selected
   Rust measurement against the same resident workers. Alternate which router
   runs first between successive model/mode comparisons.
5. Repeat only a point within the 2% ambiguity band, an isolated outlier, or a
   result inconsistent with adjacent points. Failed runs remain in the audit
   and never enter an aggregate.
6. Run the complete existing CI correctness stage through the selected Rust
   configuration.

For high-volume workloads, c128 is not an assumed endpoint. When both the
direct and selected-Rust curves improve by at least 3% from c96 to c128 with
valid tails, extend the same screen to c192 and c256. Stop at c256 because that
is the current CI admission envelope, and record that boundary rather than
calling the workers saturated.

Every matrix point uses the complete declared performance corpus. Never claim
a c64 or c128 result from a 20- or 50-request CI subset. Use the full source
dataset for the performance curve and retain the exact CI subset for the final
CI-contract replay. For generated audio, score every retained output directory;
do not infer quality from the selected point alone.

Record:

- throughput and completed/failed request counts;
- latency p50/p95/p99;
- TTFT or TTFA p50/p95/p99 and inter-chunk p95/p99 when applicable;
- RTF mean/p95/p99 and output tokens per request-second when applicable;
- router process-group CPU seconds/request and peak RSS;
- per-worker dispatch count and active-load high-water mark;
- worker queue/running-request metrics and GPU utilization/memory;
- router rejections, relay failures, probe outcomes, and resource usage before
  and after the run.

## Concurrency and policy matrices

### Router-only multithreading proof

Before model runs, test small JSON, 1 MiB upload, and chunked SSE relay at c1,
c8, c32, c128, and c512 against synthetic workers whose direct ceilings are
measured separately. Compare direct, Python round robin, Rust round robin, and
Rust least requests, with zero load-generator or HTTP errors required.

For small JSON and SSE at c512, pin only the Rust router process to 1, 2, 4, 8,
and 16 physical CPU cores while the load generator and workers remain on a
separate CPU set. Report throughput, p95/p99, CPU/request, and per-core CPU.
This is the explicit multithreading proof; GPU-bound model parity must not be
misrepresented as a router scaling limit.

### ASR

| Workload | Requests | Rust policies | Aggregate `K` |
| --- | ---: | --- | --- |
| Fun-ASR SeedTTS EN | 1,088 | RR, LR | 16, 32, 64, 96, 128 |
| Qwen3-ASR SeedTTS EN | 1,088 | RR, LR | 16, 32, 64, 96, 128 |
| Whisper SeedTTS EN | 1,088 | RR, LR | 16, 32, 64, 96, 128 |
| MOSS-TD movies800 non-stream | 800 | RR, LR | 8, 16, 32, 64, 96 |
| MOSS-TD movies800 stream | 800 | RR, LR | 8, 16, 32, 64, 96 |

Run a streaming control for all three SeedTTS ASR models at the selected
non-stream `K`. If streaming selects a different policy or changes throughput
by more than 2%, screen both policies at the adjacent lower and higher points
before selecting its own operating point.

AISHELL4 long and GoogleTime run once at the MOSS-TD selected policy with
concurrency bounded by their 20- and 25-sample corpus sizes. They are full
correctness and long-request tail checks, not high-concurrency scaling claims.

### Standalone TTS

| Model | Modes | Rust policies | Aggregate `K` |
| --- | --- | --- | --- |
| Higgs TTS | WAV, PCM stream | RR, LR | 8, 16, 32, 64, 96, 128 |
| MOSS-TTS Local | WAV, PCM stream | RR, LR | 8, 16, 32, 64, 96, 128 |
| Qwen3-TTS 1.7B Base | WAV, PCM stream | RR, LR | 8, 16, 32, 64, 96, 128 |

Use all 1,088 SeedTTS EN requests at every matrix point and for final
correctness. Generation remains unseeded. Preserve and score every retained
WAV/PCM output directory.

Qwen3-TTS currently configures 64 running requests and CUDA graph/compile batch
size per worker, so c128 is a meaningful aggregate endpoint. Do not copy those
worker settings to Higgs or MOSS. Their direct curves decide whether c96/c128
are useful.

### Higgs mixed serving

The current `stress.json` permits 128 envelopes but normally creates six
simultaneous envelopes at each collision. Changing only `max_concurrency` does
not increase offered load.

Run both policies under two complementary tests:

1. The existing 300-second schedule, unchanged, for CI behavior and every
   existing REST, batch, voice, WebSocket, and long-generation metric.
2. A qualification-only scaling schedule that preserves the workload mixture
   and request shapes while overlapping 1, 2, 4, 8, and 16 collision groups.
   This offers approximately 6, 12, 24, 48, and 96 simultaneous envelopes
   under the existing c128 safety ceiling.

The weighted 32-item batch load is part of Rust least-requests semantics.
Capture each selected worker and its active-load value at collision boundaries
so REST PCM tails can be attributed to worker placement rather than guessed
from aggregate latency.

### MPS DP

| Model | Rust policies | Aggregate `K` |
| --- | --- | --- |
| Higgs MPS pool | RR, LR | 8, 16, 32, 64 |
| MOSS-TTS MPS pool | RR, LR | 8, 16, 32, 64 |

Run the full 1,088-sample canonical benchmark at the selected point, including
overlap canary, WER, similarity, CPU/request, RSS, worker distribution, and MPS
cleanup evidence. MPS concurrency is selected independently of ordinary DP2.

### Routed Qwen3-Omni

| Stage | Performance input scope | Rust policies | Aggregate `K` |
| --- | --- | --- | --- |
| BF16 Omni TTS | full SeedTTS EN for screening; SeedTTS-50 for CI correctness | RR, LR | 8, 16, 32, 64, 96, 128 |
| FP8 MMMU text/image | full MMMU evaluation split | RR, LR | 4, 8, 16, 32, then 64 only when at least 256 requests are available |
| BF16 MMSU audio/text | 2,000 requests | RR, LR | 8, 16, 32, 64, 96, 128 |
| FP8 Video-AMME video/text | full Video-AMME evaluation split | RR, LR | 4, 8, 16, 32, then 64 only when at least 256 requests are available |

The four request shapes remain separate even when they use the same model.
Image upload, audio upload, video upload, and audio-output streaming exercise
different relay costs and can select different useful concurrency points.

## Worker-argument tuning boundary

Router selection and worker tuning are separate experiments. First complete
all matrices above with current CI worker arguments.

Only open a worker-tuning phase when the direct curve proves that a configured
worker limit, CUDA graph range, compile batch range, queue limit, or memory
partition is stopping throughput before the hardware saturates. For that
phase:

1. change one documented worker setting family;
2. measure the direct curve first;
3. retain it only when correctness passes and direct throughput or tails
   improve without OOM, allocator warnings, or reduced useful batch coverage;
4. rerun the selected Rust point against the new direct ceiling;
5. record the worker change independently from the router result.

Do not raise memory fractions or connection limits merely because headroom
exists. The direct worker curve, queue state, CUDA graph coverage, and GPU
telemetry must identify the actual constraint.

## Validity and acceptance

A point is valid only when:

- every expected request completes with the expected status and framing;
- both workers are healthy and both receive traffic;
- there are no unexpected 429/5xx responses, relay failures, or probe failures;
- admission, buffered bytes, worker load, listener slots, classification slots,
  and WebSocket sessions return to zero;
- the applicable WER, CER, cpCER, DER, accuracy, similarity, UTMOS, and stream
  consistency checks pass unchanged;
- generated WAV/PCM artifacts are readable and every expected sample is scored;
- worker and GPU state remain healthy through the timed interval.

The selected Rust result must satisfy all of the following:

- no unexplained throughput, p95, or p99 regression greater than 2% against
  Python at the same `K` and on the same workers;
- at least 20% lower router CPU/request, with RSS reported explicitly;
- no avoidable worker starvation or persistent load imbalance;
- a router-specific correctness failure is absent;
- a worker-saturation claim is made only when routed throughput is within 3%
  of the paired-wall direct-worker ceiling.

At worker saturation, throughput parity is the correct result; the router win
is lower CPU/RSS and equal or better tails. The Rust router's multi-threaded
ceiling is demonstrated by the high-concurrency router-only and fast-model
curves, not by claiming GPU-generated throughput that the workers cannot
produce.

## Evidence and final report

Store each point under a unique model/mode/policy/concurrency directory with
its request records, summary, router logs, diagnostics, worker logs, process
CPU/RSS, and GPU telemetry. Preserve invalid runs with their exclusion reason.

The final report contains:

1. the complete direct, RR, and LR curves for every row in this document;
2. the selected policy and concurrency for each model and mode;
3. one same-worker Python/Rust result at every selected point;
4. full existing CI correctness results for every model, including models that
   had already improved;
5. a separate result for every Qwen3-Omni input/output shape;
6. router CPU/RSS, worker distribution, GPU utilization, and zero-leak audit;
7. any worker-argument experiment as a separate attributable section;
8. exact proposed CI policy/concurrency changes, with no threshold changes yet.

Do not modify CI settings until the full report is complete. The final CI
configuration should encode the measured operating point for each stage rather
than one topology-wide policy or one concurrency copied across unrelated
models.
