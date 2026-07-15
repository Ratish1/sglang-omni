# Same-GPU DP validation harness

This directory measures several complete SGLang Omni TTS replicas sharing one
physical GPU, with and without CUDA MPS. It is an experimental validation tool,
not evidence that same-GPU DP is faster than a tuned single replica.

The harness deliberately uses the canonical SeedTTS client and
`/v1/audio/speech`:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --use-existing-server --generate-only
```

It does not use the older chat-completions DP load generator. Direct mode starts
one canonical client per replica at the same time. Each client runs the same
dataset and has a dedicated CPU set and output directory. The report follows the
review study's aggregate convention and sums canonical per-client QPS, audio
throughput, and output-token throughput. This is valid only for concurrently
launched clients using identical sample counts and settings; preserve the
per-worker values and reject visibly skewed or sequential windows.

## Safety contract

`run_condition.sh`:

- accepts only a GPU UUID (`GPU-...`) and exports it as
  `CUDA_VISIBLE_DEVICES`, so pipeline-local `cuda:0` consistently means that
  UUID and not a remapped physical ordinal;
- validates DP 1–10 CPU lists, online CPUs, and overlap among every server,
  client, and optional router set, and requires all of them to belong to the
  selected NUMA node when Linux exposes its CPU list;
- starts replicas sequentially and applies a bounded readiness timeout;
- creates one process group per replica, records its ID, and signals only those
  owned groups;
- starts MPS with short condition-unique pipe/log directories under
  `MPS_TMP_ROOT`, polls through asynchronous daemon startup, verifies every
  replica has a CUDA client listed by MPS `ps`, and quits only the daemon it
  started;
- asks MPS `terminate_client` to detach a stuck owned CUDA client before a final
  process-group `SIGKILL`; it never uses `pkill`;
- records commands, placement, software/hardware identity, topology, generation
  knobs, and MPS settings in the condition directory.
- gives MPS-off workers a condition-unique nonexistent pipe path, NVIDIA's
  documented MPS bypass, so an ambient default daemon cannot contaminate the
  control condition.

Run it as the same Unix user that owns the selected GPU workload. Start on an
otherwise idle GPU in compute mode `Default`. MPS is not an isolation boundary:
a fatal GPU fault can affect all MPS clients on that GPU.
The harness enforces both checks by default; `REQUIRE_IDLE_GPU=0` exists only for
deliberate interference experiments and should not be used for PR evidence.

## Configure one condition

All configuration is passed through environment variables. Required/recommended
settings are:

| Variable | Meaning |
|---|---|
| `GPU_UUID` | UUID from `nvidia-smi -L`; required except for dry-run |
| `DP` | replicas on the UUID: `1`, `2`, `3`, or `4` |
| `USE_MPS` | `0` or `1` |
| `SERVER_CORE_SETS` | semicolon-separated, dedicated set per replica |
| `CLIENT_CORE_SETS` | semicolon-separated, dedicated set per direct client |
| `NUMA_NODE` | memory node local to the GPU |
| `ROUTER_CORES` | dedicated physical cores for router mode |
| `ROUTER_POLICY` | router policy; defaults to `least_request` |
| `MEM_FRACTIONS` | `auto` to omit the CLI override, or one explicit value per replica; one value broadcasts |
| `CONCURRENCY_PER_WORKER` | canonical client concurrency per direct worker |
| `MAX_RUNNING_REQUESTS` | server generation batching limit |
| `CUDA_GRAPH_MAX_BS` | server CUDA graph maximum batch size |
| `MAX_TOTAL_TOKENS` | exact upper bound for each replica's SGLang KV-token pool |
| `MODEL` / `MODEL_NAME` | checkpoint and served/request model name |
| `CLIENT_DRIVER` | `seedtts` (default) or deterministic `manifest` replay |
| `TTS_MANIFEST` | JSONL request manifest required by `CLIENT_DRIVER=manifest` |
| `BENCH_LANG` | SeedTTS language split (`en` or `zh`) |
| `ALLOWED_LOCAL_MEDIA_PATH` | server allowlist for SeedTTS reference paths; defaults to `/` for an isolated benchmark container |
| `MAX_SAMPLES` | optional SeedTTS subset; unset means the full language split |
| `LABEL` / `OUT_ROOT` | stable condition name and artifact root |
| `KV_EQUALITY` | `warn` (default), `require`, or `off` |
| `REQUIRE_IDLE_GPU` | refuse pre-existing compute clients (default `1`) |
| `CAPACITY_ONLY` | initialize replicas, record KV capacity, and exit before load |
| `KEEP_AUDIO` | retain generated WAV files (`0` by default; metrics are retained) |
| `GPU_TELEMETRY_INTERVAL_MS` | opt-in per-condition NVML sampling interval; `0` disables it (default) |
| `MPS_TMP_ROOT` | short local MPS runtime root; defaults to per-user `/tmp/sglang-omni-mps-$UID` |

For capacity calibration, use the hardware-specific YAML after verifying its
NUMA-local CPU IDs. These profiles start models only; they do not send SeedTTS or
other request traffic:

```bash
GPU_UUID=GPU-... python benchmarks/same_gpu_dp/run_study.py benchmarks/same_gpu_dp/configs/h100_higgs_capacity.yaml --mode calibrate
GPU_UUID=GPU-... python benchmarks/same_gpu_dp/run_study.py benchmarks/same_gpu_dp/configs/h200_higgs_capacity.yaml --mode calibrate
```

The loader is strict: unknown keys and unresolved `${ENVIRONMENT_VARIABLES}`
fail before launch. Scalar environment values not specified by YAML remain
available as advanced overrides.

Example dry-run (does not call NVIDIA tools or launch a model):

```bash
GPU_UUID=GPU-dry-run DP=3 USE_MPS=1 \
SERVER_CORE_SETS='0-7;8-15;16-23' \
CLIENT_CORE_SETS='48-51;52-55;56-59' \
MEM_FRACTIONS='0.27,0.27,0.27' \
bash benchmarks/same_gpu_dp/run_condition.sh --dry-run
```

Inspect `commands.sh` and `manifest.txt` before removing `--dry-run`. A real
direct run uses the same variables with the actual UUID.

### Production manifest traffic

Use `CLIENT_DRIVER=manifest` for the workload-driven cap search. Each JSONL row has
exactly three fields:

```json
{"id":"interactive-0001","arrival_offset_s":0.0,"payload":{"input":"Approved production text.","references":[{"audio_path":"/data/voices/reference.wav","text":"Reference transcript."}],"stream":false,"response_format":"wav","max_new_tokens":512}}
```

`id` must be unique and filesystem-safe. Arrival offsets must be non-negative and
non-decreasing. Use offset `0` for every row in a closed-loop concurrency test; use
recorded or explicitly spaced offsets for an open-loop replay. Non-streaming requests
must use WAV and streaming requests must use PCM so audio duration and correctness are
measured consistently.

The driver hashes the manifest, preserves the exact payload for every cap, validates
each request's `max_new_tokens` against the fixed server ceiling, saves generated
audio, and emits canonical `speed_results.json` for the existing aggregate summarizer.
One identical client process is launched per replica:

```bash
CLIENT_DRIVER=manifest TTS_MANIFEST=/absolute/path/to/tune.jsonl \
DP=2 USE_MPS=1 CONCURRENCY_PER_WORKER=32 ... \
bash benchmarks/same_gpu_dp/run_condition.sh
```

`--ref-format references` sends local reference-audio paths to the server.
`ALLOWED_LOCAL_MEDIA_PATH=/` is convenient only inside a trusted, isolated
benchmark container; it lets API clients request any local media path readable
by the server. Set it to the narrow common parent of the prepared SeedTTS files
when possible, and never expose this benchmark server to untrusted clients.

### KV capacity caveat

`mem_fraction_static` is resolved by SGLang against memory state at each replica
startup. Later replicas see memory already consumed by earlier complete Higgs
pipelines. Equal fractions therefore do **not** imply equal
`max_total_num_tokens`, and replica launch order can change batching headroom.

SGLang's `max_total_tokens` is an upper bound applied after memory profiling and
before the KV pool is allocated. This branch exposes it as
`sgl-omni serve --max-total-tokens` and as typed pipeline YAML under
`runtime.sglang_server_args.max_total_tokens`. `MAX_TOTAL_TOKENS` passes the cap
to every replica in one harness condition. A replica whose profiled capacity is
smaller still resolves smaller, and `KV_EQUALITY=require` rejects the condition.

The harness recognizes both `max_total_num_tokens=...` and SGLang's
`KV Cache is allocated. #tokens: ...` startup formats. If extraction returns
`null`, treat that as a failed gate rather than claiming a fair comparison.

### Calibrate an exact per-DP cap

Do not use the smallest pool from an uncapped sequential launch as the final cap.
Earlier replicas over-allocate in that experiment; capping them releases memory and
changes what later replicas can fit. The calibrator instead applies one candidate to
every replica, doubles a known-safe seed until it finds a failing bound, binary-searches
the boundary, and confirms the selected cap with fresh launches.

The capacity profiles omit the CLI override (`MEM_FRACTIONS=auto`), so the model's
existing memory-fraction default applies. For Higgs this is `0.85`. The common
candidate cap prevents earlier replicas from consuming that entire budget; the
search proves the largest candidate every replica can resolve. To run a custom
layout without YAML, set the `DPn_*` layout variables, including
`DPn_INITIAL_CAP_TOKENS`, and run:

```bash
CALIBRATION_DPS=2,3,4 \
CALIBRATION_MPS_MODES=1 \
CALIBRATION_CONFIRMATIONS=3 \
CALIBRATION_TOKEN_TOLERANCE=256 \
CALIBRATION_MARGIN_BPS=0 \
bash benchmarks/same_gpu_dp/calibrate_capacity.sh
```

Every attempted bound is recorded in `capacity_trials.tsv`. The final bracket and
safety-margin calculation are written to `capacity_selection.tsv`. `capacity.env`
exports:

```text
DPn_HIGHEST_PASS_TOKENS  highest candidate proven feasible during the search
DPn_MAX_TOTAL_TOKENS     floor(highest passing candidate × (1 - margin))
```

The first failing candidate is at most `CALIBRATION_TOKEN_TOLERANCE` above the highest
passing candidate. Set a nonzero basis-point margin only when repeated confirmations
show allocator drift that justifies it. Review the trials and selection, then source:

```bash
source benchmarks/results/same_gpu_dp/<printed-calibration-label>/capacity.env
```

The older `h200_higgs.yaml` is the SeedTTS matrix interface and is not used to choose
a production cap. It remains available only for reproducing the earlier study:

```bash
source benchmarks/results/same_gpu_dp/<printed-calibration-label>/capacity.env
python benchmarks/same_gpu_dp/run_study.py \
  benchmarks/same_gpu_dp/configs/h200_higgs.yaml --mode matrix
```

For example, an uncapped H200 DP3 launch resolving `216719, 140294, 84328` establishes
that `84328` is a safe search seed, not the maximum equal cap. Keep DP1 uncapped for
aggregate-GPU comparisons: the cap removes launch-order imbalance within one DP
layout, not across different DP degrees.

After calibration, the hardware-specific screening YAMLs run the full SeedTTS EN
split with fixed physical-core budgets and exact per-DP caps:

```bash
source benchmarks/results/same_gpu_dp/<printed-calibration-label>/capacity.env
GPU_UUID=GPU-... python benchmarks/same_gpu_dp/run_study.py benchmarks/same_gpu_dp/configs/h100_higgs_seedtts_screen.yaml --mode matrix
GPU_UUID=GPU-... python benchmarks/same_gpu_dp/run_study.py benchmarks/same_gpu_dp/configs/h200_higgs_seedtts_screen.yaml --mode matrix
```

The screening matrix is direct-to-worker by design. Validate only the selected recipe
with `MODE=router`. Router mode explicitly uses `least_request` by default and writes
`router_workers_after.json` plus `router_summary.json` from the router's actual
per-worker counters.

## Tune DP1 first

Find the DP1 Pareto frontier before comparing MPS. Keep the model, checkpoint,
dataset, generation parameters, total server/client cores, and NUMA placement
fixed. Sweep client concurrency and server batching together, for example:

```bash
for c in 8 16 32 48 64 96 128; do
  for b in 64 128; do
    GPU_UUID=GPU-... DP=1 USE_MPS=0 \
    SERVER_CORE_SETS='0-31' CLIENT_CORE_SETS='48-63' \
    MEM_FRACTIONS=auto CONCURRENCY_PER_WORKER=$c \
    MAX_RUNNING_REQUESTS=$b CUDA_GRAPH_MAX_BS=$b \
    LABEL="dp1_c${c}_b${b}" \
    bash benchmarks/same_gpu_dp/run_condition.sh
  done
done
```

Select the highest-throughput DP1 point that still meets the latency/RTF SLO.
Do not use an untuned default as the baseline. Then sweep the same per-worker
concurrency values for every DP point: the corrected PR #986 reproduction found
that concurrency 16 materially under-drove replicas whose generation cap was
64, reversing the apparent conclusion.

## Run the fair DP × MPS matrix

`run_matrix.sh` runs caller-selected DP 1–10 conditions and refuses layouts whose total
server or client CPU counts differ. Define dedicated subdivisions for each DP.
This example uses 32 server CPUs and 16 client CPUs for every point:

```bash
export GPU_UUID=GPU-...
export DP1_SERVER_CORE_SETS='0-31'
export DP1_CLIENT_CORE_SETS='48-63'
export DP1_MEM_FRACTIONS='auto'
export DP2_SERVER_CORE_SETS='0-15;16-31'
export DP2_CLIENT_CORE_SETS='48-55;56-63'
export DP2_MEM_FRACTIONS='auto,auto'
export DP3_SERVER_CORE_SETS='0-9;10-19;20-31'
export DP3_CLIENT_CORE_SETS='48-52;53-57;58-63'
export DP3_MEM_FRACTIONS='auto,auto,auto'
export DP4_SERVER_CORE_SETS='0-7;8-15;16-23;24-31'
export DP4_CLIENT_CORE_SETS='48-51;52-55;56-59;60-63'
export DP4_MEM_FRACTIONS='auto,auto,auto,auto'
# Source the calibration's capacity.env here. It exports
# DP2_MAX_TOTAL_TOKENS, DP3_MAX_TOTAL_TOKENS, and DP4_MAX_TOTAL_TOKENS.
export MATRIX_ORDER='3:1,1:0,4:0,2:1,3:0,1:1,4:1,2:0'
export CONCURRENCY_VALUES='32,48,64,96'
export REPETITIONS=5
export SHUFFLE_SEED=986
export KV_EQUALITY=require
bash benchmarks/same_gpu_dp/run_matrix.sh
```

Add `--dry-run` to validate and print the entire matrix without touching CUDA.
When `KV_EQUALITY=require`, `run_matrix.sh` refuses DP2–4 before any expensive
launch unless the corresponding `DPn_MAX_TOTAL_TOKENS` is set.

`auto` is safe here because it only omits the CLI fraction override and DP2–4 also
receive the exact searched caps. Do not run uncapped same-GPU replicas and assume
their profiled pools will be equal. Each matrix condition is run at every
`CONCURRENCY_VALUES` point; compare the peak
that satisfies the same SLO, rather than choosing one concurrency for DP1 and a
different unreported search for DPk. Use a different randomized `MATRIX_ORDER`
per study or set a recorded `SHUFFLE_SEED` (the default is `1`) to shuffle every
DP×MPS×concurrency cell independently in each repetition. Set it to `off` only
for a deliberate fixed-order diagnostic. The harness saves the realized order,
continues after failed/OOM points, records them in `matrix_results.tsv`, and computes
per-point Student-t 95% confidence intervals in `matrix_summary.json`.

Optional MPS resource experiments accept one value or a value per replica:

```bash
MPS_THREAD_PERCENTAGES='25,25,25,25' \
MPS_PINNED_MEM_LIMITS='0=16G;0=16G;0=16G;0=16G' \
DP=4 USE_MPS=1 ... bash benchmarks/same_gpu_dp/run_condition.sh
```

For a matrix, use `DP2_MPS_THREAD_PERCENTAGES`,
`DP3_MPS_THREAD_PERCENTAGES`, and so on; MPS-off conditions explicitly clear
these values.

Validate the exact pinned-memory syntax against the CUDA version in the H200
container; support has changed across CUDA releases. Treat active-thread
percentage as a provisioning experiment, not as guaranteed SM isolation.

Do not prepend a host `/lib/x86_64-linux-gnu` directory blindly to
`LD_LIBRARY_PATH`. In a CUDA 13 container this can select an older CUDA 12.8
driver library and make PyTorch report that the NVIDIA driver is too old. Keep
the container's working inherited library path unless its own CUDA probe proves
otherwise.

## Measure router overhead separately

Direct results establish the serving ceiling. Router mode starts the repository
router against the already owned replica URLs and drives the same `DP` canonical
clients, CPU sets, and per-client concurrency through the router:

```bash
MODE=router ROUTER_CORES='64-67' DP=3 USE_MPS=1 \
SERVER_CORE_SETS='0-9;10-19;20-31' \
CLIENT_CORE_SETS='48-52;53-57;58-63' \
MEM_FRACTIONS='0.27,0.27,0.27' \
GPU_UUID=GPU-... LABEL=dp3_mps_router \
bash benchmarks/same_gpu_dp/run_condition.sh
```

Compare this only with the otherwise identical direct condition. Router mode
has one aggregate result and cannot infer per-worker balance from client output;
use router diagnostics/logging for that check.

## Metrics, quality, and acceptance

`summary.json` contains aggregate request throughput, exact p50/p95/p99 computed
from canonical per-request rows, RTF, audio seconds per wall second, output-token
throughput, output-token/audio-duration totals and means, failures, and
per-worker QPS balance. Review output volume alongside QPS: shorter or truncated
output is not a throughput win.

Generation-only avoids ASR contention with TTS. After servers are stopped, run
the canonical `--transcribe-only`, `--similarity-only`, or `--utmos-only` phases
against each worker output directory. Compare WER, speaker similarity, UTMOS,
generated duration, and token totals with DP1 using the same saved sample IDs.

An H200 result is strong enough to support the recipe only if:

- DP+MPS beats tuned DP1 throughput by more than 10% with non-overlapping 95%
  confidence intervals across at least five randomized repetitions;
- p95/p99 latency and RTF stay inside the chosen SLO;
- failures do not increase, per-worker traffic is balanced, and resolved KV
  capacity is equal or its difference is explicitly controlled;
- output duration/token volume and quality metrics stay within predeclared
  tolerances;
- the winning point survives a 30–60 minute soak and repeated clean start/stop;
- router throughput remains close to direct aggregate throughput, or the claim
  explicitly excludes the router.

For device-level evidence, run Nsight Systems in a separate terminal over an
aligned steady window without replacing the canonical client:

```bash
nsys profile --gpu-metrics-devices <physical-index> \
  --gpu-metrics-set gh100 --gpu-metrics-frequency 10000 \
  -d 60 -o "$OUT/steady_gpu" -f true sleep 63
```

Record the physical index-to-UUID mapping in the artifact. Use SM Active, DRAM,
and Tensor activity to explain a result; they do not replace latency, output,
failure, or quality checks.
