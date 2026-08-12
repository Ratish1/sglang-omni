# Qwen3-TTS hidden-sync H100 runbook

This runbook profiles only SGLang-Omni-owned Qwen3-TTS code. It calibrates the
installed Torch synchronization detector, captures bounded request and Kineto
traces, and produces machine-readable attribution. It intentionally does not
change transfer behavior; the first H100 artifacts select the repair.

Use the same container and Python environment for the probe, server, benchmark,
and analyzer. Commands below assume Bash and repository-root working directory.

## 1. Check out the instrumentation branch

For an existing clone:

```bash
git fetch origin perf/qwen3-tts-hidden-h2d-sync
git switch --track origin/perf/qwen3-tts-hidden-h2d-sync
```

For a fresh clone:

```bash
git clone --branch perf/qwen3-tts-hidden-h2d-sync \
  https://github.com/Ratish1/sglang-omni.git
cd sglang-omni
```

Confirm that the worktree is clean and record the exact commit:

```bash
git status --short
git rev-parse HEAD
```

Install SGLang-Omni through the normal project instructions. Qwen3-TTS also
requires the supported dependency combination documented in
`docs/cookbook/qwen3_tts.md`:

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

Do not let `qwen-tts` replace the repository's Torch, Transformers, or SGLang
versions.

## 2. Establish provenance

```bash
export EXP_ROOT=/tmp/q3tts-hidden-sync
mkdir -p "$EXP_ROOT/provenance" "$EXP_ROOT/probes"

git rev-parse HEAD > "$EXP_ROOT/provenance/omni_commit.txt"
git status --short > "$EXP_ROOT/provenance/omni_status.txt"
python -VV > "$EXP_ROOT/provenance/python.txt" 2>&1
python -m pip freeze > "$EXP_ROOT/provenance/pip_freeze.txt"
sha256sum "$EXP_ROOT/provenance/pip_freeze.txt" \
  > "$EXP_ROOT/provenance/pip_freeze.sha256"
nvidia-smi -q > "$EXP_ROOT/provenance/nvidia_smi_q.txt"
nvidia-smi topo -m > "$EXP_ROOT/provenance/nvidia_smi_topo.txt"

python - <<'PY' > "$EXP_ROOT/provenance/torch_cuda.txt"
import json
import torch

print(json.dumps({
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "device_name": torch.cuda.get_device_name(0),
    "device_capability": torch.cuda.get_device_capability(0),
}, sort_keys=True, indent=2))
PY
```

Also record the container image digest from the host/container orchestrator if
available; it usually cannot be recovered reliably from inside the container.

The immutable experiment inputs are:

```bash
export MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export MODEL_REVISION=5d83992436eae1d760afd27aff78a71d676296fc
export DATASET_ID=zhaochenyang20/seed-tts-eval-arrow
export DATASET_REVISION=27f4c1adee83b5b29b7c4b375f6b976324bda308
export MODEL_PATH
MODEL_PATH=$(hf download "$MODEL_ID" --revision "$MODEL_REVISION")
printf '%s\n' "$MODEL_PATH" > "$EXP_ROOT/provenance/model_path.txt"
```

The exported dataset revision is provenance metadata. When this exact
`DATASET_ID` is passed, the benchmark independently applies the same hard-coded
revision from `benchmarks/dataset/prepare.py`.

## 3. Calibrate `torch.cuda.set_sync_debug_mode`

Run each operation in a fresh process:

```bash
for mode in warn error; do
  : > "$EXP_ROOT/probes/${mode}.jsonl"
  for case in \
    item dtoh_pageable h2d_pageable \
    h2d_pinned_nonblocking dtoh_pinned_nonblocking \
    stream_synchronize device_synchronize; do
    python benchmarks/profiling/cuda_sync_debug_probe.py \
      "$case" "$mode" 2>&1 | tee -a "$EXP_ROOT/probes/${mode}.jsonl"
  done
done
```

Before interpreting server warnings, inspect the JSONL:

- known blocking scalar/D2H operations and explicit stream/device waits should
  warn in `warn` or raise in `error` wherever this Torch build has coverage;
- pinned nonblocking H2D and D2H controls should return without a detector hit;
- record pageable H2D exactly as observed rather than assuming coverage;
- if positive controls are silent, a silent server log is not absence evidence.

## 4. Freeze target and warmup manifests

Create the target list without contacting a server. `--unique-reference-audio`
deduplicates by file-content SHA256, not by temporary path or row ID. The output
records order plus reference/text hashes and a checksum over the whole manifest:

```bash
export TARGET_MANIFEST_DIR="$EXP_ROOT/manifests/target-c16-64"
python -m benchmarks.eval.benchmark_tts_seedtts \
  --prepare-manifest-only \
  --model "$MODEL_ID" --meta "$DATASET_ID" --ref-format references \
  --max-samples 64 --unique-reference-audio \
  --output-dir "$TARGET_MANIFEST_DIR"
export TARGET_MANIFEST="$TARGET_MANIFEST_DIR/input_manifest.json"
```

Create a 64-row warmup list whose reference hashes are provably absent from the
target list:

```bash
export WARM_MANIFEST_DIR="$EXP_ROOT/manifests/warm-c16-64"
python -m benchmarks.eval.benchmark_tts_seedtts \
  --prepare-manifest-only \
  --model "$MODEL_ID" --meta "$DATASET_ID" --ref-format references \
  --sample-offset 1000 --max-samples 64 \
  --exclude-reference-manifest "$TARGET_MANIFEST" \
  --output-dir "$WARM_MANIFEST_DIR"
export WARM_MANIFEST="$WARM_MANIFEST_DIR/input_manifest.json"
```

The benchmark revalidates all hashes whenever `--sample-manifest` is used; a
dataset or manifest mismatch fails before traffic.

## 5. Start and fully warm a fresh server (terminal A)

Use a fresh server for each cache condition and later for every baseline/candidate
comparison. The detector is not armed during model loading or graph capture.

```bash
export EXP_ROOT=/tmp/q3tts-hidden-sync
export MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base
export MODEL_REVISION=5d83992436eae1d760afd27aff78a71d676296fc
export MODEL_PATH
MODEL_PATH=$(hf download "$MODEL_ID" --revision "$MODEL_REVISION")
export SERVER_LOG="$EXP_ROOT/server-$(date +%s).log"

SGLANG_TORCH_PROFILER_DIR="$EXP_ROOT" \
sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000 2>&1 | tee "$SERVER_LOG"
```

Wait for health and full readiness. From terminal B:

```bash
until curl -fsS http://127.0.0.1:8000/health; do sleep 2; done
```

Warm batch-size 1, then concurrency 16, with only the disjoint warmup list. This
keeps allocator/compile/lazy predictor-graph work outside the capture without
priming any target reference:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model "$MODEL_ID" --meta "$DATASET_ID" --ref-format references \
  --sample-offset 1000 --max-samples 1 \
  --exclude-reference-manifest "$TARGET_MANIFEST" \
  --max-new-tokens 128 --seed 20260812 \
  --warmup 0 --concurrency 1 \
  --output-dir "$EXP_ROOT/warmup-c1"

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model "$MODEL_ID" --meta "$DATASET_ID" --ref-format references \
  --sample-manifest "$WARM_MANIFEST" \
  --max-new-tokens 128 --seed 20260812 \
  --warmup 0 --concurrency 16 \
  --output-dir "$EXP_ROOT/warmup-c16"
```

## 6. Short first pass: warning discovery

This is the cheapest useful run: no Torch Profiler, the first 16 entries of a
separate c1 target manifest, and concurrency 1. Prepare that manifest exactly as
in section 4 with `--max-samples 16 --unique-reference-audio`. It is a subset of
the 64-entry target list, so the already completed disjoint warmup remains valid.

```bash
export TARGET_MANIFEST_DIR="$EXP_ROOT/manifests/target-c1-16"
python -m benchmarks.eval.benchmark_tts_seedtts \
  --prepare-manifest-only \
  --model "$MODEL_ID" --meta "$DATASET_ID" --ref-format references \
  --max-samples 16 --unique-reference-audio \
  --output-dir "$TARGET_MANIFEST_DIR"
export TARGET_MANIFEST="$TARGET_MANIFEST_DIR/input_manifest.json"
```

```bash
export BASE=http://127.0.0.1:8000
export RUN="q3tts-warn-c1-nonstream-miss-$(date +%s)"
export RUN_DIR="$EXP_ROOT/$RUN"
mkdir -p "$RUN_DIR/events"

curl -fsS -X POST "$BASE/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\",\"event_dir\":\"$RUN_DIR/events\",\"enable_torch\":false,\"config\":{\"cuda_sync_debug_mode\":\"warn\",\"target_stage\":\"tts_engine\"}}" \
  | tee "$RUN_DIR/start_response.json"
```

Before sending traffic, require all of these from `start_response.json`:

- `.acknowledgement.success == true`;
- target session thread name `scheduler-tts_engine`;
- `cuda_sync_debug_mode == "warn"` and `cuda_sync_debug_applied == true`;
- `snapshot_before.model_runner.predictor_graph.captured_keys` already contains
  the batch keys exercised by the active workload;
- reference/speaker cache counters are recorded before traffic.

For example:

```bash
jq '.acknowledgement' "$RUN_DIR/start_response.json"
```

Then run exactly the bounded target workload—no in-window warmup:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url "$BASE" \
  --model "$MODEL_ID" --meta "$DATASET_ID" --ref-format references \
  --sample-manifest "$TARGET_MANIFEST" \
  --max-new-tokens 128 --seed 20260812 \
  --warmup 0 --concurrency 1 \
  --output-dir "$RUN_DIR/client"

curl -fsS -X POST "$BASE/stop_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\"}" \
  | tee "$RUN_DIR/stop_response.json"

python -m sglang_omni.profiler "$RUN_DIR/events" \
  --format json --out "$RUN_DIR/request_report.json"
```

Require `acknowledgement.success=true` and `artifacts_finalized=true` in the stop
response. Compare `snapshot_before` from start with `snapshot_before_stop` from
stop: all 16 target references must be misses for this fresh-server condition,
and no lazy graph key may first appear inside the warning window. Preserve the
complete server log, including warning stack locations.

## 7. Low-overhead CPU+CUDA trace

Point `TARGET_MANIFEST` back to the 64-entry manifest from section 4. Restart the
server, repeat only the disjoint c1/c16 warmup, and verify graph keys before
capturing. The trace pass deliberately uses sync-debug `default`: warning logging
would perturb the host timeline we are trying to measure.

```bash
export TARGET_MANIFEST="$EXP_ROOT/manifests/target-c16-64/input_manifest.json"
export WARM_MANIFEST="$EXP_ROOT/manifests/warm-c16-64/input_manifest.json"
```

```bash
export RUN="q3tts-trace-c16-nonstream-miss-$(date +%s)"
export RUN_DIR="$EXP_ROOT/$RUN"
mkdir -p "$RUN_DIR/events"

curl -fsS -X POST "$BASE/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\",\"trace_path_template\":\"$RUN_DIR/trace\",\"event_dir\":\"$RUN_DIR/events\",\"enable_torch\":true,\"config\":{\"cuda_sync_debug_mode\":\"default\",\"target_stage\":\"tts_engine\"}}" \
  | tee "$RUN_DIR/start_response.json"
```

Require a successful acknowledgement on `scheduler-tts_engine`,
`torch_active=true`, and the complete pre-warmed graph-key snapshot, then:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url "$BASE" \
  --model "$MODEL_ID" --meta "$DATASET_ID" --ref-format references \
  --sample-manifest "$TARGET_MANIFEST" \
  --max-new-tokens 128 --seed 20260812 \
  --warmup 0 --concurrency 16 \
  --output-dir "$RUN_DIR/client"

curl -fsS -X POST "$BASE/stop_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\"}" \
  | tee "$RUN_DIR/stop_response.json"
```

Targeted stop is synchronous with export and gzip completion. Require
`artifacts_finalized=true`, a non-null trace path in every rank session, and no
errors. Then verify the returned paths exist and no raw JSON remains:

```bash
jq -e '.artifacts_finalized == true and .acknowledgement.success == true' \
  "$RUN_DIR/stop_response.json"
test -z "$(find "$RUN_DIR" -name '*.trace.json' -print -quit)"
test -n "$(find "$RUN_DIR" -name '*.trace.json.gz' -print -quit)"
```

Analyze all per-process traces without comparing their clocks:

```bash
mapfile -t TRACES < <(find "$RUN_DIR" -name '*.trace.json.gz' -type f | sort)
test "${#TRACES[@]}" -gt 0

python benchmarks/profiling/analyze_cuda_sync_trace.py \
  "${TRACES[@]}" --output-dir "$RUN_DIR/analysis"

python -m sglang_omni.profiler "$RUN_DIR/events" \
  --format json --out "$RUN_DIR/request_report.json"

find "$RUN_DIR" -type f ! -name SHA256SUMS -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_DIR/SHA256SUMS"
```

The first inspection is mechanical, not visual:

- count `blocking_copy != null` occurrences and split them by HtoD/DtoH;
- use `compound_host_block_*`, not only `sync_wait_*`, for the article's
  `cudaMemcpyAsync -> cudaStreamSynchronize` mechanism;
- inspect correlated transfer bytes/duration and
  `gpu_idle.global_idle_in_interval_us` together;
- remember that per-occurrence intervals can overlap, so aggregate idle/bubble
  sums are prioritization signals, not unique wall-clock time;
- require semantic `qwen3_tts.*` ranges and scheduler-thread ATen operators in
  this targeted trace. If they are absent, the start acknowledgement's thread
  ownership is wrong and the trace is not accepted.

The start/stop snapshots are equally mandatory: target reference hashes are
unique and disjoint from warmup; target-cache misses must increase as expected;
and captured predictor-graph keys must not change inside the active window.

## 8. Targeted stack pass

Only after the low-overhead trace identifies a range, restart the server with:

```bash
SGLANG_TORCH_PROFILER_WITH_STACK=1 \
SGLANG_TORCH_PROFILER_RECORD_SHAPES=1 \
SGLANG_TORCH_PROFILER_DIR="$EXP_ROOT" \
sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000 2>&1 | tee "$SERVER_LOG"
```

Repeat the smallest workload that reproduces the target. Do not combine memory
profiling with this pass. `config.cuda_sync_debug_mode="error"` may be used for
one localization request after warning discovery, but it may abort that request
and is never performance evidence.

## 9. Complete baseline matrix

After the short pass works, collect these conditions. Restart and infrastructure-
warm the server between miss and hit conditions:

| Condition | Active sample list | Concurrency | Extra flag | Cache preparation |
| --- | ---: | ---: | --- | --- |
| Non-stream miss | verified unique-ref manifest, 16 | 1 | none | disjoint c1+c16 graph warmup |
| Non-stream miss | verified unique-ref manifest, 64 | 16 | none | disjoint c1+c16 graph warmup |
| Stream miss | same verified manifest, 16 | 1 | `--stream` | disjoint c1+c16 graph warmup |
| Stream miss | same verified manifest, 64 | 16 | `--stream` | disjoint c1+c16 graph warmup |
| Exact replay/hit | exact miss manifest | matching | matching | replay manifest once before arming, then replay it again |

For a hit condition, priming uses `--sample-manifest "$TARGET_MANIFEST"` with the
same output mode as the active capture but writes to a separate `prime`
directory. Only arm profiling after priming finishes. Confirm the start snapshot
shows populated cache entries/hits. Never compare a miss baseline with a hit
candidate.

## 10. What to send back

Please return, or make available, the following files for the first warning and
first low-overhead trace runs:

- `provenance/` and `probes/`;
- complete server logs;
- `start_response.json` and `stop_response.json`;
- client `speed_results.json`, generated-audio metadata, and failure records;
- every `input_manifest.json`, including disjoint warmup and target manifests;
- request-event JSONL plus `request_report.json`;
- every `*.trace.json.gz`;
- the analyzer occurrence JSON and aggregate JSON/CSV;
- `SHA256SUMS`.

Interpretation rules:

- a warning is a locator, not proof of wasted time;
- prioritize synchronization with matching semantic range/stack, transfer or wait,
  subsequent launch gap, request interval, and repeatable workload dependence;
- `nearest_time_heuristic` analyzer matches require manual trace verification;
- do not classify required async token publication or a real CPU consumer commit
  as unsafe merely because it waits;
- select a repair only when the target's count/time and post-sync bubble are
  material on the request critical path.
