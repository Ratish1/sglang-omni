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

## 4. Start a fresh server (terminal A)

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

Wait for health, full readiness, and CUDA graph capture to finish. From terminal
B:

```bash
until curl -fsS http://127.0.0.1:8000/health; do sleep 2; done
```

Warm infrastructure using one sample outside the measured range. Do not warm the
first 64 samples before a unique/miss capture:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url http://127.0.0.1:8000 \
  --model "$MODEL_ID" \
  --meta "$DATASET_ID" --ref-format references \
  --sample-offset 1000 --max-samples 1 \
  --max-new-tokens 128 --seed 20260812 \
  --warmup 0 --concurrency 1 \
  --output-dir "$EXP_ROOT/warmup"
```

## 5. Short first pass: warning discovery

This is the cheapest useful run: no Torch Profiler, 16 unique non-streaming
requests at concurrency 1.

```bash
export BASE=http://127.0.0.1:8000
export RUN="q3tts-warn-c1-nonstream-miss-$(date +%s)"
export RUN_DIR="$EXP_ROOT/$RUN"
mkdir -p "$RUN_DIR/events"

curl -fsS -X POST "$BASE/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\",\"event_dir\":\"$RUN_DIR/events\",\"enable_torch\":false,\"config\":{\"cuda_sync_debug_mode\":\"warn\"}}" \
  | tee "$RUN_DIR/start_response.json"
```

The HTTP response confirms only that messages were broadcast. Before sending
traffic, inspect terminal A or `SERVER_LOG` and require:

- one `Process profiler session started` line for this run in the CUDA-owning PID;
- `cuda_sync_debug_mode=warn applied=True` in that line;
- all expected colocated stages joining the same PID/run; no second Torch session.

Then run exactly the bounded target workload—no in-window warmup:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url "$BASE" \
  --model "$MODEL_ID" \
  --meta "$DATASET_ID" --ref-format references \
  --sample-offset 0 --max-samples 16 \
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

Require the matching process-session stop log and confirm that sync-debug was
reset. Preserve the complete server log, including warning stack locations.

## 6. Low-overhead CPU+CUDA trace

Restart the server to clear the 2 GiB CPU reference cache, perform only the
offset-1000 infrastructure warmup, then capture 64 unique non-streaming requests
at concurrency 16:

```bash
export RUN="q3tts-trace-c16-nonstream-miss-$(date +%s)"
export RUN_DIR="$EXP_ROOT/$RUN"
mkdir -p "$RUN_DIR/events"

curl -fsS -X POST "$BASE/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\",\"trace_path_template\":\"$RUN_DIR/trace\",\"event_dir\":\"$RUN_DIR/events\",\"enable_torch\":true,\"config\":{\"cuda_sync_debug_mode\":\"warn\"}}" \
  | tee "$RUN_DIR/start_response.json"
```

Wait for the matching `Process profiler session started` log with `applied=True`
before traffic, then:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server \
  --base-url "$BASE" \
  --model "$MODEL_ID" \
  --meta "$DATASET_ID" --ref-format references \
  --sample-offset 0 --max-samples 64 \
  --max-new-tokens 128 --seed 20260812 \
  --warmup 0 --concurrency 16 \
  --output-dir "$RUN_DIR/client"

curl -fsS -X POST "$BASE/stop_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$RUN\"}" \
  | tee "$RUN_DIR/stop_response.json"
```

Stop is also asynchronous. Wait for `Trace exported` and for every gzip to have
the same size and mtime on two polls:

```bash
until find "$RUN_DIR" -name '*.trace.json.gz' -print -quit | grep -q .; do sleep 2; done
while find "$RUN_DIR" -name '*.trace.json' -print -quit | grep -q .; do sleep 2; done

find "$RUN_DIR" -name '*.trace.json.gz' -type f -printf '%p %s %T@\n' \
  | sort | tee "$RUN_DIR/trace_state_1.txt"
sleep 5
find "$RUN_DIR" -name '*.trace.json.gz' -type f -printf '%p %s %T@\n' \
  | sort | tee "$RUN_DIR/trace_state_2.txt"
diff -u "$RUN_DIR/trace_state_1.txt" "$RUN_DIR/trace_state_2.txt"
```

An empty trace list is a failure, not a successful stable state. Analyze all
per-process traces without comparing their clocks:

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

## 7. Targeted stack pass

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

## 8. Complete baseline matrix

After the short pass works, collect these conditions. Restart and infrastructure-
warm the server between miss and hit conditions:

| Condition | Active sample list | Concurrency | Extra flag | Cache preparation |
| --- | ---: | ---: | --- | --- |
| Non-stream miss | first 16 | 1 | none | offset-1000 warm only |
| Non-stream miss | first 64 | 16 | none | offset-1000 warm only |
| Stream miss | first 16 | 1 | `--stream` | offset-1000 warm only |
| Stream miss | first 64 | 16 | `--stream` | offset-1000 warm only |
| Exact replay/hit | fixed first 16 or 64 | matching | matching | run that exact list once before arming, then replay it |

For a hit condition, priming uses the exact same benchmark command and output
mode as the active capture but writes to a separate `prime` directory. Only arm
profiling after priming finishes. Never compare a miss baseline with a hit
candidate.

## 9. What to send back

Please return, or make available, the following files for the first warning and
first low-overhead trace runs:

- `provenance/` and `probes/`;
- complete server logs;
- `start_response.json` and `stop_response.json`;
- client `speed_results.json`, generated-audio metadata, and failure records;
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
