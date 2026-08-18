# Qwen3-TTS hidden H2D synchronization: minimal H100 qualification

This cookbook qualifies the scalar-mask correction `86884103` and all-thread
profiler correction `167a2573` without repeating the full SeedTTS corpus
unnecessarily. It runs one server, one bounded warning pass, and one bounded
clean trace. Perfetto is optional; retain the trace, but use the mechanical
analyzer first.

The commands intentionally omit `--seed`, `--temperature`, `--top-p`,
`--top-k`, and `--repetition-penalty`. Qwen3-TTS resolves its normal generation
defaults. `--max-samples`, `--sample-offset`, and `--warmup 0` are present only
to make the diagnostic windows bounded, disjoint, and free of an unreported
benchmark warm-up request.

Sections 1-5 passed at `aebd777e`. Do not repeat their warning gate or the full
corpus merely to qualify profiler annotations. Section 7 is the single bounded
follow-up for attributing the remaining synchronization owners.

## 1. Checkout and provenance

```bash
git fetch origin
git switch --detach origin/perf/qwen3-tts-hidden-h2d-sync-v2
git merge-base --is-ancestor 86884103 HEAD
git merge-base --is-ancestor 167a2573 HEAD
git rev-parse HEAD

python - <<'PY'
import sglang_omni
import torch

print("sglang_omni", sglang_omni.__file__)
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
print("capability", torch.cuda.get_device_capability(0))
PY
```

Set `MODEL_PATH` to the existing pinned snapshot, not a floating repository
checkout. The previously qualified snapshot is
`5d83992436eae1d760afd27aff78a71d676296fc`.

```bash
export MODEL_PATH=/absolute/path/to/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc
export QUAL_DIR=/tmp/q3tts-hidden-sync-167a2573
export SERVER_LOG="$QUAL_DIR/server.log"
mkdir -p "$QUAL_DIR"
```

Confirm that the editable install resolves to this worktree before continuing.

## 2. Start once and warm outside both windows

```bash
CUDA_VISIBLE_DEVICES=0 \
SGLANG_TORCH_PROFILER_DIR="$QUAL_DIR/profiles" \
sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000 \
  >"$SERVER_LOG" 2>&1 &
export SERVER_PID=$!

until curl -fsS http://127.0.0.1:8000/health >/dev/null; do sleep 2; done
```

Warm compilation, allocators, and c16 graph keys with the final 64 English
samples. Those references are disjoint from both measured diagnostic windows.

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --model "$MODEL_PATH" \
  --output-dir "$QUAL_DIR/warmup" \
  --max-samples 64 \
  --sample-offset 1024 \
  --concurrency 16 \
  --warmup 0
```

The dataset loader pins the canonical SeedTTS repository revision in source.
Do not start either diagnostic window if warm-up reports an HTTP/CUDA error.

## 3. One warning-only pass

```bash
export WARN_RUN=q3tts-167a2573-warn-c1
mkdir -p "$QUAL_DIR/$WARN_RUN/events"
curl -fsS -X POST http://127.0.0.1:8000/start_profile \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$WARN_RUN\",\"event_dir\":\"$QUAL_DIR/$WARN_RUN/events\",\"enable_torch\":false,\"config\":{\"cuda_sync_debug_mode\":\"warn\"}}" \
  | tee "$QUAL_DIR/$WARN_RUN/start_response.json"

until grep -Fq "CUDA sync debug enabled run_id=$WARN_RUN" "$SERVER_LOG"; do
  sleep 1
done

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --model "$MODEL_PATH" \
  --output-dir "$QUAL_DIR/$WARN_RUN/client" \
  --max-samples 16 \
  --sample-offset 0 \
  --concurrency 1 \
  --warmup 0

curl -fsS -X POST http://127.0.0.1:8000/stop_profile \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$WARN_RUN\"}" \
  | tee "$QUAL_DIR/$WARN_RUN/stop_response.json"

until grep -Fq "CUDA sync debug disabled run_id=$WARN_RUN" "$SERVER_LOG"; do
  sleep 1
done

awk -v run="$WARN_RUN" '
  index($0, "CUDA sync debug enabled run_id=" run) { active=1 }
  active { print }
  index($0, "CUDA sync debug disabled run_id=" run) { exit }
' "$SERVER_LOG" > "$QUAL_DIR/$WARN_RUN/warning_window.log"
```

Pass conditions:

- 16/16 requests complete and the server reports no CUDA error.
- Every expected CUDA-owning PID/rank has one detector-enable record. Colocated
  stages share one process-global detector and therefore one enable record.
- No warning originates from the new pinned/nonblocking sites in
  `qwen3_tts/sglang_model.py`, `qwen3_tts/model_runner.py`, or
  `Qwen3TTSTalker.prepare_decode_buffers`.
- In particular, there is no warning from the repetition/suppression mask
  `index_fill_` rebuild or the steady per-row `scatter_` update. These replace
  the three advanced assignments reported at tested HEAD `32d8bf30`.
- Warnings at cache-key/cache D2H, final-code D2H, or an explicit publication
  wait are retained ownership boundaries, not a reason to delete their
  synchronization. Prepared-reference normalization should be a device no-op
  after the prompt reference-code copy.

Use detector mode `error` only if a selected source still warns and its exact
call stack is unclear. A global error-mode pass will stop at the first retained
D2H boundary and is not a useful clean-server gate.

## 4. One clean c16 trace

```bash
export TRACE_RUN=q3tts-167a2573-trace-c16
mkdir -p "$QUAL_DIR/$TRACE_RUN/events" "$QUAL_DIR/$TRACE_RUN/analysis"
curl -fsS -X POST http://127.0.0.1:8000/start_profile \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$TRACE_RUN\",\"trace_path_template\":\"$QUAL_DIR/$TRACE_RUN/trace\",\"event_dir\":\"$QUAL_DIR/$TRACE_RUN/events\",\"enable_torch\":true}" \
  | tee "$QUAL_DIR/$TRACE_RUN/start_response.json"

until grep -Fq "Starting End-to-End Torch profiler (run_id=$TRACE_RUN)" "$SERVER_LOG"; do
  sleep 1
done

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --model "$MODEL_PATH" \
  --output-dir "$QUAL_DIR/$TRACE_RUN/client" \
  --max-samples 64 \
  --sample-offset 128 \
  --concurrency 16 \
  --warmup 0

curl -fsS -X POST http://127.0.0.1:8000/stop_profile \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$TRACE_RUN\"}" \
  | tee "$QUAL_DIR/$TRACE_RUN/stop_response.json"

until find "$QUAL_DIR/$TRACE_RUN" -name '*.trace.json.gz' -size +0c | grep -q .; do
  sleep 2
done
export TRACE_FILE
TRACE_FILE=$(find "$QUAL_DIR/$TRACE_RUN" -name '*.trace.json.gz' -size +0c -print -quit)
size_1=$(stat -c %s "$TRACE_FILE")
mtime_1=$(stat -c %Y "$TRACE_FILE")
sleep 5
size_2=$(stat -c %s "$TRACE_FILE")
mtime_2=$(stat -c %Y "$TRACE_FILE")
test "$size_1:$mtime_1" = "$size_2:$mtime_2"
```

Reuse the already reviewed streaming analyzer from the earlier profiling branch
without adding its 1,000-line diagnostic implementation to this production
candidate:

```bash
git fetch origin perf/qwen3-tts-hidden-h2d-sync
export TRACE_ANALYZER="$QUAL_DIR/analyze_cuda_sync_trace.py"
git show origin/perf/qwen3-tts-hidden-h2d-sync:benchmarks/profiling/analyze_cuda_sync_trace.py \
  > "$TRACE_ANALYZER"
python "$TRACE_ANALYZER" \
  "$TRACE_FILE" \
  --output-dir "$QUAL_DIR/$TRACE_RUN/analysis"
```

Pass conditions:

- 64/64 requests complete with no CUDA error and one stable gzip trace.
- The trace contains the `qwen3_tts.*` semantic ranges and scheduler-thread
  ATen/runtime events.
- The six candidate ranges contain no compound
  `cudaMemcpyAsync -> cudaStreamSynchronize` blocking-copy occurrence and no
  replacement stream/event/device wait.
- Async H2D copies may remain in the ranges; their absence would mean the trace
  did not exercise the intended cache-miss path.
- Retained D2H/handoff ranges are reported separately. Trace-wide sync count is
  not an acceptance metric by itself.

No Perfetto screenshots are required. Preserve the gzip trace so any ambiguous
occurrence can be inspected later without rerunning the workload.

## 5. Artifacts to return

```bash
kill "$SERVER_PID"
wait "$SERVER_PID" || true
find "$QUAL_DIR" -type f -print0 | sort -z | xargs -0 sha256sum \
  > "$QUAL_DIR/SHA256SUMS"
tar -C /tmp -czf /tmp/q3tts-hidden-sync-167a2573.tar.gz \
  q3tts-hidden-sync-167a2573
```

Return the tarball. The required contents are the full server log, both
start/stop responses, the extracted warning window, client result JSON/CSV,
event JSONL, stable trace gzip, analyzer occurrence/aggregate JSON and CSV, and
`SHA256SUMS`.

## 6. Full-corpus A/B only after the mechanical gates pass

Use one baseline run and one candidate run. For each revision, start from a
fresh checkout/GPU and run the canonical managed benchmark with no sampling
overrides and no `--max-samples`:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_tts_seedtts \
  --model "$MODEL_PATH" \
  --server-config examples/configs/qwen3_tts_0_6b.yaml \
  --output-dir /tmp/q3tts-full-REVISION
```

Baseline is `2cac60e8`; candidate must contain `86884103` and `167a2573`. This
is the full 1,088-sample English set at the benchmark's canonical c16, unseeded,
2,048-token defaults.
One A/B pair is descriptive: it can rule out a large regression, but it cannot
support a performance claim when the difference is within run-to-run variance.
Report the count of 2,048-token length caps and both raw and outlier-excluded WER
because the established unseeded baseline itself has a rare runaway tail.

## 7. Remaining-sync attribution trace

This is an observability-only follow-up. The added ranges do not change tensor
construction, transfer mechanics, streams, sampling, or stage payloads. Start a
fresh server at the range-instrumentation revision, warm it once, and collect
one c16 trace. Do not run sync-debug or another full-corpus A/B.

Use the same pinned model snapshot and server command from Sections 1-2 with a
new output directory:

```bash
export ATTR_REV
ATTR_REV=$(git rev-parse --short=8 HEAD)
export QUAL_DIR=/tmp/q3tts-sync-attribution-$ATTR_REV
export SERVER_LOG="$QUAL_DIR/server.log"
mkdir -p "$QUAL_DIR"

CUDA_VISIBLE_DEVICES=0 \
SGLANG_TORCH_PROFILER_DIR="$QUAL_DIR/profiles" \
sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000 \
  >"$SERVER_LOG" 2>&1 &
export SERVER_PID=$!
until curl -fsS http://127.0.0.1:8000/health >/dev/null; do sleep 2; done

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --model "$MODEL_PATH" \
  --output-dir "$QUAL_DIR/warmup" \
  --max-samples 64 \
  --sample-offset 1024 \
  --concurrency 16 \
  --warmup 0
```

Capture the same disjoint 64-request population used by the accepted mechanical
trace:

```bash
export TRACE_RUN=q3tts-$ATTR_REV-remaining-sync-c16
mkdir -p "$QUAL_DIR/$TRACE_RUN/events" "$QUAL_DIR/$TRACE_RUN/analysis"
curl -fsS -X POST http://127.0.0.1:8000/start_profile \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$TRACE_RUN\",\"trace_path_template\":\"$QUAL_DIR/$TRACE_RUN/trace\",\"event_dir\":\"$QUAL_DIR/$TRACE_RUN/events\",\"enable_torch\":true}" \
  | tee "$QUAL_DIR/$TRACE_RUN/start_response.json"

until grep -Fq "Starting End-to-End Torch profiler (run_id=$TRACE_RUN)" "$SERVER_LOG"; do
  sleep 1
done

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --model "$MODEL_PATH" \
  --output-dir "$QUAL_DIR/$TRACE_RUN/client" \
  --max-samples 64 \
  --sample-offset 128 \
  --concurrency 16 \
  --warmup 0

curl -fsS -X POST http://127.0.0.1:8000/stop_profile \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$TRACE_RUN\"}" \
  | tee "$QUAL_DIR/$TRACE_RUN/stop_response.json"

until find "$QUAL_DIR/$TRACE_RUN" -name '*.trace.json.gz' -size +0c | grep -q .; do
  sleep 2
done
export TRACE_FILE
TRACE_FILE=$(find "$QUAL_DIR/$TRACE_RUN" -name '*.trace.json.gz' -size +0c -print -quit)
size_1=$(stat -c %s "$TRACE_FILE")
mtime_1=$(stat -c %Y "$TRACE_FILE")
sleep 5
test "$size_1:$mtime_1" = "$(stat -c %s "$TRACE_FILE"):$(stat -c %Y "$TRACE_FILE")"

git fetch origin perf/qwen3-tts-hidden-h2d-sync
export TRACE_ANALYZER="$QUAL_DIR/analyze_cuda_sync_trace.py"
git show origin/perf/qwen3-tts-hidden-h2d-sync:benchmarks/profiling/analyze_cuda_sync_trace.py \
  > "$TRACE_ANALYZER"
python "$TRACE_ANALYZER" \
  "$TRACE_FILE" \
  --output-dir "$QUAL_DIR/$TRACE_RUN/analysis"
```

Count CPU ownership ranges directly from the trace. This also proves that a
zero synchronization count is not merely an unexercised call path:

```bash
export OWNERSHIP_COUNTS="$QUAL_DIR/$TRACE_RUN/analysis/ownership_range_counts.json"
python - "$TRACE_FILE" "$TRACE_ANALYZER" <<'PY' | tee "$OWNERSHIP_COUNTS"
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

trace_path = Path(sys.argv[1])
analyzer_path = Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("q3tts_sync_analyzer", analyzer_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

expected = {
    "qwen3_tts.preprocess.text_tokenizer",
    "qwen3_tts.preprocess.reference_tokenizer.encode",
    "qwen3_tts.preprocess.speaker_encoder.forward",
    "qwen3_tts.preprocess.prompt.build",
    "qwen3_tts.sampling.base_pipeline",
    "qwen3_tts.vocoder.tokenizer.decode",
}
counts = Counter()
for event in module.iter_trace_events(trace_path):
    if event.get("ph") != "X" or event.get("cat") != "user_annotation":
        continue
    name = event.get("name")
    if name in expected:
        counts[name] += 1
missing = sorted(expected - counts.keys())
print(json.dumps({"counts": dict(sorted(counts.items())), "missing": missing}, indent=2))
if missing:
    raise SystemExit(f"missing ownership ranges: {missing}")
PY
```

Print the newly scoped synchronizations and the residual unscoped mechanisms:

```bash
jq -r '
  .aggregates[]
  | select(.semantic_range | startswith("qwen3_tts."))
  | [.semantic_range, .sync_name, .parent_cpu_op, .transfer_direction,
     .count, .compound_host_block_total_us, .sync_wait_total_us,
     .transfer_bytes_total]
  | @tsv
' "$QUAL_DIR/$TRACE_RUN/analysis/cuda_sync_aggregates.json" \
  | sort > "$QUAL_DIR/$TRACE_RUN/analysis/qwen3_tts_scoped_syncs.tsv"

jq -r '
  .aggregates[]
  | select(.semantic_range == "unscoped")
  | [.sync_name, .parent_cpu_op, .transfer_direction, .count,
     .compound_host_block_total_us, .sync_wait_total_us,
     .transfer_bytes_total]
  | @tsv
' "$QUAL_DIR/$TRACE_RUN/analysis/cuda_sync_aggregates.json" \
  | sort > "$QUAL_DIR/$TRACE_RUN/analysis/residual_unscoped_syncs.tsv"
```

Interpretation gate:

- all six ownership ranges are present on CPU lanes;
- the prior unscoped scalar D2H, pageable H2D, and final D2H occurrences move
  under one of the ownership ranges when they originate in those calls;
- the existing candidate ranges remain synchronization-free;
- explicit async-resolution, reference-publication, cache-key/cache-publication,
  and process-safe payload waits remain separately classified; and
- any residual unscoped group is reported rather than guessed from proximity.

This trace is for choosing the next design, not accepting a performance claim.
Return the stable trace, analyzer outputs, ownership counts, client output, and
server log. A stack-enabled trace is justified only if a material unscoped group
remains after these ownership ranges.
