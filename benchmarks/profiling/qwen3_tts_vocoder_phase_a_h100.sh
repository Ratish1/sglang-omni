#!/usr/bin/env bash
# Local qualification runner for the Qwen3-TTS direct-vocoder prototype.
# This file belongs to the debug branch and is not a production PR artifact.

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a pinned Qwen3-TTS-12Hz-1.7B-Base snapshot}"

PORT="${PORT:-8000}"
QUAL_DIR="${QUAL_DIR:-/tmp/q3tts-vocoder-phase-a-$(git rev-parse --short=8 HEAD)}"
CONFIG_PATH="${CONFIG_PATH:-examples/configs/qwen3_tts_1_7b.yaml}"
SERVER_LOG="$QUAL_DIR/server.log"
TRACE_RUN="q3tts-vocoder-phase-a-c16"
FIXED_CODE_RESULT="$QUAL_DIR/fixed_code_differential.json"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

git merge-base --is-ancestor 6ea90f91a HEAD
git merge-base --is-ancestor e92d8b11d HEAD
mkdir -p "$QUAL_DIR"
rm -f "$FIXED_CODE_RESULT"

MODEL_PATH="$MODEL_PATH" \
FIXED_CODE_RESULT="$FIXED_CODE_RESULT" \
PYTHONUNBUFFERED=1 \
python - <<'PY' 2>&1 | tee "$QUAL_DIR/fixed_code_differential.log"
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from sglang_omni.models.qwen3_tts.stages import _load_qwen3_tts_tokenizer
from sglang_omni.models.qwen3_tts.streaming_vocoder import (
    Qwen3TTSStreamingVocoderScheduler,
)


tokenizer = _load_qwen3_tts_tokenizer(
    os.environ["MODEL_PATH"],
    device="cuda:0",
    dtype="bfloat16",
    attn_implementation=None,
)
scheduler = Qwen3TTSStreamingVocoderScheduler(
    tokenizer,
    device="cuda:0",
    initial_cuda_graph=False,
)
decoder_config = tokenizer.model.config.decoder_config
num_quantizers = int(decoder_config.num_quantizers)
codebook_size = int(decoder_config.codebook_size)
generator = torch.Generator(device="cpu").manual_seed(20260824)
records = []

for batch_size in (1, 2, 8):
    lengths = [8 + 2 * row for row in range(batch_size)]
    codes = [
        torch.randint(
            0,
            codebook_size,
            (length, num_quantizers),
            dtype=torch.long,
            generator=generator,
        )
        for length in lengths
    ]
    expected, expected_sample_rate = tokenizer.decode(
        [{"audio_codes": item} for item in codes]
    )
    actual = scheduler._decode_nonstreaming_codes(codes)
    assert expected_sample_rate == scheduler._sample_rate
    assert len(expected) == len(actual) == batch_size
    hashes = []
    for row, (want, got) in enumerate(zip(expected, actual)):
        got_array = got.detach().to(torch.float32).cpu().numpy()
        if want.shape != got_array.shape or not np.array_equal(want, got_array):
            max_abs = (
                float(np.max(np.abs(want - got_array)))
                if want.shape == got_array.shape
                else None
            )
            raise AssertionError(
                f"B={batch_size} row={row} waveform mismatch: "
                f"expected={want.shape}, actual={got_array.shape}, max_abs={max_abs}"
            )
        hashes.append(hashlib.sha256(got_array.tobytes()).hexdigest())
    record = {
        "batch_size": batch_size,
        "frame_lengths": lengths,
        "sample_lengths": [int(item.shape[0]) for item in expected],
        "sha256": hashes,
    }
    records.append(record)
    print(record, flush=True)

result = {
    "schema_version": 1,
    "status": "pass",
    "seed": 20260824,
    "torch_version": torch.__version__,
    "qwen_tts_version": version("qwen-tts"),
    "records": records,
}
result_path = Path(os.environ["FIXED_CODE_RESULT"])
temporary_path = result_path.with_suffix(result_path.suffix + ".tmp")
temporary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
temporary_path.replace(result_path)
print("fixed-code official/direct differential: PASS", flush=True)
PY

python - "$FIXED_CODE_RESULT" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert result["status"] == "pass"
assert [record["batch_size"] for record in result["records"]] == [1, 2, 8]
assert all(len(record["sha256"]) == record["batch_size"] for record in result["records"])
PY

if [[ "${FIXED_CODE_ONLY:-0}" == "1" ]]; then
  exit 0
fi

CUDA_VISIBLE_DEVICES=0 \
SGLANG_TORCH_PROFILER_DIR="$QUAL_DIR/profiles" \
sgl-omni serve \
  --model-path "$MODEL_PATH" \
  --config "$CONFIG_PATH" \
  --port "$PORT" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 300); do
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "Server exited during startup; inspect $SERVER_LOG" >&2
    exit 1
  fi
  sleep 2
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --model "$MODEL_PATH" \
  --output-dir "$QUAL_DIR/warmup" \
  --max-samples 64 \
  --sample-offset 1024 \
  --concurrency 16 \
  --warmup 0

mkdir -p "$QUAL_DIR/$TRACE_RUN/events"
curl -fsS -X POST "http://127.0.0.1:$PORT/start_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$TRACE_RUN\",\"trace_path_template\":\"$QUAL_DIR/$TRACE_RUN/trace\",\"event_dir\":\"$QUAL_DIR/$TRACE_RUN/events\",\"enable_torch\":true}" \
  | tee "$QUAL_DIR/$TRACE_RUN/start_response.json"

for _ in $(seq 1 60); do
  if grep -Fq "Starting End-to-End Torch profiler (run_id=$TRACE_RUN)" "$SERVER_LOG"; then
    break
  fi
  sleep 1
done
grep -Fq "Starting End-to-End Torch profiler (run_id=$TRACE_RUN)" "$SERVER_LOG"

python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only \
  --use-existing-server \
  --model "$MODEL_PATH" \
  --output-dir "$QUAL_DIR/$TRACE_RUN/client" \
  --max-samples 64 \
  --sample-offset 128 \
  --concurrency 16 \
  --warmup 0

curl -fsS -X POST "http://127.0.0.1:$PORT/stop_profile" \
  -H 'content-type: application/json' \
  -d "{\"run_id\":\"$TRACE_RUN\"}" \
  | tee "$QUAL_DIR/$TRACE_RUN/stop_response.json"

for _ in $(seq 1 120); do
  TRACE_FILE=$(find "$QUAL_DIR/$TRACE_RUN" -name '*.trace.json.gz' -size +0c -print -quit)
  if [[ -n "$TRACE_FILE" ]]; then
    break
  fi
  sleep 2
done
: "${TRACE_FILE:?No finalized trace gzip found}"

size_1=$(stat -c %s "$TRACE_FILE")
mtime_1=$(stat -c %Y "$TRACE_FILE")
sleep 5
size_2=$(stat -c %s "$TRACE_FILE")
mtime_2=$(stat -c %Y "$TRACE_FILE")
test "$size_1:$mtime_1" = "$size_2:$mtime_2"

grep -F "Qwen3-TTS non-streaming code staging capacity" "$SERVER_LOG" \
  >"$QUAL_DIR/$TRACE_RUN/staging_capacity.log" || true
grep -E "Traceback|CUDA error|cudaError|HTTP/[0-9.]+ [45][0-9][0-9]" "$SERVER_LOG" \
  >"$QUAL_DIR/$TRACE_RUN/server_error_scan.log" || true

cleanup
SERVER_PID=""
trap - EXIT

ARTIFACT_PATH="${QUAL_DIR}.tar.gz"
tar -czf "$ARTIFACT_PATH" -C "$(dirname "$QUAL_DIR")" "$(basename "$QUAL_DIR")"
echo "Phase A artifacts: $ARTIFACT_PATH"
