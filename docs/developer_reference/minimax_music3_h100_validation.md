# MiniMax Music 3 H100 validation

This guide compares a candidate change with the merged MiniMax Music 3 support
on an exclusive H100 host. It tests correctness first, then quality, and only
then performance. Keep the two arms identical except for the Git revision.

The supported layouts are:

- one GPU, with AR and DIT/DAV colocated;
- two GPUs, with AR on GPU 0 and DIT/DAV on GPU 1.

The second layout separates model stages across processes and GPUs. It is not
SGLang prefill/decode disaggregation, which MiniMax Music 3 does not support.
Tensor parallelism and external audio streaming are also unsupported and belong
in the negative test matrix, not the success matrix.

## Pin both arms

Use the merge commit as the baseline and record the exact candidate commit:

```bash
export MINIMAX_BASE_REV=05e268a4
export MINIMAX_CANDIDATE_REV="$(git rev-parse HEAD)"
export MINIMAX_MODEL_REV=bd348f9c49ea3c1b39f33ace3436f8fad435f24e
export MINIMAX_RUN_ROOT=/sgl-workspace/minimax-music3-ab
export MINIMAX_SOURCE_MODEL=/sgl-workspace/sglang-omni/MiniMax-Music3

mkdir -p "$MINIMAX_RUN_ROOT"
git worktree add --detach "$MINIMAX_RUN_ROOT/baseline" "$MINIMAX_BASE_REV"
git worktree add --detach "$MINIMAX_RUN_ROOT/candidate" "$MINIMAX_CANDIDATE_REV"
hf download MiniMaxAI/MiniMax-Music3 \
  --revision "$MINIMAX_MODEL_REV" \
  --local-dir "$MINIMAX_SOURCE_MODEL"
```

Use separate virtual environments because both installs are editable. Run this
once in each worktree, changing only the worktree path:

```bash
cd "$MINIMAX_RUN_ROOT/baseline"
uv venv .venv-ab -p 3.12
source .venv-ab/bin/activate
uv pip uninstall flashinfer-cubin
uv pip install -v -e .
deactivate

cd "$MINIMAX_RUN_ROOT/candidate"
uv venv .venv-ab -p 3.12
source .venv-ab/bin/activate
uv pip uninstall flashinfer-cubin
uv pip install -v -e .
deactivate
```

Do not let the baseline and candidate share the same writable checkpoint view.
The baseline rewrites the legacy backbone `config.json` during startup. On a
Linux H100 host, hard-linked private views avoid duplicating the weight files;
the baseline unlinks its own config entry before writing, so the source remains
unchanged:

```bash
cp -al "$MINIMAX_SOURCE_MODEL" "$MINIMAX_RUN_ROOT/model-baseline"
cp -al "$MINIMAX_SOURCE_MODEL" "$MINIMAX_RUN_ROOT/model-candidate"
```

Before either server starts, save the environment and checkpoint identity:

```bash
{
  nvidia-smi -q
  nvidia-smi topo -m
  nvcc --version
  uname -a
} > "$MINIMAX_RUN_ROOT/host.txt"

for arm in baseline candidate; do
  cd "$MINIMAX_RUN_ROOT/$arm"
  {
    git rev-parse HEAD
    .venv-ab/bin/python -V
    uv pip freeze --python .venv-ab/bin/python
  } > "$MINIMAX_RUN_ROOT/$arm-environment.txt"
done

sha256sum \
  "$MINIMAX_SOURCE_MODEL/language_model/config.json" \
  "$MINIMAX_SOURCE_MODEL/language_model/model.safetensors.index.json" \
  "$MINIMAX_SOURCE_MODEL/qwen_7B/qwen_7B/config.json" \
  "$MINIMAX_SOURCE_MODEL/qwen_7B/qwen_7B/model.safetensors.index.json" \
  "$MINIMAX_SOURCE_MODEL/flowmatching_vae.pth" \
  "$MINIMAX_SOURCE_MODEL/dav.pth" \
  > "$MINIMAX_RUN_ROOT/checkpoint-files.sha256"
```

The released `language_model` must contain exactly the backbone tensors from
the legacy mixed checkpoint. This CPU check reads one tensor at a time and is a
one-time correctness gate for the checkpoint revision:

```bash
MINIMAX_MODEL="$MINIMAX_SOURCE_MODEL" \
  "$MINIMAX_RUN_ROOT/candidate/.venv-ab/bin/python" - <<'PY'
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

root = Path(os.environ["MINIMAX_MODEL"])
old_root = root / "qwen_7B" / "qwen_7B"
new_root = root / "language_model"
old = json.loads((old_root / "model.safetensors.index.json").read_text())["weight_map"]
new = json.loads((new_root / "model.safetensors.index.json").read_text())["weight_map"]
backbone = {
    name: shard
    for name, shard in old.items()
    if not name.startswith(("model.audio_decoder.", "model.audio_extra_embedding"))
}
assert backbone.keys() == new.keys()
for name, new_shard in new.items():
    with safe_open(old_root / backbone[name], framework="pt", device="cpu") as f:
        expected = f.get_tensor(name)
    with safe_open(new_root / new_shard, framework="pt", device="cpu") as f:
        actual = f.get_tensor(name)
    assert torch.equal(actual, expected), name
    del actual, expected
print(f"matched {len(new)} backbone tensors")
PY
```

## Run order

Use an exclusive host, fixed clocks if the environment permits them, and no
other GPU work. Run `baseline, candidate, candidate, baseline` for each layout
to expose thermal or host drift. Restart the server between arms. Do not collect
a Torch trace during timing runs.

For the single-GPU arm:

```bash
cd "$MINIMAX_RUN_ROOT/candidate"
source .venv-ab/bin/activate
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path "$MINIMAX_RUN_ROOT/model-candidate" \
  --port 8000 2>&1 | tee "$MINIMAX_RUN_ROOT/candidate-single.log"
```

For the two-GPU stage-separated arm:

```bash
cd "$MINIMAX_RUN_ROOT/candidate"
source .venv-ab/bin/activate
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path "$MINIMAX_RUN_ROOT/model-candidate" \
  --port 8000 2>&1 | tee "$MINIMAX_RUN_ROOT/candidate-dual.log"
```

Replace both `candidate` path components with `baseline` for the baseline arm.
Wait for `curl --fail http://localhost:8000/health` before sending traffic.
Capture `curl http://localhost:8000/model_info` and the startup log for every
run. The candidate startup must leave both config files byte-identical, create
no `.bak`, select `language_model`, and retain the native Qwen3 decode and RVQ
CUDA graphs.

## Correctness matrix

Run every executable request in the [MiniMax Music 3 cookbook](../cookbook/minimax_music3.md)
against both layouts and both revisions. Save response headers as well as WAVs.
The minimum matrix is:

1. the curl, `requests`, and OpenAI client examples;
2. `/health`, `/v1/models`, `/model_info`, one valid
   `/v1/audio/speech/batch` request, and a batch containing one invalid item;
3. 250-, 750-, 1,500-, and 9,000-frame caps;
4. explicit seeds 0, 1, 2, 3, and 2^64-1, plus omitted seed;
5. two repeats alone, two repeats in a fixed five-request set, and a fixed set
   at concurrency 3, 16, and 32;
6. default admission and `--max-running-requests 32`;
7. every unsupported parameter in the cookbook, an empty lyric, an empty
   caption, seed -1, seed 2^64, zero frames, 9,001 frames, and a prompt over
   5,000 tokens;
8. unsupported `/generate`, chat-completion, HTTP streaming, and speech
   WebSocket requests, confirming that none is silently treated as music
   generation;
9. unsupported tensor-parallel and prefill/decode-disaggregation launch
   configurations, which must fail before model loading;
10. client disconnect and server-side abort during AR decode and during acoustic
   decode;
11. one-GPU and two-GPU placement, followed by a restart of each layout.

For each successful speech render require:

- HTTP 200 and a readable, non-empty WAV;
- 32,000 Hz, two channels, finite samples, and no gross clipping or DC offset;
- `X-Finish-Reason`, prompt-token, completion-token, and engine-time headers;
- completion frames consistent with duration at 25 frames per second;
- no scheduler exception, CUDA error, OOM, leaked request, or unexpected eager
  fallback in the server log.

Save an objective manifest for every WAV directory:

```bash
WAV_DIR="$MINIMAX_RUN_ROOT/candidate-single-wavs" \
  "$MINIMAX_RUN_ROOT/candidate/.venv-ab/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

for path in sorted(Path(os.environ["WAV_DIR"]).glob("*.wav")):
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    assert sample_rate == 32_000, path
    assert audio.shape[1] == 2 and audio.size, path
    assert np.isfinite(audio).all(), path
    print(json.dumps({
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "frames": len(audio),
        "seconds": len(audio) / sample_rate,
        "peak": float(np.abs(audio).max()),
        "rms": float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))),
        "dc": float(np.mean(audio, dtype=np.float64)),
        "clipped_fraction": float(np.mean(np.abs(audio) >= 0.999)),
    }))
PY
```

For each rejected request, record the status and JSON error. A request-contract
error should be HTTP 400; an internal model, transport, or CUDA failure should
remain HTTP 500. Do not make the A/B harness accept either class.

The CFG scheduler also needs a constrained-KV run. First test normal pressure by
lowering `--mem-fraction-static` enough to observe a retraction without
preventing startup. Then run a dedicated debug pass with:

```bash
SGLANG_TEST_RETRACT=1 SGLANG_TEST_RETRACT_INTERVAL=3 \
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path "$MINIMAX_RUN_ROOT/model-candidate" --port 8000
```

Send at least two concurrent 750-frame requests. A valid implementation must
never send an odd or non-adjacent `[conditioned, unconditional]` batch to the
model runner. If forced retraction produces a pair-layout assertion or aborts
unrelated requests, preserve the log and stop qualification; this is a
scheduler correctness failure, not a quality or performance result.

## Determinism and quality

Use exact SHA-256 equality only when revision, checkpoint, layout, request
order, admission timing, and batch shape are fixed. Record fixed-shape repeats
separately from c1-versus-concurrent comparisons. Continuous batching can
change BF16 kernel shapes; after the first AR code differs, the autoregressive
trajectory and song length can diverge even though the per-request RNG is
correct.

For the default FP32 acoustic configuration:

- require byte-identical repeats within each fixed execution shape;
- compare baseline and candidate hashes only after the backbone tensor check
  above passes;
- compare one-GPU and two-GPU hashes under the same fixed request schedule;
- if hashes differ, locate the first differing AR frame/code before judging the
  waveform. Correlation after an AR divergence is not a useful quality metric.

The BF16 acoustic mode is a separate supported-path test, not a replacement for
the default. Save this config and launch it with `sgl-omni serve --config`:

```yaml
config_cls: MiniMaxMusic3DualGPUPipelineConfig
model_path: /sgl-workspace/minimax-music3-ab/model-candidate
runtime_overrides:
  dit_dav:
    dtype: bfloat16
    attention_backend: torch_sdpa
```

Require a successful 250-frame render, finite DIT outputs, and FP32 RoPE phase
construction before the phase tensors are cast to BF16. Compare a single DIT
window with the MiniMax/Hugging Face implementation in Diffusers PR
[#14456](https://github.com/huggingface/diffusers/pull/14456), pinned to commit
`c6da9936e4bda83107943a16eb8682e9a37d8527`, before treating BF16 as qualified.
Do not compare BF16 audio byte-for-byte with the FP32 default.

For listening tests, render at least five fixed prompts spanning vocal pop,
rock, acoustic, electronic instrumental, and orchestral material. Randomize the
baseline/candidate labels and compare:

- lyric intelligibility and missing/repeated lines;
- caption, genre, instrumentation, tempo, and structure adherence;
- clicks, discontinuities, clipping, hiss, collapse to mono, and tail quality;
- overall preference.

Keep the original and loudness-matched listening results separate. Five songs
are useful for detecting a severe regression, not for claiming a population
quality improvement. Frozen H200 WAVs are illustrative samples rather than a
cross-hardware numerical oracle.

## Performance and resource checks

Warm up with one 250-frame and one 750-frame request before measuring. For each
arm and layout, run at least three measured repetitions at c1, c3, c16, and
c32. Use the same prompt set and alternate arm order. Record:

- startup time and CUDA-graph capture time;
- end-to-end latency, p50/p95, completed audio frames, and audio-seconds per
  wall-second;
- request-level AR prefill-to-first-chunk and AR-to-DIT/DAV handoff time;
- steady and peak GPU memory on each device;
- GPU utilization, power, clocks, and host CPU utilization;
- retraction count, graph bucket selection, and eager fallbacks.

The request event profiler adds less disturbance than a Torch trace:

```bash
curl -X POST http://localhost:8000/start_request_profile \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"minimax-c16","event_dir":"/sgl-workspace/minimax-music3-ab/events"}'
# Run one dedicated c16 pass.
curl -X POST http://localhost:8000/stop_request_profile \
  -H 'Content-Type: application/json' -d '{"run_id":"minimax-c16"}'
python -m sglang_omni.profiler \
  "$MINIMAX_RUN_ROOT/events" --format json \
  --out "$MINIMAX_RUN_ROOT/events-c16.json"
```

Use Torch or Nsight profiling only after ordinary measurements identify a
regression. The correctness changes are outside the steady-state default hot
path, so a repeatable performance difference should be explained, not waived.
Do not set a pass threshold from a single run: report the samples, median, and
spread for both arms and rerun any difference comparable to run-to-run noise.

## Qualification record

The final record must identify:

- both Git commits, the checkpoint revision, package lock/freeze, CUDA/driver,
  GPU SKU, topology, and launch command;
- all raw request JSON, headers, WAVs, logs, profiler events, and timing samples;
- pass/fail by layout and concurrency;
- fixed-shape determinism separately from batch-shape sensitivity;
- objective audio checks separately from blind listening notes;
- every unsupported or untested surface.

Do not declare the candidate qualified if a correctness gate fails, even when
the WAVs that completed sound good. Performance and listening results become
actionable only after request lifecycle, scheduler pairing, checkpoint
immutability, and API behavior are correct.
