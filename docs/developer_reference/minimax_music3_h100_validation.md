# MiniMax Music 3 H100 A/B validation

Use this guide to compare a MiniMax Music 3 change with the merged release on
an exclusive H100. The complete cookbook is part of the A/B test: do not test
only one smoke prompt or only the candidate.

This guide calls the merged revision **A** and the candidate revision **B**.
Correctness is the first gate. Sound quality and performance are evaluated only
after both arms complete the same requests successfully.

## Fixed inputs

Pin the code and released checkpoint before running either arm:

```bash
export MINIMAX_A_REV=05e268a4fde2aeefbf5ccf1945f57d509b2ae20b
export MINIMAX_B_REV=origin/fix/minimax-music3-correctness-v2
export MINIMAX_MODEL_REV=bd348f9c49ea3c1b39f33ace3436f8fad435f24e
export MINIMAX_DIFFUSERS_REV=dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d
export MINIMAX_ROOT=/sgl-workspace/minimax-music3-ab
export MINIMAX_SOURCE_MODEL="$MINIMAX_ROOT/model-source"

mkdir -p "$MINIMAX_ROOT"
git fetch origin fix/minimax-music3-correctness-v2
git worktree add --detach "$MINIMAX_ROOT/a" "$MINIMAX_A_REV"
git worktree add --detach "$MINIMAX_ROOT/b" "$MINIMAX_B_REV"
export MINIMAX_B_REV="$(git -C "$MINIMAX_ROOT/b" rev-parse HEAD)"

hf download MiniMaxAI/MiniMax-Music3 \
  --revision "$MINIMAX_MODEL_REV" \
  --local-dir "$MINIMAX_SOURCE_MODEL"
```

Start from a pristine checkpoint. The source must have neither a
`config.json.bak` file nor a rewritten legacy config left over from an earlier
A run.

Use one virtual environment per editable checkout. Save both package freezes;
apart from the editable repository path, they must agree.

```bash
for arm in a b; do
  cd "$MINIMAX_ROOT/$arm"
  uv venv .venv-h100 -p 3.12
  uv pip install --python .venv-h100/bin/python -v -e .
  uv pip uninstall --python .venv-h100/bin/python flashinfer-cubin
  uv pip freeze --python .venv-h100/bin/python \
    > "$MINIMAX_ROOT/$arm-packages.txt"
done
```

Do not share one writable model directory. A rewrites the legacy backbone
config during startup, while B must leave the released checkpoint unchanged.
Hard-linked private views share the large immutable weights without sharing a
directory entry that A replaces:

```bash
cp -al "$MINIMAX_SOURCE_MODEL" "$MINIMAX_ROOT/model-a"
cp -al "$MINIMAX_SOURCE_MODEL" "$MINIMAX_ROOT/model-b"
```

Record the host and checkpoint before starting a server:

```bash
{
  date --iso-8601=seconds
  nvidia-smi -q
  nvidia-smi topo -m
  nvcc --version
  uname -a
} > "$MINIMAX_ROOT/host.txt"

find "$MINIMAX_SOURCE_MODEL" -name '*.bak' -print \
  > "$MINIMAX_ROOT/source-backups-before.txt"
test ! -s "$MINIMAX_ROOT/source-backups-before.txt"

find "$MINIMAX_SOURCE_MODEL" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$MINIMAX_ROOT/source-before.sha256"

find "$MINIMAX_ROOT/model-b" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$MINIMAX_ROOT/model-b-before.sha256"
```

## Gate 1: prove the new backbone is the same model

B loads the released `language_model` Qwen3 checkpoint instead of rewriting
the mixed legacy checkpoint. Before treating that as equivalent, require exact
tensor equality between all native backbone tensors and the non-audio tensors
in the legacy checkpoint:

```bash
MINIMAX_MODEL="$MINIMAX_SOURCE_MODEL" \
  "$MINIMAX_ROOT/b/.venv-h100/bin/python" - <<'PY'
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

root = Path(os.environ["MINIMAX_MODEL"])
legacy_root = root / "qwen_7B" / "qwen_7B"
native_root = root / "language_model"
legacy = json.loads(
    (legacy_root / "model.safetensors.index.json").read_text()
)["weight_map"]
native = json.loads(
    (native_root / "model.safetensors.index.json").read_text()
)["weight_map"]
backbone = {
    name: shard
    for name, shard in legacy.items()
    if not name.startswith(
        ("model.audio_decoder.", "model.audio_extra_embedding.")
    )
}
assert backbone.keys() == native.keys()
for name, native_shard in native.items():
    with safe_open(
        legacy_root / backbone[name], framework="pt", device="cpu"
    ) as file:
        expected = file.get_tensor(name)
    with safe_open(
        native_root / native_shard, framework="pt", device="cpu"
    ) as file:
        actual = file.get_tensor(name)
    assert torch.equal(actual, expected), name
print(f"matched {len(native)} backbone tensors")
PY
```

The required result for the pinned release is `matched 399 backbone tensors`.
Do not continue if a name, shape, dtype, or value differs.

## Gate 2: start both supported layouts

The supported layouts are:

- single GPU: AR and DIT/DAV colocated on GPU 0;
- dual GPU: AR on GPU 0 and DIT/DAV on GPU 1.

The dual-GPU layout is stage separation, not SGLang prefill/decode
disaggregation. Tensor parallelism and external audio streaming are not part of
the success matrix.

Use these commands for B and replace both `b` path components with `a` for A:

```bash
# Single GPU
cd "$MINIMAX_ROOT/b"
CUDA_VISIBLE_DEVICES=0 .venv-h100/bin/sgl-omni serve \
  --model-path "$MINIMAX_ROOT/model-b" --port 8000 \
  2>&1 | tee "$MINIMAX_ROOT/b-single.log"

# Dual GPU, in a separate run
cd "$MINIMAX_ROOT/b"
CUDA_VISIBLE_DEVICES=0,1 .venv-h100/bin/sgl-omni serve \
  --model-path "$MINIMAX_ROOT/model-b" --port 8000 \
  2>&1 | tee "$MINIMAX_ROOT/b-dual.log"
```

For every launch, wait for `/health`, save `/model_info`, and record steady and
peak memory on every GPU. B must:

- load the native `language_model` as Qwen3;
- load the audio embedding and RVQ weights from the released audio checkpoint;
- capture the normal Qwen decode and RVQ CUDA graphs;
- leave both model configs byte-for-byte unchanged;
- create no `.bak` file.

After the first B launch, verify its private checkpoint view directly:

```bash
find "$MINIMAX_ROOT/model-b" -type f -print0 | sort -z | \
  xargs -0 sha256sum > "$MINIMAX_ROOT/model-b-after.sha256"
diff -u \
  "$MINIMAX_ROOT/model-b-before.sha256" \
  "$MINIMAX_ROOT/model-b-after.sha256"
test -z "$(find "$MINIMAX_ROOT/model-b" -name '*.bak' -print -quit)"
```

Rerun the source hash command after all A and B launches and compare it with
`source-before.sha256` as a separate check that neither private view wrote
through to the pristine source.

## Gate 3: A/B the complete cookbook

Run **every executable command** in the
[MiniMax Music 3 cookbook](../cookbook/minimax_music3.md) against A and B. Use
the exact same JSON bytes, request order, client concurrency, checkpoint,
layout, and GPU assignment. Do not rewrite prompts or replace the documented
clients with a smaller smoke script.

The complete A/B includes:

1. the first-song curl request;
2. the Python `requests` example;
3. the OpenAI client example;
4. the arena-rock, structured-blues, and short ambient requests;
5. the seed 1/2/3 loop, an explicit-seed repeat, and an omitted-seed repeat;
6. all five full reference requests;
7. the exact concurrent client example;
8. default 16-request admission;
9. `--max-running-requests 32` and a complete 32-request load;
10. every documented rejected parameter;
11. structure-tag, Markdown, and special-caption-tag normalization;
12. the prompt limit and maximum frame boundaries.

Run the full list once in the single-GPU layout and once in the dual-GPU
layout for each revision. Preserve response headers, response JSON for errors,
WAVs, and server logs. Use filenames containing all four dimensions, for
example `a_dual_01_jpop_bright.wav` and `b_dual_01_jpop_bright.wav`.

Run A then B for the complete matrix. Repeat the five reference requests in
the reverse order, B then A, after fresh server starts. This distinguishes a
code difference from host temperature, clocks, or residual server state.

Every successful response must be a readable, non-empty, finite 32 kHz stereo
WAV. Record its SHA-256, frame count, duration, peak, RMS, DC offset, clipped
sample fraction, finish reason, prompt tokens, completion frames, and engine
time. Every rejected request must return the same status and error category in
A and B. Preserve any mismatch instead of normalizing it in the client.

## Gate 4: fixed-revision hidden and latent comparison

The default FP32 path should not change numerically. Run one 750-frame c1
request with seed 42 against fresh A and B servers. Enable the existing hidden
dump on both arms:

```bash
MINIMAX_MUSIC3_HIDDEN_DUMP="$MINIMAX_ROOT/hidden-a" \
  CUDA_VISIBLE_DEVICES=0,1 "$MINIMAX_ROOT/a/.venv-h100/bin/sgl-omni" serve \
  --model-path "$MINIMAX_ROOT/model-a" --port 8000

MINIMAX_MUSIC3_HIDDEN_DUMP="$MINIMAX_ROOT/hidden-b" \
  CUDA_VISIBLE_DEVICES=0,1 "$MINIMAX_ROOT/b/.venv-h100/bin/sgl-omni" serve \
  --model-path "$MINIMAX_ROOT/model-b" --port 8000
```

These are separate launches. Send the identical request to each, stop the
server, and compare every same-named dumped tensor with `torch.equal`. The
overlapping 100 frames in adjacent 200-frame chunks must also be exactly equal
within each arm.

For the fixed-commit latent comparison, feed the same dumped hidden chunks,
seed, chunk index, initial overlap latent, and initial overlap condition into A
and B. Capture and compare, in order:

1. the aligned DIT condition;
2. the initial Gaussian noise;
3. the latent after each of the 30 Euler steps;
4. the final latent and saved overlap latent;
5. the DAV waveform before resampling;
6. the final 32 kHz waveform.

Apply any temporary capture hook identically to both worktrees and do not
commit it. Require exact equality for the FP32 A/B. On the first mismatch,
record the tensor name, step, first differing index, maximum absolute error,
and maximum relative error. Do not judge a later waveform after an earlier AR
code or latent divergence.

## Gate 5: branch-specific edge paths

### Overlap override

Launch B with this stage override:

```yaml
config_cls: MiniMaxMusic3DualGPUPipelineConfig
model_path: /sgl-workspace/minimax-music3-ab/model-b
runtime_overrides:
  minimax_music3_ar:
    server_args_overrides:
      disable_overlap_schedule: false
```

B must still resolve `disable_overlap_schedule=true`, start normally, and
produce the same fixed c1 output as B without the override. This proves an
unsupported public override cannot re-enable the overlap event loop.

### BF16 acoustic path

Run both arms with the same BF16 acoustic configuration:

```yaml
config_cls: MiniMaxMusic3DualGPUPipelineConfig
model_path: /sgl-workspace/minimax-music3-ab/model-b
runtime_overrides:
  dit_dav:
    dtype: bfloat16
    attention_backend: torch_sdpa
```

Change only `model-b` to `model-a` for A. Record A's actual outcome; do not
assume it fails. B must complete a 250-frame render with finite output. Confirm
that the rotary inverse-frequency buffer and phase calculation remain FP32 and
that only the cosine and sine tensors applied to BF16 activations are cast.

BF16 is a supported-path test, not the default A/B quality oracle. If A cannot
render BF16, compare B BF16 with B FP32 perceptually and compare a fixed DIT
window with the pinned released Diffusers implementation. Do not require BF16
and FP32 WAV hashes to match.

The direct required-state accesses need no synthetic test. Successful startup,
prefill, CUDA-graph RVQ replay, decode, chunk emission, and final WAV production
exercise the affected `self_attn`, schedule-batch, graph, and feedback-buffer
paths. Any missing required state must fail at its owner rather than be replaced
with a fallback state.

## Gate 6: blind sound-quality comparison

The complete cookbook A/B produces the listening set. Do not listen only to a
single short clip. At minimum, the blinded set must span vocal pop, rock,
acoustic, electronic instrumental, orchestral, short 250-frame audio, and a
natural early stop.

Keep the A/B mapping hidden from the evaluator. Randomize each pair independently
and use the original WAVs first. For every pair record:

- lyric intelligibility, missing lines, repeated lines, and pronunciation;
- caption, genre, instrumentation, tempo, and song-structure adherence;
- vocal naturalness and balance against the accompaniment;
- clicks, discontinuities, clipping, hiss, stereo collapse, and tail quality;
- whether the files are indistinguishable, or which one is preferred.

Then repeat with loudness-matched copies, keeping those scores separate from
the original-output scores. Do not use loudness matching to replace the exact
files produced by the server.

When an A/B pair is byte-identical, sound quality is proven unchanged for that
request and execution shape; listening is only a sanity check. When hashes
differ, first establish whether the AR hidden/code trajectory or DIT latent
diverged. A low waveform correlation after an autoregressive divergence does
not identify a quality regression by itself.

## Gate 7: performance and resources

After correctness passes, benchmark A and B with the same fixed prompt corpus
at c1, c3, c16, and c32. Warm up each fresh server with one 250-frame and one
750-frame request. Run at least three measured repetitions and alternate
`A, B, B, A` to expose host drift.

Record startup and graph-capture time, end-to-end latency, p50/p95, completed
audio frames, audio-seconds per wall-second, peak memory per GPU, GPU
utilization, power, clocks, CPU utilization, selected graph buckets, and eager
fallbacks. Report every sample, median, and spread. Do not derive a regression
threshold from a single run.

## Scheduler boundary probes

Run these three probes from
`test/minimax-music3-scheduler-boundaries` on the candidate only. They exercise
the released model through the HTTP server; they do not replace the A/B gates
above and do not constitute a benchmark.

Use two exclusive H100s and record the exact diagnostic revision:

```bash
export MINIMAX_PROBE_ROOT=/sgl-workspace/minimax-music3-scheduler-probes
export MINIMAX_MODEL=/sgl-workspace/minimax-music3-ab/model-b
mkdir -p "$MINIMAX_PROBE_ROOT"
git rev-parse HEAD > "$MINIMAX_PROBE_ROOT/revision.txt"
```

### Pair admission

The checked-in config sets `max_prefill_tokens=256`. The probe uses the released
music tokenizer to construct a prompt with `128 < P < 256` tokens, so one
physical row fits the prefill budget while its `2P`-token CFG pair does not.

Start the server in terminal 1:

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 .venv-h100/bin/sgl-omni serve \
  --config examples/configs/minimax_music3_pair_admission.yaml \
  --model-path "$MINIMAX_MODEL" --port 8000 \
  2>&1 | tee "$MINIMAX_PROBE_ROOT/pair-admission-server.log"
```

After `/health` is ready, run in terminal 2:

```bash
.venv-h100/bin/python scripts/minimax_music3_scheduler_probe.py \
  pair-admission \
  --model-path "$MINIMAX_MODEL" \
  --server-log "$MINIMAX_PROBE_ROOT/pair-admission-server.log" \
  --output "$MINIMAX_PROBE_ROOT/pair-admission.json"
```

`reproduced` requires both a failed HTTP request and a server traceback saying
the batch has an odd number of rows or non-adjacent CFG rows. `handled` requires
a successful WAV. Any other result is `inconclusive`; preserve the JSON and log.

### Decode retraction

The next config targets 16 logical requests as 32 CFG rows in a 7,200-token KV
pool. The clients use the cookbook's 250-frame ambient request, which normally
reaches the cap. This is intended to make actual decode growth exceed SGLang's
admission estimate. The report records the released tokenizer's exact prompt
count; use the server log, not this target, to establish how many rows ran.

Start a fresh server without `SGLANG_TEST_RETRACT`:

```bash
unset SGLANG_TEST_RETRACT
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 .venv-h100/bin/sgl-omni serve \
  --config examples/configs/minimax_music3_kv_pressure.yaml \
  --model-path "$MINIMAX_MODEL" --port 8000 \
  2>&1 | tee "$MINIMAX_PROBE_ROOT/kv-pressure-server.log"
```

Then run:

```bash
.venv-h100/bin/python scripts/minimax_music3_scheduler_probe.py \
  kv-pressure \
  --model-path "$MINIMAX_MODEL" \
  --server-log "$MINIMAX_PROBE_ROOT/kv-pressure-server.log" \
  --output "$MINIMAX_PROBE_ROOT/kv-pressure.json"
```

Only the log prefix `KV cache pool is full. Retract requests.` proves that the
production memory-pressure condition occurred. A subsequent CFG-pair or replay
error is `reproduced`; 16 successful responses after that prefix is `handled`.
`not_triggered` is inconclusive and must not be reported as a pass.

As a separate control, restart the same server with
`SGLANG_TEST_RETRACT=1`:

```bash
set -o pipefail
SGLANG_TEST_RETRACT=1 CUDA_VISIBLE_DEVICES=0,1 \
  .venv-h100/bin/sgl-omni serve \
  --config examples/configs/minimax_music3_kv_pressure.yaml \
  --model-path "$MINIMAX_MODEL" --port 8000 \
  2>&1 | tee "$MINIMAX_PROBE_ROOT/forced-retract-server.log"
```

Run the same client against the new log and output paths:

```bash
.venv-h100/bin/python scripts/minimax_music3_scheduler_probe.py \
  kv-pressure \
  --model-path "$MINIMAX_MODEL" \
  --server-log "$MINIMAX_PROBE_ROOT/forced-retract-server.log" \
  --output "$MINIMAX_PROBE_ROOT/forced-retract.json"
```

The expected report label is `fault_injection_reproduced` when the row-wise
path fails. This proves the mechanics of the retraction path only; it is not
evidence that a default server reaches it.

### Public retract pause

Start a fresh server with the ordinary dual-GPU launch from Gate 2 and no
retraction environment variable. The probe starts one long request, polls
`/model_info` until the AR stage reports its two CFG rows running, calls
`pause_generation` with `mode=retract`, continues generation, and waits for the
original request.

```bash
unset SGLANG_TEST_RETRACT
set -o pipefail
CUDA_VISIBLE_DEVICES=0,1 .venv-h100/bin/sgl-omni serve \
  --model-path "$MINIMAX_MODEL" --port 8000 \
  2>&1 | tee "$MINIMAX_PROBE_ROOT/pause-retract-server.log"
```

After `/health` is ready, run in terminal 2:

```bash
.venv-h100/bin/python scripts/minimax_music3_scheduler_probe.py \
  pause-retract \
  --server-log "$MINIMAX_PROBE_ROOT/pause-retract-server.log" \
  --output "$MINIMAX_PROBE_ROOT/pause-retract.json"
```

Pass `--admin-api-key` when the server protects admin endpoints. `reproduced`
requires that live rows were observed, both admin calls succeeded, and the
request then failed with MiniMax's CFG-pair or retract/replay invariant.
`handled` requires the original request to complete successfully. A request
that finished before live rows were observed is `not_triggered`, not a pass.

## Qualification record

Return one artifact directory containing:

- exact A, B, model, and Diffusers revisions;
- host data, package freezes, launch commands, and checkpoint hashes;
- every cookbook request, response header/error, WAV, and server log from both
  arms and both layouts;
- the objective WAV manifest and paired A/B diff;
- hidden/latent comparison tensors and the first-difference report, if any;
- blind and loudness-matched listening sheets with the mapping kept separately;
- raw benchmark samples and resource traces;
- a pass/fail row for every gate above.

Do not qualify B from valid WAVs alone. The candidate passes only when the full
cookbook A/B is complete, default FP32 numerics are unchanged, model files are
not mutated, both supported layouts work, the BF16 and overlap paths behave as
specified, and no material sound-quality or performance regression remains.
