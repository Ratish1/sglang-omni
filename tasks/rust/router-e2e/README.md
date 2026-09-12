# Rust router H100 comparison

This package runs direct-worker, Python-router, and Rust-router comparisons
against one persistent two-worker model deployment. Run every command from the
repository root in the H100 container. GPU performance results from other hosts
are not qualification evidence.

The matrix runner tests these candidates in the requested order at every
aggregate concurrency:

```text
two persistent workers on GPUs 0 and 1
                  |
                  +-> direct worker pair
                  +-> Python round robin
                  +-> Rust round robin
                  +-> Python least request
                  '--> Rust least requests
```

Only the router restarts between trials. The workers, request corpus, sample
order, and model arguments remain fixed.

## Prepare

```bash
cd /sgl-workspace/sglang-omni

export OMNI_PYTHON="$PWD/.venv/bin/python"
export RUST_ROUTER_BIN="$PWD/sglang_omni_router/rust/target/release/sgl-omni-router"
export RUN_ROOT="results/router-ab-$(date -u +%Y%m%dT%H%M%SZ)"

"$OMNI_PYTHON" -m benchmarks.dataset.prepare --dataset seedtts
"$OMNI_PYTHON" -m benchmarks.dataset.prepare --dataset mmmu

cargo build --release --locked \
  --manifest-path sglang_omni_router/rust/Cargo.toml

git rev-parse HEAD
sha256sum "$RUST_ROUTER_BIN"
ulimit -n
nvidia-smi
mkdir -p "$RUN_ROOT"
```

The Rust TOMLs are generated from `tests/test_model/rust_router_config.py`, the
same renderer used by model CI. No qualification-only router limits or worker
profiles are maintained.

## Router-only baseline

This command starts synthetic workers and measures small JSON, a 1 MiB upload,
and SSE relay at c1/c8/c32/c128/c512:

```bash
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/router_microbench.py \
  --rust-binary "$RUST_ROUTER_BIN" \
  --oha-timeout-s 300 \
  --output-dir "$RUN_ROOT/router-only"
```

## Model workflow

Use two terminals. Terminal 1 owns the workers. Terminal 2 runs one complete
matrix. Stop the workers only after every mode for that model has completed.

The benchmark command after `--` may contain these placeholders:

| Placeholder | Value |
| --- | --- |
| `{router_port}` | router port, or the direct worker port |
| `{router_url}` | router URL, or the direct worker URL |
| `{output_dir}` | unique evidence directory for the trial |
| `{concurrency}` | aggregate concurrency requested from the matrix |
| `{target_concurrency}` | aggregate concurrency for a router; half for each direct worker |
| `{candidate}` | `direct`, `python-rr`, `rust-rr`, `python-lr`, or `rust-lr` |

Always inspect the exact resolved commands before launching the matrix by
adding `--print-plan` to the `run_matrix.py` invocation.

### ASR

Start two workers, replacing `MODEL` with the selected CI model path:

```bash
export MODEL="Qwen/Qwen3-ASR-1.7B"

"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/manage_workers.py \
  --model-path "$MODEL" \
  --model-name "$MODEL" \
  --gpu-ids 0,1 \
  --worker-base-port 8011 \
  --wait-timeout 900 \
  2>&1 | tee "$RUN_ROOT/asr-workers.log"
```

Run the full 1,088-sample SeedTTS English corpus at every point:

```bash
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_matrix.py \
  --topology asr \
  --model "$MODEL" \
  --worker-url http://127.0.0.1:8011 \
  --worker-url http://127.0.0.1:8012 \
  --concurrencies 32,64,96,128 \
  --candidates direct,rust-rr,rust-lr \
  --rust-binary "$RUST_ROUTER_BIN" \
  --output-dir "$RUN_ROOT/asr/nonstream" -- \
  "$OMNI_PYTHON" -m benchmarks.eval.benchmark_asr_seedtts \
    --port '{router_port}' \
    --model-path "$MODEL" \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --lang en \
    --max-samples 0 \
    --concurrencies '{target_concurrency}' \
    --repeats 1 \
    --warmup \
    --output '{output_dir}/asr.json' \
    --save-raw-dir '{output_dir}/raw'
```

Repeat with a new output directory and add `--stream` for transcription SSE.
Run this pair for:

| CI key | Model |
| --- | --- |
| `fun` | pinned `FunAudioLLM/Fun-ASR-Nano-2512-hf` revision from `AsrCiPreset` |
| `qwen3` | `Qwen/Qwen3-ASR-1.7B` |
| `whisper` | `openai/whisper-large-v3` |

Use the exact model path printed by the CI preset when a model is locally
pinned; do not replace it with a different revision.

MOSS-Transcribe-Diarize uses its own full-corpus benchmark. Start workers with
the CI engine limits:

```bash
export MODEL="OpenMOSS-Team/MOSS-Transcribe-Diarize"
export WORKER_ARGS="--asr.engine.max_running_requests 16 \
--asr.engine.cuda_graph_max_bs 16 \
--mem-fraction-static 0.80"

"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/manage_workers.py \
  --model-path "$MODEL" \
  --model-name "$MODEL" \
  --worker-extra-args "$WORKER_ARGS" \
  --gpu-ids 0,1 \
  --worker-base-port 8011 \
  --wait-timeout 900 \
  2>&1 | tee "$RUN_ROOT/moss-td-workers.log"
```

Then run `run_matrix.py --topology asr` with this benchmark template:

```bash
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
  benchmarks.eval.benchmark_asr_transcribe_diarize -- \
  --use-existing-server \
  --base-url '{router_url}' \
  --model-path "$MODEL" \
  --dataset movies800times \
  --max-concurrency '{target_concurrency}' \
  --output-dir '{output_dir}/movies800times' \
  --disable-tqdm
```

Repeat with `--stream`. AISHELL4-long and GoogleTime remain correctness
replays after selecting the routing policy and concurrency from Movies800Time.

### Standalone TTS

Set the model and its current CI arguments before starting the workers. This
example is Qwen3-TTS:

```bash
export MODEL="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
export REF_FORMAT="references"
export WORKER_ARGS="--allowed-local-media-path /tmp \
--tts_engine.engine.max_running_requests 64 \
--tts_engine.engine.cuda_graph_max_bs 64 \
--tts_engine.engine.torch_compile_max_bs 64 \
--vocoder.process vocoder \
--tts_engine.gpu_memory_fraction 0.85 \
--vocoder.gpu_memory_fraction 0.10"

"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/manage_workers.py \
  --model-path "$MODEL" \
  --model-name "$MODEL" \
  --worker-extra-args "$WORKER_ARGS" \
  --gpu-ids 0,1 \
  --worker-base-port 8011 \
  --wait-timeout 900 \
  2>&1 | tee "$RUN_ROOT/tts-workers.log"
```

Run the full SeedTTS corpus without `--max-samples`. Generation remains
unseeded, matching normal serving:

```bash
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_matrix.py \
  --topology tts \
  --model "$MODEL" \
  --worker-url http://127.0.0.1:8011 \
  --worker-url http://127.0.0.1:8012 \
  --concurrencies 16,32 \
  --candidates direct,rust-rr,rust-lr \
  --rust-binary "$RUST_ROUTER_BIN" \
  --output-dir "$RUN_ROOT/tts/wav" -- \
  "$OMNI_PYTHON" -m benchmarks.eval.benchmark_tts_seedtts \
    --use-existing-server \
    --generate-only \
    --port '{router_port}' \
    --model "$MODEL" \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --ref-format "$REF_FORMAT" \
    --concurrencies '{target_concurrency}' \
    --output-dir '{output_dir}/audio' \
    --disable-tqdm
```

Repeat with a new output directory and add `--stream` for PCM streaming. Use
the current `TtsCiPreset` values for each model:

| CI key | Model | Reference format | Additional benchmark argument |
| --- | --- | --- | --- |
| `higgs` | `bosonai/higgs-tts-3-4b` | `flat` | none |
| `moss` | `OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5` | `references` | `--token-count auto` |
| `qwen3-tts` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | `references` | none |

The worker arguments must come from the same preset. Do not reuse the Qwen3
engine and memory settings for Higgs or MOSS.

The unchanged Qwen3-TTS CI preset is valid at aggregate c16 and c32. It raises
`max_running_requests` to 64 per worker but leaves
`max_queued_requests` at 16, and every request enters the waiting queue first.
The direct workers, Python RR, and Rust RR therefore all reject a synchronized
c64 aggregate burst with `The request queue is full.` This is not a router
failure, and c64/c96/c128 do not need to be repeated against that preset.

To evaluate a higher worker operating point separately, restart both workers
with this additional argument and measure the direct curve before routed runs:

```bash
--tts_engine.engine.max_queued_requests 64
```

Latest `upstream/main` rejects a MOSS-TTS Local empty audio generation before
the zero-byte tensor reaches the SHM relay. Rerun PCM c32 on the updated revision.
The worker should remain alive, but an empty generation is still a failed
request and makes that point invalid.

### Routed Qwen3-Omni

Use only the existing two-H100 colocated DP2 topologies. Start one complete
worker per GPU with the exact worker arguments from `tests/test_model/conftest.py`.
Do not convert the direct TP2 or disaggregated two-GPU stages into router tests.

For Omni audio, use `--topology omni_audio` and the same SeedTTS-50 repository
as CI:

```bash
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_matrix.py \
  --topology omni_audio \
  --model qwen3-omni \
  --worker-url http://127.0.0.1:8011 \
  --worker-url http://127.0.0.1:8012 \
  --concurrencies 16,32 \
  --candidates direct,rust-rr,rust-lr \
  --rust-binary "$RUST_ROUTER_BIN" \
  --output-dir "$RUN_ROOT/omni/audio" -- \
  "$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
    benchmarks.eval.benchmark_omni_seedtts -- \
    --base-url '{router_url}' \
    --model qwen3-omni \
    --meta zhaochenyang20/seed-tts-eval-50-arrow \
    --max-concurrency '{target_concurrency}' \
    --voice-clone \
    --generate-only \
    --output-dir '{output_dir}/seedtts' \
    --disable-tqdm
```

Repeat with `--stream` and a new output directory. Use the exact routed CI
datasets and request arguments for the remaining Omni text-output stages:

| Stage | Dataset and request count | Aggregate concurrency |
| --- | --- | --- |
| MMMU | `zhaochenyang20/mmmu-ci-50`, 50 requests, warmup 2 | 16, 32 |
| MMSU | `zhaochenyang20/mmsu-ci-2000`, 2,000 requests, warmup 0 | 16, 32, 64, 96, 128 |
| Video-AMME | `zhaochenyang20/Video_AMME_ci`, 50 requests | 16, 32 |

MMMU and Video-AMME must not load their full upstream evaluation splits. MMSU
CI is not a small subset: all 2,000 requests are part of its current contract.
Use `--topology omni_text`,
`--candidates direct,rust-rr,rust-lr`, and the corresponding benchmark template:

```bash
# MMMU
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
  benchmarks.eval.benchmark_omni_mmmu -- \
  --base-url '{router_url}' --model qwen3-omni \
  --repo-id zhaochenyang20/mmmu-ci-50 --max-samples 50 \
  --warmup 2 --max-concurrency '{target_concurrency}' \
  --output-dir '{output_dir}/mmmu' --disable-tqdm

# MMSU
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
  benchmarks.eval.benchmark_omni_mmsu -- \
  --base-url '{router_url}' --model qwen3-omni --modalities text \
  --repo-id zhaochenyang20/mmsu-ci-2000 --max-tokens 32 --warmup 0 \
  --max-concurrency '{target_concurrency}' \
  --output-dir '{output_dir}/mmsu' --disable-tqdm

# Video-AMME
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
  benchmarks.eval.benchmark_omni_videoamme -- \
  --base-url '{router_url}' --model qwen3-omni \
  --repo-id zhaochenyang20/Video_AMME_ci --max-samples 50 \
  --max-concurrency '{target_concurrency}' --video-fps 2 \
  --video-max-frames 128 --video-max-pixels 401408 --timeout-s 500 \
  --output-dir '{output_dir}/videoamme' --disable-tqdm
```

## Final comparison and correctness

The sweep identifies the best valid Rust policy and concurrency separately for
each model and mode. At that exact point, run one additional Python/Rust pair
with the same workers. Alternate which implementation runs first between
successive model modes. Repeat only points within 2%, isolated outliers, or
results inconsistent with adjacent concurrency points.

After selection, run the existing CI correctness stage through the selected
Rust configuration. Score every retained generated-audio directory. A
performance point is valid only when it has the expected sample count, zero
unexpected HTTP failures, healthy workers after the trial, and zero retained
Rust admission or session leases.

Each matrix writes:

- `matrix.json` with the commit, Rust binary hash, resolved commands, order,
  wall time, and return codes;
- one `trial.json`, router log, benchmark log, diagnostics snapshot, CPU total,
  and peak RSS for every routed trial;
- one `direct-pair.json` and child benchmark directory per direct point;
- generated benchmark artifacts under the candidate directory.

## Router commands without the matrix

Generate a current-schema Rust config and start the router directly:

```bash
"$OMNI_PYTHON" tasks/rust/router-e2e/scripts/render_router_config.py \
  --topology tts \
  --policy least_requests \
  --router-port 30000 \
  --worker-url http://127.0.0.1:8011 \
  --worker-url http://127.0.0.1:8012 \
  --model "$MODEL" \
  --output "$RUN_ROOT/router.toml"

"$RUST_ROUTER_BIN" --config "$RUN_ROOT/router.toml"
```

The equivalent Python comparison is:

```bash
"$OMNI_PYTHON" -m sglang_omni_router.python.serve \
  --host 127.0.0.1 \
  --port 30000 \
  --worker-urls http://127.0.0.1:8011 http://127.0.0.1:8012 \
  --model "$MODEL" \
  --policy least_request \
  --log-level info
```
