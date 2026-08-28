# Rust router qualification

This package answers three separate questions:

1. Do the existing Rust tests prove every supported route, protocol, failure, and cleanup contract?
2. When the proxy is the bottleneck, is the Rust process materially faster and more CPU-efficient than the Python router?
3. With two identical H100 workers, does Rust preserve model quality and avoid throughput or tail regressions while using less host CPU?

All Rust Omni, TTS, and ASR routing behavior is in the single `sglang-omni-router/` crate. The legacy `sglang_omni_router/` Python product remains separate for comparison and RL use; the two source directories are not merged.

Run every command from the repository root. Do not run another benchmark, model server, or GPU job on the selected CPUs or GPUs.

The qualification at carry-forward baseline
`a1452e143406bfe94aa1fd2b5203ee68b4308e9d` already completed the router-only,
ASR, Qwen3-TTS, and Qwen3-Omni campaigns. For the incremental Higgs control,
the candidate is the new HEAD and has a different binary hash. Run only
[`HIGGS_INCREMENTAL.md`](HIGGS_INCREMENTAL.md). Do not rerun Layers B or C.

## Prerequisites

```bash
git status --short --branch
git rev-parse HEAD
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.dataset.prepare --dataset mmmu
(cd sglang-omni-router && cargo build --release --locked)
oha --version
ulimit -n
```

The candidate commit must remain fixed for the complete campaign. Use Linux for process-group CPU/RSS measurements. `router_microbench.py` refuses to run without an executable release binary and oha JSON/body-file support. `run_candidate.py` performs the readiness check. The approved Omni module wrapper skips only the benchmark modules' redundant legacy `/health` waiter because the Rust router exposes `/ready`.
If `oha` is absent, install it with `cargo install --locked oha`, then record
`oha --version`. The c512 points require a file-descriptor limit of at least
4096; raise it before the run or omit c512 and record why.

## Layer A: existing contract proof

Do not duplicate these contracts in a qualification harness.

```bash
(cd sglang-omni-router && cargo fmt --all -- --check)
(cd sglang-omni-router && \
  cargo clippy --workspace --all-targets --all-features --locked -- -D warnings)
(cd sglang-omni-router && \
  cargo test --workspace --all-targets --all-features --locked)
```

| Operation | Existing authoritative proof |
| --- | --- |
| Chat, exact body/header relay, request IDs, small/large homogeneous direct path | `tests/chat_http.rs::worker_owns_body_semantics_and_receives_exact_bytes_and_request_id`, `small_and_large_homogeneous_bodies_use_the_same_direct_worker_path` |
| Round robin, health filtering, heterogeneous generation classification | `tests/chat_http.rs::homogeneous_replicas_rotate_and_unhealthy_workers_are_filtered`, `heterogeneous_typed_image_request_reaches_only_the_compatible_worker` |
| Speech, speech batch, transcription, translation, streaming media, multipart | `tests/media_http.rs::relays_all_media_routes_with_exact_bytes_headers_and_large_direct_uploads`, `heterogeneous_media_classification_selects_only_a_correlated_capable_worker` |
| Admission, timeout, reset, early EOF, downstream disconnect, ownership release | `tests/chat_http.rs::relay_holds_admission_and_is_not_cut_off_after_commitment`, `precommit_timeout_and_upstream_reset_are_bounded_and_release_admission`, `early_upload_eof_and_downstream_disconnect_release_admission` |
| Speech and realtime WebSockets, ordering, pinning, setup deadline | `tests/websocket.rs::speech_exact_replay_and_realtime_precommit_and_server_first_ordering`, `setup_deadline_releases_stalled_speech_and_realtime_capacity`, `explicit_realtime_model_selects_and_pins_one_heterogeneous_worker` |
| Voice list/upload/delete and exact owner | `tests/voice_state.rs::exact_owner_voice_crud_preserves_contract_and_upload_ordering` |
| `/live`, `/ready`, `/v1/models`, `/metrics`, `/diagnostics`, request methods, shutdown | `tests/process.rs::serves_exact_local_health_and_operations_routes_and_shuts_down_cleanly` |
| Connection bound and drain behavior | `tests/process.rs::connection_cap_holds_the_next_request_until_capacity_returns`, `graceful_shutdown_waits_for_capped_connection_to_release`, `drain_timeout_terminates_with_capped_connection` |
| Strict config, route subsets, correlated profiles, static worker origins | `tests/config.rs` |

Layer A must pass before performance work. A failure is a product defect, not benchmark noise.

## Layer B: router-only real-socket benchmark

This starts two deterministic HTTP/1.1 keep-alive workers, then measures direct-worker headroom, Python round robin, and Rust round robin at concurrency 1, 8, 32, 128, and 512 for:

- small fixed-length JSON chat;
- 1 MiB fixed-length chat relay;
- chunked SSE response relay.

It also runs one Rust round-robin versus least-requests variable-duration sentinel and three paired Rust `error` versus `info` logging-filter sentinels. The current hot path has no per-request tracing span or log; the logging result should remain within run noise.

```bash
python tasks/rust/router-e2e/scripts/router_microbench.py \
  --output-dir results/router-microbench-$(date -u +%Y%m%dT%H%M%SZ) \
  --rust-binary sglang-omni-router/target/release/sgl-omni-router
```

Record requests/s, failures, p50/p95/p99, aggregate router-process CPU seconds, and peak RSS from `microbench.json`.

Router-only acceptance is deliberately conservative:

- zero load-generator or HTTP failures and no p99 regression greater than 10%;
- direct workers must sustain at least 1.20 times Python throughput. A Rust result at the direct ceiling remains valid when it materially exceeds Python;
- Rust must provide at least 1.15 times Python throughput and 1.15 times CPU efficiency at a conclusive proxy-bound point;
- if `info` versus `error` median throughput or p99 differs by more than 5%, repeat the sentinel with an otherwise idle host; do not attribute the difference until it repeats;
- the least-requests sentinel checks selector mechanics only. The full-model campaign below selects the H100 policy.

## Layer C: two-H100 model qualification

This is the next H100 run. It selects the Rust policy before comparing Rust with Python. Do not change production router code during this campaign.

Use one complete worker per H100 at ports 8011 and 8012 and the temporary router at port 30000. Keep workers alive while switching policies or router implementations. Never rebuild the router or restart workers inside a paired block.

### 1. Prepare the full datasets and policy configs

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.dataset.prepare --dataset mmmu

mkdir -p results/router-policy-config
for NAME in asr tts omni-fp8 omni-bf16; do
  cp "tasks/rust/router-e2e/config/${NAME}-rust.toml" \
    "results/router-policy-config/${NAME}-round-robin.toml"
  sed 's/strategy = "round_robin"/strategy = "least_requests"/' \
    "tasks/rust/router-e2e/config/${NAME}-rust.toml" \
    > "results/router-policy-config/${NAME}-least-requests.toml"
done
```

These are runtime test configs under `results/`; do not commit them. Rust policy spelling is `round_robin` or `least_requests`. Python policy spelling is `round_robin` or `least_request`.

Full-corpus selection is intentionally different between benchmarks:

| Workload | Full-corpus arguments |
| --- | --- |
| ASR | `--meta zhaochenyang20/seed-tts-eval-arrow --max-samples 0` (1,088 EN samples) |
| Standalone and Omni TTS | `--meta zhaochenyang20/seed-tts-eval-arrow` and omit `--max-samples` |
| MMMU | Omit both `--repo-id` and `--max-samples` (all validation subjects, about 900 samples) |

Do not pass `--max-samples 0` to either TTS benchmark: its shared loader treats zero as an empty dataset.

### 2. Start one worker topology

Terminal 1 owns both workers. Replace `asr` with `tts`, `omni-fp8`, or `omni-bf16` for the next topology.

```bash
export ROUTER_GPU_IDS=0,1
python tasks/rust/router-e2e/scripts/manage_workers.py \
  --config tasks/rust/router-e2e/config/asr-workers.yaml \
  --gpu-ids "$ROUTER_GPU_IDS" \
  2>&1 | tee results/asr-workers.log
```

Stop the old topology before changing models and confirm ports 8011 and 8012 are closed. Terminal 2 runs the commands below. `run_candidate.py` starts and stops only the router, captures CPU/RSS and worker counters, verifies worker health, and requires every Rust lease to return to zero.

### 3. Screen both Rust policies on the full corpus

Run every screen once with `POLICY=round_robin`, then repeat it with `POLICY=least_requests`. Set the matching config each time:

```bash
export POLICY=round_robin
export CONFIG_SUFFIX=round-robin
```

or:

```bash
export POLICY=least_requests
export CONFIG_SUFFIX=least-requests
```

ASR, at c32/c64/c96:

```bash
python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy "$POLICY" \
  --rust-config "results/router-policy-config/asr-${CONFIG_SUFFIX}.toml" \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model Qwen/Qwen3-ASR-1.7B \
  --output-dir "results/policy-screen/asr-${CONFIG_SUFFIX}" -- \
  python -m benchmarks.eval.benchmark_asr_seedtts \
    --port {router_port} --model-path Qwen/Qwen3-ASR-1.7B \
    --meta zhaochenyang20/seed-tts-eval-arrow --lang en --max-samples 0 \
    --concurrencies 32,64,96 --repeats 1 --warmup \
    --output {output_dir}/asr.json --save-raw-dir {output_dir}/raw
```

Standalone TTS, at c16/c32/c64. Qualification uses normal unseeded serving requests. The full corpus and paired AB/BA repeated trials below measure stochastic variance. A repeated c64 correctness failure is a measured overload boundary, not a reason to rerun it.

```bash
python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy "$POLICY" \
  --rust-config "results/router-policy-config/tts-${CONFIG_SUFFIX}.toml" \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output-dir "results/policy-screen/tts-${CONFIG_SUFFIX}" -- \
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --use-existing-server --generate-only --port {router_port} \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --meta zhaochenyang20/seed-tts-eval-arrow --ref-format references \
    --concurrencies 16,32,64 \
    --output-dir {output_dir}/audio --disable-tqdm
```

Full MMMU, one complete run at c8/c16/c32:

```bash
for K in 8 16 32; do
  python tasks/rust/router-e2e/scripts/run_candidate.py \
    --candidate rust --policy "$POLICY" \
    --rust-config "results/router-policy-config/omni-fp8-${CONFIG_SUFFIX}.toml" \
    --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
    --model qwen3-omni \
    --output-dir "results/policy-screen/mmmu-${CONFIG_SUFFIX}-c${K}" -- \
    python tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
      benchmarks.eval.benchmark_omni_mmmu -- \
      --base-url {router_url} --model qwen3-omni \
      --max-concurrency "$K" --warmup 2 --temperature 0 \
      --output-dir {output_dir}/mmmu --disable-tqdm
done
```

Full Omni audio, one complete fixed-temperature generation run at c8/c16/c32:

```bash
for K in 8 16 32; do
  python tasks/rust/router-e2e/scripts/run_candidate.py \
    --candidate rust --policy "$POLICY" \
    --rust-config "results/router-policy-config/omni-bf16-${CONFIG_SUFFIX}.toml" \
    --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
    --model qwen3-omni \
    --output-dir "results/policy-screen/omni-audio-${CONFIG_SUFFIX}-c${K}" -- \
    python tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
      benchmarks.eval.benchmark_omni_seedtts -- \
      --base-url {router_url} --model qwen3-omni \
      --meta zhaochenyang20/seed-tts-eval-arrow \
      --max-concurrency "$K" --temperature 0 --voice-clone --generate-only \
      --output-dir {output_dir}/seedtts --disable-tqdm
done
```

### 4. Select concurrency and repeat the Rust policy comparison

For each workload, keep only concurrency points with the full expected sample count, zero unexpected 429/5xx responses, healthy workers, and zero retained Rust leases. Select the lowest concurrency within 3% of that workload's maximum valid throughput. Compare both policies at this same `K`; do not choose a separate concurrency for each policy.

Run three paired full-corpus rounds in this order:

| Round | First | Second |
| ---: | --- | --- |
| 1 | Rust round robin | Rust least requests |
| 2 | Rust least requests | Rust round robin |
| 3 | Rust round robin | Rust least requests |

Use the screen command with only the selected `K` and a new output directory for every trial. Then repeat the paired rounds for the streaming modes:

- ASR: add `--stream`.
- Standalone TTS: replace `--concurrencies ...` with `--concurrency K` and add `--stream`.
- Omni audio: add `--stream`.
- MMMU has no streaming variant.

Do not rerun c1 or every integer concurrency. The three full-corpus points represent below-knee, knee, and pressure behavior. Router-only microbenchmarks are not a substitute for this model test.

### 5. Compare the selected Rust policy with the best Python policy

At the selected `K`, run one full-corpus screen for Python `round_robin` and one for Python `least_request` using the same benchmark command and generation parameters. For Python, omit `--rust-config` and use:

```text
--candidate python --policy round_robin
```

or:

```text
--candidate python --policy least_request
```

Choose the valid Python policy with higher throughput; use p95 and then p99 to break a throughput tie within 2%. Finally run three paired full-corpus rounds of selected Python versus selected Rust in `AB`, `BA`, `AB` order, including each applicable streaming mode. This is the final implementation comparison. Do not compare Rust only against the slower Python policy.

Use the benchmark's existing WER, accuracy, and audio validation. Score all retained standalone/Omni TTS output directories through the same fixed ASR topology after timed generation is complete; this scoring phase is not router performance evidence.

### Direct-worker saturation point

At each topology’s selected high concurrency, run the same benchmark concurrently against ports 8011 and 8012 with half the selected concurrency per command. For example, if ASR selects c32:

```bash
python tasks/rust/router-e2e/scripts/run_direct_pair.py \
  --worker-port 8011 --worker-port 8012 \
  --output-dir results/asr-direct-c32 -- \
  python -m benchmarks.eval.benchmark_asr_seedtts \
    --port {router_port} --model-path Qwen/Qwen3-ASR-1.7B \
    --meta zhaochenyang20/seed-tts-eval-arrow --lang en --max-samples 0 \
    --concurrencies 16 --repeats 1 --warmup \
    --output {output_dir}/asr.json --save-raw-dir {output_dir}/raw
```

The direct pair intentionally processes one corpus per worker. Aggregate completed samples and divide by `direct-pair.json.wall_s`. Use the same pattern for the selected TTS/Omni command. If direct-pair throughput is no higher than routed throughput within run noise, the GPUs/workers are saturated and equal Rust/Python QPS is expected.

## Decision gates

Correctness is absolute for every measured trial:

- expected sample count and existing WER/accuracy/audio-validity checks pass;
- no unexpected 429 or 5xx response;
- both workers show request-counter movement in captured `worker_metrics_before/after`, or unambiguous request traffic in each retained worker log when the worker exposes no counter;
- both workers remain healthy after the trial;
- every Rust admission and worker-capacity `in_flight` value is zero.

Select Rust and Python policies independently, then compare their medians over the three paired rounds at the same concurrency. Rust passes model E2E when throughput is within 2%, p95 is within 5%, p99 is within 10%, and correctness is identical. Router CPU-seconds/request must improve by at least 20% when Linux process metrics are available; peak RSS is reported, not used alone to fail a GPU-bound result. When direct workers establish GPU saturation, equal QPS is expected.

Use `RESULTS_TEMPLATE.md`. Do not average incompatible models, concurrency points, streaming modes, or failed/inconclusive runs.

## Cleanup

Stop Terminal 1 with one `Ctrl-C`; `manage_workers.py` signals both complete worker process groups and escalates if needed. Then verify:

```bash
curl --fail --max-time 1 http://127.0.0.1:30000/live || true
curl --fail --max-time 1 http://127.0.0.1:8011/health || true
curl --fail --max-time 1 http://127.0.0.1:8012/health || true
nvidia-smi
```

Do not delete results until the tables, failures, worker balance, diagnostics, and candidate identity have been checked.
