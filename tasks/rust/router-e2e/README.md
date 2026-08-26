# Rust router qualification

This package answers three separate questions:

1. Do the existing Rust tests prove every supported route, protocol, failure, and cleanup contract?
2. When the proxy is the bottleneck, is the Rust process materially faster and more CPU-efficient than the Python router?
3. With two identical H100 workers, does Rust preserve model quality and avoid throughput or tail regressions while using less host CPU?

All Rust Omni, TTS, and ASR routing behavior is in the single `sglang-omni-router/` crate. The legacy `sglang_omni_router/` Python product remains separate for comparison and RL use; the two source directories are not merged.

Run every command from the repository root. Do not run another benchmark, model server, or GPU job on the selected CPUs or GPUs.

## Prerequisites

```bash
git status --short --branch
git rev-parse HEAD
python -m benchmarks.dataset.prepare --dataset seedtts
python -m benchmarks.dataset.prepare --dataset seedtts-50
python -m benchmarks.dataset.prepare --dataset mmmu-ci-50
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
- least requests is a mechanics sentinel only. It does not reopen the H100 policy search.

## Layer C: two-H100 model qualification

Use one complete worker per H100 at ports 8011 and 8012, with the candidate router at 30000. Keep both workers alive while switching routers. Never rebuild or restart workers within an A/B block.

Use three measured paired rounds in this order:

| Trial | Candidate |
| ---: | --- |
| 1 | Python round robin (A) |
| 2 | Rust round robin (B) |
| 3 | Rust round robin (B) |
| 4 | Python round robin (A) |
| 5 | Python round robin (A) |
| 6 | Rust round robin (B) |

This is `AB`, `BA`, `AB`: three paired rounds with the first four trials forming ABBA. Add two rounds (`BA`, then `AB`) only when the observed ordering/noise can change the decision. Run one separate Python `least_request` trial at the selected concurrency to record current CI behavior; never use it for the implementation-isolation comparison.

### Terminal layout

Terminal 1 owns workers for one topology and remains open:

```bash
export ROUTER_GPU_IDS=0,1  # for example, 2,3 on a reserved pair
python tasks/rust/router-e2e/scripts/manage_workers.py \
  --config tasks/rust/router-e2e/config/asr-workers.yaml \
  --gpu-ids "$ROUTER_GPU_IDS" \
  2>&1 | tee results/asr-workers.log
```

Terminal 2 runs one candidate trial at a time. Python spelling is `round_robin` or `least_request`; Rust spelling is `round_robin`. Example:

```bash
python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust \
  --policy round_robin \
  --rust-config tasks/rust/router-e2e/config/asr-rust.toml \
  --worker-url http://127.0.0.1:8011 \
  --worker-url http://127.0.0.1:8012 \
  --model Qwen/Qwen3-ASR-1.7B \
  --output-dir results/asr-final/02-rust \
  -- \
  python -m benchmarks.eval.benchmark_asr_seedtts \
    --port {router_port} \
    --model-path Qwen/Qwen3-ASR-1.7B \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --lang en --max-samples 0 --concurrencies 32 --repeats 1 --warmup \
    --output {output_dir}/asr.json --save-raw-dir {output_dir}/raw
```

For Python, replace the candidate/config/policy fields with:

```text
--candidate python --policy round_robin
```

and omit `--rust-config`. The wrapper starts only the router, waits for `/ready` or `/health`, captures diagnostics and worker metrics, samples the complete router process group, runs the benchmark without a shell, verifies zero Rust in-flight ownership, and stops the router cleanly.

### ASR: Qwen3-ASR 1.7B

Worker config: `config/asr-workers.yaml`. Rust config: `config/asr-rust.toml`.

Find the saturation knee with a 256-sample screen for both candidates:

```bash
python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy round_robin \
  --rust-config tasks/rust/router-e2e/config/asr-rust.toml \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model Qwen/Qwen3-ASR-1.7B --output-dir results/asr-screen/rust -- \
  python -m benchmarks.eval.benchmark_asr_seedtts \
    --port {router_port} --model-path Qwen/Qwen3-ASR-1.7B \
    --meta zhaochenyang20/seed-tts-eval-arrow --lang en --max-samples 256 \
    --concurrencies 1,16,32,64 --repeats 1 --warmup \
    --output {output_dir}/asr.json --save-raw-dir {output_dir}/raw
```

Repeat for Python round robin. Select the lowest concurrency within 3% of maximum throughput with no correctness or tail failure. The screen supplies the c1 latency point; do not repeat a full corpus serially. Execute the six-trial order on the full 1,088-sample EN corpus using `--max-samples 0 --concurrencies K`. Repeat the six trials with `--stream` to qualify SSE and TTFT while preserving WER.

### Standalone TTS: Qwen3-TTS 1.7B Base

Stop the ASR workers, confirm ports 8011/8012 are closed, then launch `config/tts-workers.yaml`. It contains the exact tuned Qwen3-TTS CI worker arguments from `tests/test_model/tts_ci_config.py`. Rust config: `config/tts-rust.toml`.

Screen c1/c16/c32/c64:

```bash
python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy round_robin \
  --rust-config tasks/rust/router-e2e/config/tts-rust.toml \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --output-dir results/tts-screen/rust -- \
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --use-existing-server --generate-only --port {router_port} \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --meta zhaochenyang20/seed-tts-eval-50-arrow --max-samples 50 \
    --ref-format references --concurrencies 1,16,32,64 \
    --output-dir {output_dir}/audio --disable-tqdm
```

Repeat for Python round robin, choose `K` by the same 3% knee rule, and run six measured non-stream trials with `--concurrency K` instead of the sweep option. Run six more with `--concurrency K --stream`. The screen supplies c1; do not regenerate it in every final trial. Every generated WAV/non-stream output and reconstructed streaming PCM WAV must be readable, nonempty, and have the expected 50 samples. Use the benchmark’s existing WER phase when making the final quality comparison; do not invent another score.

After all TTS generation trials, stop the TTS workers, launch the ASR topology, and keep one ASR router fixed while scoring every retained output directory:

```bash
sglang-omni-router/target/release/sgl-omni-router \
  --config tasks/rust/router-e2e/config/asr-rust.toml
```

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --use-existing-server --transcribe-only --port 30000 \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --meta zhaochenyang20/seed-tts-eval-50-arrow \
  --ref-format references --lang en \
  --output-dir results/tts-final/02-rust/audio
```

Repeat only the `--output-dir` for each Python/Rust trial. This scoring phase is not timed router evidence.

### Omni text/image: Qwen3-Omni FP8

Launch `config/omni-fp8-workers.yaml`; use `config/omni-fp8-rust.toml`. Run one paired c1 latency check with 10 samples, then run the six measured rounds at c16 with the complete MMMU-50 set:

```bash
python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy round_robin \
  --rust-config tasks/rust/router-e2e/config/omni-fp8-rust.toml \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model qwen3-omni --output-dir results/omni-fp8-c16/02-rust -- \
  python tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
    benchmarks.eval.benchmark_omni_mmmu -- \
    --base-url {router_url} --model qwen3-omni \
    --repo-id zhaochenyang20/mmmu-ci-50 --max-samples 50 \
    --max-concurrency 16 --warmup 2 --temperature 0 \
    --output-dir {output_dir}/mmmu --disable-tqdm
```

For the c1 check, change to `--max-concurrency 1 --max-samples 10` and use a distinct output directory. The c16 gate requires 50/50 completed samples and unchanged MMMU accuracy.

### Omni audio output: Qwen3-Omni BF16

Launch `config/omni-bf16-workers.yaml`; use `config/omni-bf16-rust.toml`. Run SeedTTS-50 non-stream at c16:

```bash
python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy round_robin \
  --rust-config tasks/rust/router-e2e/config/omni-bf16-rust.toml \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model qwen3-omni --output-dir results/omni-bf16/02-rust -- \
  python tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
    benchmarks.eval.benchmark_omni_seedtts -- \
    --base-url {router_url} --model qwen3-omni \
    --meta zhaochenyang20/seed-tts-eval-50-arrow --max-samples 50 \
    --max-concurrency 16 --voice-clone --generate-only \
    --output-dir {output_dir}/seedtts --disable-tqdm
```

Then confirm streaming TTFT for each candidate:

```bash
python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy round_robin \
  --rust-config tasks/rust/router-e2e/config/omni-bf16-rust.toml \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model qwen3-omni --output-dir results/omni-ttft/02-rust -- \
  python tasks/rust/router-e2e/scripts/run_repo_benchmark.py \
    benchmarks.eval.benchmark_omni_streaming_ttft -- \
    --base-url {router_url} --model qwen3-omni --label rust \
    --warmup 2 --repeats 5 --output {output_dir}/ttft.json
```

After generation is complete, use the same fixed ASR topology to run the benchmark’s existing transcription/WER phase against each retained SeedTTS output directory:

```bash
python -m benchmarks.eval.benchmark_omni_seedtts \
  --transcribe-only --port 30000 --model qwen3-omni \
  --meta zhaochenyang20/seed-tts-eval-50-arrow --lang en \
  --output-dir results/omni-bf16/02-rust/seedtts
```

This quality-only phase is outside the timed Omni router comparison.

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

For matched Python round robin versus Rust round robin, compare medians over the three paired rounds. Rust passes model E2E when throughput is within 2%, p95 is within 5%, p99 is within 10%, and correctness is identical. Router CPU-seconds/request must improve by at least 20% when Linux process metrics are available; peak RSS is reported, not used alone to fail a GPU-bound result. When direct workers establish GPU saturation, equal QPS is expected. The separate Python least-request point describes the current CI policy only.

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
