# Incremental Higgs TTS qualification

This run adds one stable standalone-TTS control to the qualification completed
at carry-forward baseline `a1452e143406bfe94aa1fd2b5203ee68b4308e9d`.
The incremental candidate is the new HEAD and must have its own recorded binary
hash. Carry-forward is justified by semantic isolation: the production change
only preserves one worker-owned speech response metadata header; the remaining
changes affect benchmark artifacts, documentation, and qualification config.
No request routing, request body, response body, or streaming relay mechanics
changed. Do not rerun the router-only, ASR, Qwen3-TTS, or Qwen3-Omni campaigns.

Use normal unseeded requests, the full 1,088-sample SeedTTS English corpus, and
one `bosonai/higgs-tts-3-4b` worker on each H100. Do not rebuild the router or
restart workers within a paired block. Every output directory must be new.

## Prepare

```bash
git status --short --branch
git rev-parse HEAD
python -m benchmarks.dataset.prepare --dataset seedtts
(cd sglang-omni-router && cargo build --release --locked)

mkdir -p results/higgs-policy-config
cp tasks/rust/router-e2e/config/higgs-rust.toml \
  results/higgs-policy-config/round-robin.toml
sed 's/strategy = "round_robin"/strategy = "least_requests"/' \
  tasks/rust/router-e2e/config/higgs-rust.toml \
  > results/higgs-policy-config/least-requests.toml

sglang-omni-router/target/release/sgl-omni-router \
  --config results/higgs-policy-config/round-robin.toml --check-config
sglang-omni-router/target/release/sgl-omni-router \
  --config results/higgs-policy-config/least-requests.toml --check-config
```

Start both workers in Terminal 1:

```bash
export ROUTER_GPU_IDS=0,1
python tasks/rust/router-e2e/scripts/manage_workers.py \
  --config tasks/rust/router-e2e/config/higgs-workers.yaml \
  --gpu-ids "$ROUTER_GPU_IDS" \
  2>&1 | tee results/higgs-workers.log
```

Run the remaining commands from Terminal 2.

## Screen Rust at c8, c16, and c32

Run this once with `POLICY=round_robin CONFIG=round-robin`, then once with
`POLICY=least_requests CONFIG=least-requests`:

```bash
export POLICY=round_robin
export CONFIG=round-robin

python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy "$POLICY" \
  --rust-config "results/higgs-policy-config/${CONFIG}.toml" \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model bosonai/higgs-tts-3-4b \
  --output-dir "results/higgs/screen/rust-${CONFIG}" -- \
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --use-existing-server --generate-only --port {router_port} \
    --model bosonai/higgs-tts-3-4b \
    --meta zhaochenyang20/seed-tts-eval-arrow \
    --concurrencies 8,16,32 \
    --output-dir {output_dir}/audio --disable-tqdm
```

A valid point completes 1,088 requests, has no unexpected 429/5xx response,
uses both healthy workers, and returns all Rust leases to zero. Select the
lowest common concurrency within 3% of the maximum valid Rust throughput.
At that `K`, choose the Rust policy by throughput; break a tie within 2% with
p95 and then p99.

## Select the Rust policies

Use the non-streaming c8/c16/c32 screen to select the non-streaming policy when
the result is clear. Do not repeat it by default. At the selected `K`, run one
full-corpus streaming screen for each Rust policy by using the command below
once with RR and once with LR:

```bash
export K=16
export TRIAL=rr
export POLICY=round_robin
export CONFIG=round-robin

python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate rust --policy "$POLICY" \
  --rust-config "results/higgs-policy-config/${CONFIG}.toml" \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model bosonai/higgs-tts-3-4b \
  --output-dir "results/higgs/rust-stream-screen/${TRIAL}" -- \
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --use-existing-server --generate-only --port {router_port} \
    --model bosonai/higgs-tts-3-4b \
    --meta zhaochenyang20/seed-tts-eval-arrow --concurrency "$K" \
    --stream --output-dir {output_dir}/audio --disable-tqdm
```

Select the streaming policy from these two trials when the result is clear.
Additional paired RR/LR policy trials are allowed only when throughput is
within the 2% tie band, tails conflict with throughput, or a correctness result
is inconsistent. If needed, add two paired rounds in LR/RR then RR/LR order,
using unique output directories. Apply the same conditional rule separately to
non-stream and stream. Do not add a seed or deterministic-inference option.

## Select Python and run the final comparison

At the same selected `K`, screen Python `round_robin` and `least_request` once
each for non-stream and once each for PCM streaming. The Python router command
intentionally has no Rust config; add `--stream` and use a distinct output
directory for the streaming screen:

```bash
export K=16
export POLICY=round_robin
export TRIAL=python-round-robin

python tasks/rust/router-e2e/scripts/run_candidate.py \
  --candidate python --policy "$POLICY" \
  --worker-url http://127.0.0.1:8011 --worker-url http://127.0.0.1:8012 \
  --model bosonai/higgs-tts-3-4b \
  --output-dir "results/higgs/python-screen/${TRIAL}" -- \
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --use-existing-server --generate-only --port {router_port} \
    --model bosonai/higgs-tts-3-4b \
    --meta zhaochenyang20/seed-tts-eval-arrow --concurrency "$K" \
    --output-dir {output_dir}/audio --disable-tqdm
```

Choose the valid Python policy independently for each mode by throughput; break
a tie within 2% with p95 and then p99. Additional Python policy pairs are
conditional under the same ambiguity rules as Rust.

The selected Python-versus-Rust comparison is mandatory: run three paired
full-corpus rounds in AB/BA/AB order for non-stream and three for PCM streaming,
where A is that mode's selected Python policy and B is its selected Rust policy.
Reuse the applicable command above, use a unique
`results/higgs/final/{nonstream,stream}/${TRIAL}` directory for every trial, and
add `--stream` for the PCM trials.

## Direct-worker saturation pair

Run one non-streaming pair at half the selected concurrency per worker. Set
`HALF_K` to `K / 2`:

```bash
export K=16
export HALF_K=8
python tasks/rust/router-e2e/scripts/run_direct_pair.py \
  --worker-port 8011 --worker-port 8012 \
  --output-dir "results/higgs/direct-c${K}" -- \
  python -m benchmarks.eval.benchmark_tts_seedtts \
    --use-existing-server --generate-only --port {router_port} \
    --model bosonai/higgs-tts-3-4b \
    --meta zhaochenyang20/seed-tts-eval-arrow --concurrency "$HALF_K" \
    --output-dir {output_dir}/audio --disable-tqdm
```

## Quality and termination audit

After timed generation, score every retained Higgs output directory with the
same fixed Qwen3-ASR evaluator used by the completed campaign. This is offline
quality scoring, not a rerun of the ASR router workload:

```bash
export AUDIO_DIR=results/higgs/final/nonstream/01-python/audio
export ASR_PORT=8000
python -m benchmarks.eval.benchmark_tts_seedtts \
  --use-existing-server --transcribe-only --port "$ASR_PORT" \
  --model bosonai/higgs-tts-3-4b \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --output-dir "$AUDIO_DIR" --disable-tqdm
```

For every non-streaming request, audit `finish_reason`, `completion_tokens`,
audio duration, and WER together. Streaming `finish_reason` may be null because
the worker sends HTTP headers before generation terminates; never infer a value.

Classify failures without deleting or waiving evidence:

- A failure in a selected Rust final trial fails the candidate.
- A failure confined to a rejected policy disqualifies that policy/configuration.
- The same generation failure across Python and Rust is a cross-router/model
  failure and remains recorded as a model defect.
- Any request bytes, response bytes, framing, header, lease, or disconnect
  difference attributable to Rust fails the router candidate.

Record the incremental result in `HIGGS_RESULTS_TEMPLATE.md`. Stop both workers
with one `Ctrl-C`, then verify ports 30000, 8011, and 8012 are closed.
