# Higgs sampling experiment

Run these commands in the GPU container from this checkout. The experiment
targets one H100, TP=1, Higgs BF16 weights and FP32 sampling. It is disabled by
default and does not change Moss or the shared SGLang sampler.

## 1. Validate the CUDA sampling contract

```bash
python -m benchmarks.profiling.higgs_sampling \
  --mode validate --output-dir results/higgs_sampling/validate

python -m pytest -q \
  tests/unit_test/higgs_tts/test_unseeded_sampling.py \
  tests/unit_test/higgs_tts/test_seeded_sampling.py \
  tests/unit_test/higgs_tts/test_batched_step.py
```

The standalone validator starts separate baseline and candidate processes.
It checks masked-token exclusion and categorical frequencies, probability
immutability, baseline RNG/output parity, mixed seeded/greedy rows, row
reordering, graph RNG advancement, BOC/EOC wind-down, finished rows, and pool
reuse. It compares deterministic result digests across the two processes.
Validation synchronizes intentionally; its timing is not a benchmark.

## 2. Capture a short real-pipeline profile

```bash
python -m benchmarks.profiling.higgs_sampling_profile \
  --mode profile --lang en --concurrency 16 \
  --profile-requests 32 --profile-seconds 5 \
  --output-dir results/higgs_sampling/profile_graph
```

This starts a fresh baseline server, warms it, requests a five-second trace while
32 requests run, then repeats with the candidate in a fresh server. Requests
finish normally after the trace window closes. Repeat with `--graph off`
and a different output directory for eager attribution.
The driver waits for both Higgs worker recorders before submitting traffic and
for closed recorder descriptors plus completed gzip exports before stopping the
server. This requires same-user Linux `/proc` access and `gzip` in the container;
the HTTP control response itself is only a broadcast acknowledgement.

Each leg saves the server command, switches, logs, benchmark outputs, raw
trace/events, stage breakdown, and graph-use histogram. A trace with no Higgs
decode events is rejected. The histogram reports live requests and the
forward batch metadata, **not an inferred CUDA graph bucket**; sampler shapes
are logged once per shape during startup/capture.

The ranges are nested: `higgs.sampler_fsm` includes `higgs.sampler`, which
includes filtering and both draws. Do not sum their inclusive times.
`higgs.pack_and_d2h_submit` records submission, not device completion.
Python ranges inside a captured graph do not execute at replay. Inspect the
`higgs.forward` range and GPU graph nodes for production decode. If the torch
trace lacks node detail, use a separate Nsight Systems run with
`--trace=cuda,nvtx,osrt --cuda-graph-trace=node`; do not enable two CUDA
profilers simultaneously.

## 3. Measure full-dataset serving performance at C16

```bash
python -m benchmarks.profiling.higgs_sampling_profile \
  --mode benchmark --lang en --concurrency 16 --repetitions 5 \
  --output-dir results/higgs_sampling/full_en_c16

python -m benchmarks.profiling.higgs_sampling_profile \
  --mode benchmark --lang zh --concurrency 16 --repetitions 5 \
  --output-dir results/higgs_sampling/full_zh_c16
```

Benchmark mode always loads the **entire selected split**. It has no sample
limit, and `--profile-requests` does not shorten it. The default source is the
repository-pinned full SeedTTS dataset. `dataset.json` records the actual count,
ordered prompts and reference-audio hashes. Pass `--meta /data/en/meta.lst`
for a frozen local split, or `--dataset-revision SHA` for a dataset revision.
The model is resolved once to a local snapshot shared by every server launch;
use `--model-revision SHA` to select a commit explicitly. A local `--model`
directory is used directly and must remain unchanged throughout the campaign.

Five paired repetitions mean ten fresh server launches per command, alternating
baseline/candidate order. Both modes use identical capacity (default 64),
graph settings, normal async decode, references, and warmup. Every warmup must
succeed. Full-run trace ranges and torch/event profiling are off. WAVs and
per-request results are saved for later quality evaluation. An incomplete
cohort fails the run. Existing output directories are refused.

`pairs.json` preserves run summaries; `comparison.json` contains per-pair
relative changes and bootstrap confidence intervals. A positive throughput
change or negative latency/RTF change is favorable. These measurements do not
establish quality parity or guarantee a gain. Compare output duration, generated
token counts where the endpoint supplies them, cap hits from diagnostic/server
evidence, and the worst changed clips. The speech API does not guarantee token
usage/finish-reason telemetry; missing fields must not be counted as zero.

Use `--stream` in a separate output directory for first PCM chunk, chunk gaps,
and playback continuity. Use `--seed 42` and `--temperature 0` as separate
output controls. Higgs still executes and discards the unseeded draw on these
rows, so they can show a timing change. Unseeded output is not required to be
byte-identical; different output lengths can confound serving throughput.
Use `--graph off` for a separate diagnostic comparison, not the primary claim.

The driver owns and stops only its own server process group, including startup
failure and interruption. It does not kill unrelated GPU processes. Use an
exclusive GPU allocation and avoid running quality scorers during timing.
The driver needs a free port (default 8000) and a local server; it grants media
access to the directory containing the benchmark's reference files.

## 4. Isolate draw cost from full-sampler cost

```bash
python -m benchmarks.profiling.higgs_sampling \
  --mode bench --batch-sizes 1 2 4 8 16 32 64 \
  --pairs 3 --output-dir results/higgs_sampling/microbench
```

This measures the actual draw helper, complete production sampler (including
the discarded seeded branch), and stateful sampler/FSM. Both eager and CUDA
graph execution use eight codebooks and 1026 vocabulary entries. Fixtures span
flat/peaked/long-tail probabilities, default/filtered sampling, and one partially
occupied bucket. Synthetic cb0 EOC is suppressed to keep timed work active.

CPU enqueue time and CUDA event intervals are reported separately per block.
The event interval includes GPU idle gaps between eager submissions; it is not
a sum of kernel times. Warmup, graph capture, validation, and state reset are
outside timed blocks. For a small **instrumented** micro trace, add
`--trace --batch-sizes 16 --iterations 5 --blocks 1 --pairs 1`.
Do not use instrumented timings as the speed result.

## 5. Score saved speech after the performance runs

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model bosonai/higgs-audio-v3-tts-4b --lang en --transcribe-only \
  --skip-gpu-cleanup \
  --output-dir results/higgs_sampling/full_en_c16/pair_01/baseline

python -m benchmarks.eval.benchmark_tts_seedtts \
  --model bosonai/higgs-audio-v3-tts-4b --lang en --similarity-only \
  --output-dir results/higgs_sampling/full_en_c16/pair_01/baseline
```

Repeat for candidate, other pairs and ZH. Keep scorer versions/checkpoints
identical. For a custom dataset, pass the same `--meta` to the scorers and keep
the reference files frozen. Inspect repetition, truncation and missing tails in addition to
WER/CER and speaker similarity. Run existing streaming/cancellation/recovery
checks before promoting the experiment.

## Switches and call path

- `SGLANG_OMNI_HIGGS_USE_GUMBEL_SAMPLE=0|1`: default 0. On CUDA FP32
  probabilities, 1 draws exponential noise, clamps it away from zero, and
  selects `argmax(probs / noise)`. Other device/dtype paths retain multinomial.
- `SGLANG_OMNI_HIGGS_PROFILE_SAMPLING=0|1`: default 0. At 0, range
  decorators return the original callables without tracing wrappers.

Both switches resolve before graph capture and require a restart to change.
The driver sets them explicitly for each child server.

`before_decode → shadow buffers → forward/backbone → modality head →
batched_step_direct → _sample_independent_batched → temperature/filters →
unseeded and seeded draws → row selections → delay/EOC state → GPU pack →
host collect`. Prefill reaches the same sampler through `batched_step`.
The existing request state, RNG seed/position mapping, filters, precision,
stopping, and publication paths are preserved.

The candidate removes multinomial's invalid-distribution checks. It requires
finite nonnegative probability rows with positive mass; it does not sanitize
invalid inputs or claim the baseline's error behavior. Leave it disabled until
CUDA validation and real-model quality/performance evidence support enabling it.
