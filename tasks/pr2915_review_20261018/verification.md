# Verification and limits

## Environment

All CUDA work ran on the existing node-radixark-16-0001 H200 allocation, GPU 0 inside the per-task `sglang-omni-review2915` Podman container. The container uses the normal development image, a pushed Mac review branch and an editable project install. No local pytest was run. No author branch was changed.

Runtime: Torch 2.13.0+cu130, SGLang 0.5.21, FlashInfer Python 0.6.18, Transformers 5.12.1. Exact imported source hashes are in runtime_sources.json. No separate dependency upgrade or experimental sampler implementation was installed.

## Existing test suites

Command, inside the task container:

```bash
python3 -m pytest -q \
  tests/unit_test/qwen3_omni/test_sglang_thinker.py \
  tests/unit_test/qwen3_omni/test_sglang_ar_budget.py \
  tests/unit_test/model_runner/test_install_sampling_seeds.py
```

Result: **58 passed**, 16 dependency warnings, 51.79 seconds. This covers the changed tests, real fused-MoE variant bookkeeping, factory policy and the shared real CUDA seed/fallback sampling tests. No tests were broadened after this pass without a new question to resolve.

## Sampling contract probe

`probe_sampling.py` invokes the registered SGLang sampler on CUDA with real SamplingBatchInfo and float32 logits at vocabulary size 152064. Only the distributed-group boundary is supplied as a single-rank context; no sampler/filter kernel is mocked.

- Fixed-logit greedy, temperature-only, seeded filtered and mixed seeded/unseeded draws preserve the PyTorch-control tokens.
- The four-token counterexample proves the min-p filter-order difference. The two-row case proves it also affects a zero-min-p neighbor.
- The support assertion is deterministic: it reads the real positive support and lengths, not a sample-frequency threshold. The cutoff values avoid equality boundaries. Source inspection confirms those are the same renormalization operations used by the actual draw.
- The tied top-k case is recorded separately as an existing, intentional FlashInfer threshold convention; it is not conflated with F1.

Regression assertion commands, on the same pushed probe:

```bash
python3 tasks/pr2915_review_20261018/probe_sampling.py --assert-min-p-backend pytorch
python3 tasks/pr2915_review_20261018/probe_sampling.py --assert-min-p-backend flashinfer
```

The expected contract is the previous thinker's support `{0, 1}`, both for the min-p request and for its zero-min-p neighbor. The PyTorch command is the old-path control; the FlashInfer command deliberately exposes the current regression. These are review diagnostics, not changes to the author's CI suite.

## Component performance

Warm 100-iteration CUDA-event measurements of the real sampler, including logits cloning, softmax and host submission gaps. These are not kernel-only timings or an end-to-end throughput benchmark. A second diagnostic run showed:

| Rows | PyTorch control, microseconds | FlashInfer, microseconds |
|---:|---:|---:|
| 1 | 281.7 | 242.9 |
| 16 | 681.7 | 398.2 |
| 64 | 1418.6 | 537.4 |

The first run measured 253.5/237.7, 681.9/392.5 and 1415.6/538.8 respectively. The multi-row reduction is clear in this component check; small single-row or end-to-end percentages should not be treated as statistically established from this probe. Later runs used only for contract assertions are not extra performance evidence.

The installed wheel sorts the full vocabulary on the PyTorch path. The local SGLang checkout additionally supports an unseeded max_top_k shortcut; this review does not claim the table compares against that newer optimization.

## Full-model serving

Model: Qwen/Qwen3-Omni-30B-A3B-Instruct, BF16, the checked-in H200 speech pipeline configuration. GPU stages run in separate stage processes colocated on one H200. This is not a one-process pipeline.

Head launch:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_colocated_h200.yaml \
  --host 127.0.0.1 --port 18215 --model-name review2915
```

Control launch uses the identical PR source/configuration and adds:

```bash
--thinker.engine.sampling_backend pytorch
```

This isolates the changed backend and disables the new FlashInfer load-hook action in the control. It is an explicit PyTorch control, **not a claim that a separate latest-main checkout was benchmarked**. Each arm is fully stopped before starting the next.

`probe_serving.py` sends 11 requests per arm: greedy text, seeded filtered text, temperature-only text, four concurrent unseeded filtered text requests, filtered min-p text, seeded filtered speech, unseeded filtered speech and streamed unseeded filtered speech. Speech bytes are checked and hashed, not transcribed. The streamed case requires text and audio chunks followed by DONE without an error event. Normal readiness warmup includes image and audio inputs before the public checks.

Detailed arm outcomes and cleanup are recorded in evidence.json after the runs. The unit and component checks establish the sampling mechanics; a successful model request alone does not prove quality equivalence or disprove F1.

## Boundaries

No full-corpus WER/SIM rerun, low/high-concurrency corpus benchmark, TP>1 execution, disaggregated multi-GPU run or alternate accelerator test was performed. This review checks the CUDA implementation and reads the static non-CUDA selection; it makes no new cross-hardware performance claim. The author-reported corpus results in pr.json are not our measurements.

Raw logs and model outputs remain under `/data/review2915/` on the node. Only small text evidence is saved with this report. Owned servers are stopped when checks finish; the GPU lease and task container remain allocated as requested.
