# Qwen3-TTS repetition-tail diagnostic workflow

This workflow belongs to the diagnostic branch only. It is not part of the
minimal upstream repetition-penalty fix.

It generates four arms from the same commit and environment:

| Arm | Public penalty | Qwen owner | SGLang owner | Nominal effective penalty |
|---|---:|---:|---:|---:|
| `sglang_once_p105` | 1.05 | 1.0 | 1.05 | 1.05 |
| `qwen_once_p105` | 1.05 | 1.05 | 1.0 | 1.05 |
| `double_sqrt_p105` | sqrt(1.05) | sqrt(1.05) | sqrt(1.05) | 1.05 |
| `double_p105` | 1.05 | 1.05 | 1.05 | 1.1025 |

Every measured sample receives a stable seed derived from `(panel seed,
sample ID)`. The runner forces benchmark warmup to zero so completion artifacts
have a one-to-one mapping to measured WAVs. Each arm gets a fresh TTS server.
When ASR is enabled, all arms use one shared fresh ASR server at concurrency 1.

## H100 commands

First run a small generation-only mechanical gate:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.run_qwen3_tts_repetition_tail \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output-dir tmp/qwen3_tts_repetition_tail_smoke \
  --seeds 20260823 \
  --max-samples 42 \
  --generation-only
```

Then run the full c16 panels with shared serial ASR:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.run_qwen3_tts_repetition_tail \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output-dir tmp/qwen3_tts_repetition_tail_full \
  --seeds 20260823,20260824,20260825,20260826 \
  --concurrency 16 \
  --max-running-requests 64 \
  --cuda-graph-max-bs 64 \
  --max-new-tokens 2048
```

Use the exact pinned model, dataset, and ASR snapshot paths in place of mutable
Hub names for qualification. Add `--dry-run` to print every command and
diagnostic environment variable without starting a server.

Before the H100 run, execute the focused tests in that container:

```bash
pytest -q \
  tests/unit_test/benchmarks/test_tts_seedtts_benchmark_config.py \
  tests/unit_test/benchmarks/test_qwen3_tts_completion_diagnostics.py \
  tests/unit_test/benchmarks/test_qwen3_tts_repetition_tail_runner.py \
  tests/unit_test/qwen3_tts/test_mask_logit_shaping.py \
  tests/unit_test/qwen3_tts/test_pipeline.py
```

## Artifacts and interpretation

Each seed/arm directory contains the normal benchmark output plus
`completion_diagnostics/*.jsonl`. A completion record includes the public,
semantic, and subtalker seeds; all terminal semantic token IDs; generated codec
codes with reference frames excluded; finish reason; all three penalty values;
and exact sequence hashes.

The runner creates all six pairwise reports under each panel's `comparisons/`
directory. Reports classify the earliest observable differing boundary:

- `semantic_decoder`: semantic token IDs first differ.
- `code_predictor`: semantic IDs match, but generated codec codes differ.
- `vocoder`: semantic IDs and codec codes match, but WAV bytes differ.
- `asr_only`: WAV bytes match, but serial-ASR transcripts differ.
- `identical`: semantic IDs, codec codes, WAV, and available transcript match.

The loader rejects missing records, unmatched seeds/text hashes, and duplicate
records. A duplicate normally means a warmup request was captured and must not
be silently paired with the measured request.

Sample-specific RNG seeds do not make c16 floating-point execution
deterministic. They make each request's RNG stream pairable. The comparison
reports show whether an observed output difference begins in semantic decoding,
the code predictor, the vocoder, or ASR, while the multi-seed WER results remain
the quality and rare-tail evidence.
