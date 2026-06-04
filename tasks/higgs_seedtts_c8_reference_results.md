# Higgs SeedTTS c=8 Reference Results

Date captured from local terminal output: 2026-06-04.

Model: `boson-sglang/higgs-audio-v3-TTS-4B-grpo05200410999`
Dataset: `zhaochenyang20/seed-tts-eval-arrow`
Language: `en`
Generation mode: non-streaming
Samples: 1088
ASR model: `Qwen/Qwen3-ASR-1.7B`
ASR concurrency: 32

## Main Branch, Async Decode On

### TTS Speed

| Metric | Value |
|---|---:|
| Concurrency | 8 |
| Completed requests | 1088 |
| Failed requests | 0 |
| Latency mean (s) | 0.891 |
| Latency median (s) | 0.850 |
| Latency p95 (s) | 1.263 |
| Latency p99 (s) | 1.483 |
| RTF mean | 0.2073 |
| RTF median | 0.2021 |
| RTF p95 | 0.2563 |
| RTF p99 | 0.3051 |
| Audio duration mean (s) | 4.417 |
| Audio throughput (s/s) | 37.100 |
| Output throughput (tok/s) | 986.3 |
| Output tokens/request-s | 146.4 |
| Output tokens mean | 117 |
| Output tokens total | 127751 |
| Prompt tokens mean | 157 |
| Prompt tokens total | 171225 |
| Throughput (req/s) | 8.400 |

### TTS WER

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| WER corpus micro-average | 0.0116 |
| WER corpus micro-average (%) | 1.16 |
| WER per-sample mean | 0.0116 |
| WER per-sample mean (%) | 1.16 |
| WER per-sample median | 0.0000 |
| WER per-sample std | 0.0421 |
| WER per-sample p95 | 0.0909 |
| WER per-sample max | 0.5000 |
| WER per-sample max (%) | 50.00 |
| WER corpus excl >50% | 0.0116 |
| WER corpus excl >50% (%) | 1.16 |
| >50% WER samples | 0 |
| >50% WER samples (%) | 0.0 |
| Latency mean (s) | 0.8912770220588235 |
| Latency p95 (s) | 1.2628499999999996 |
| RTF mean | 0.20737307969141647 |
| Throughput (req/s) | 8.400 |
| Audio duration mean (s) | 4.416727941176471 |

### ASR Speed

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| ASR concurrency | 32 |
| ASR latency mean (s) | 1.299247111148468 |
| ASR latency median (s) | 1.2803384035360068 |
| ASR latency p95 (s) | 1.941616602079004 |
| ASR latency p99 (s) | 2.6448816516809166 |
| ASR RTF mean | 0.31602228623921363 |
| ASR RTF median | 0.3016449776278489 |
| ASR total time (s) | 44.28446609410457 |
| ASR latency sum (s) | 1413.5808569295332 |
| ASR throughput (samples/s) | 24.568434396115286 |
| Audio processed (s) | 4805.4 |

## Main Branch, Async Decode Off

### TTS Speed

| Metric | Value |
|---|---:|
| Concurrency | 8 |
| Completed requests | 1088 |
| Failed requests | 0 |
| Latency mean (s) | 0.893 |
| Latency median (s) | 0.852 |
| Latency p95 (s) | 1.256 |
| Latency p99 (s) | 1.462 |
| RTF mean | 0.2059 |
| RTF median | 0.2004 |
| RTF p95 | 0.2530 |
| RTF p99 | 0.3091 |
| Audio duration mean (s) | 4.439 |
| Audio throughput (s/s) | 39.682 |
| Output throughput (tok/s) | 1054.6 |
| Output tokens/request-s | 146.4 |
| Output tokens mean | 118 |
| Output tokens total | 128352 |
| Prompt tokens mean | 157 |
| Prompt tokens total | 171225 |
| Throughput (req/s) | 8.940 |

### TTS WER

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| WER corpus micro-average | 0.0116 |
| WER corpus micro-average (%) | 1.16 |
| WER per-sample mean | 0.0112 |
| WER per-sample mean (%) | 1.12 |
| WER per-sample median | 0.0000 |
| WER per-sample std | 0.0383 |
| WER per-sample p95 | 0.0909 |
| WER per-sample max | 0.4286 |
| WER per-sample max (%) | 42.86 |
| WER corpus excl >50% | 0.0116 |
| WER corpus excl >50% (%) | 1.16 |
| >50% WER samples | 0 |
| >50% WER samples (%) | 0.0 |
| Latency mean (s) | 0.8929508272058824 |
| Latency p95 (s) | 1.255525 |
| RTF mean | 0.20601984075938892 |
| Throughput (req/s) | 8.940 |
| Audio duration mean (s) | 4.438823529411765 |

### ASR Speed

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| ASR concurrency | 32 |
| ASR latency mean (s) | 1.3005136454489463 |
| ASR latency median (s) | 1.2749539801152423 |
| ASR latency p95 (s) | 1.9546896129380893 |
| ASR latency p99 (s) | 2.669341992929112 |
| ASR RTF mean | 0.31541548159032795 |
| ASR RTF median | 0.29736808287238603 |
| ASR total time (s) | 44.29596303612925 |
| ASR latency sum (s) | 1414.9588462484535 |
| ASR throughput (samples/s) | 24.562057700666564 |
| Audio processed (s) | 4829.44 |

## Branch `feat/higgs-torch-profiler`, Async Decode On

### TTS Speed

| Metric | Value |
|---|---:|
| Concurrency | 8 |
| Completed requests | 1088 |
| Failed requests | 0 |
| Latency mean (s) | 0.895 |
| Latency median (s) | 0.865 |
| Latency p95 (s) | 1.259 |
| Latency p99 (s) | 1.521 |
| RTF mean | 0.2097 |
| RTF median | 0.2035 |
| RTF p95 | 0.2629 |
| RTF p99 | 0.3057 |
| Audio duration mean (s) | 4.361 |
| Audio throughput (s/s) | 38.882 |
| Output throughput (tok/s) | 1034.5 |
| Output tokens/request-s | 144.8 |
| Output tokens mean | 116 |
| Output tokens total | 126234 |
| Prompt tokens mean | 157 |
| Prompt tokens total | 171225 |
| Throughput (req/s) | 8.916 |

### TTS WER

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| WER corpus micro-average | 0.0104 |
| WER corpus micro-average (%) | 1.04 |
| WER per-sample mean | 0.0100 |
| WER per-sample mean (%) | 1.00 |
| WER per-sample median | 0.0000 |
| WER per-sample std | 0.0349 |
| WER per-sample p95 | 0.0909 |
| WER per-sample max | 0.4286 |
| WER per-sample max (%) | 42.86 |
| WER corpus excl >50% | 0.0104 |
| WER corpus excl >50% (%) | 1.04 |
| >50% WER samples | 0 |
| >50% WER samples (%) | 0.0 |
| Latency mean (s) | 0.8954435661764706 |
| Latency p95 (s) | 1.2585649999999997 |
| RTF mean | 0.20979780960469685 |
| Throughput (req/s) | 8.916 |
| Audio duration mean (s) | 4.360955882352941 |

### ASR Speed

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| ASR concurrency | 32 |
| ASR latency mean (s) | 1.3143654215979468 |
| ASR latency median (s) | 1.296523732598871 |
| ASR latency p95 (s) | 1.9639598179957825 |
| ASR latency p99 (s) | 2.8629346954566426 |
| ASR RTF mean | 0.3195485737776035 |
| ASR RTF median | 0.2998341248133155 |
| ASR total time (s) | 44.76906260987744 |
| ASR latency sum (s) | 1430.0295786985662 |
| ASR throughput (samples/s) | 24.302496781783265 |
| Audio processed (s) | 4744.72 |

## Branch `feat/higgs-torch-profiler`, Async Decode Off

### TTS Speed

| Metric | Value |
|---|---:|
| Concurrency | 8 |
| Completed requests | 1088 |
| Failed requests | 0 |
| Latency mean (s) | 1.046 |
| Latency median (s) | 0.857 |
| Latency p95 (s) | 1.281 |
| Latency p99 (s) | 2.379 |
| RTF mean | 0.2499 |
| RTF median | 0.2032 |
| RTF p95 | 0.2635 |
| RTF p99 | 0.3552 |
| Audio duration mean (s) | 4.400 |
| Audio throughput (s/s) | 33.594 |
| Output throughput (tok/s) | 893.3 |
| Output tokens/request-s | 125.2 |
| Output tokens mean | 117 |
| Output tokens total | 127286 |
| Prompt tokens mean | 157 |
| Prompt tokens total | 171225 |
| Throughput (req/s) | 7.636 |

### TTS WER

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| WER corpus micro-average | 0.0101 |
| WER corpus micro-average (%) | 1.01 |
| WER per-sample mean | 0.0099 |
| WER per-sample mean (%) | 0.99 |
| WER per-sample median | 0.0000 |
| WER per-sample std | 0.0363 |
| WER per-sample p95 | 0.0909 |
| WER per-sample max | 0.4286 |
| WER per-sample max (%) | 42.86 |
| WER corpus excl >50% | 0.0101 |
| WER corpus excl >50% (%) | 1.01 |
| >50% WER samples | 0 |
| >50% WER samples (%) | 0.0 |
| Latency mean (s) | 1.0460913602941178 |
| Latency p95 (s) | 1.2812149999999995 |
| RTF mean | 0.24999839385998204 |
| Throughput (req/s) | 7.636 |
| Audio duration mean (s) | 4.399632352941175 |

### ASR Speed

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| ASR concurrency | 32 |
| ASR latency mean (s) | 1.3445571844191255 |
| ASR latency median (s) | 1.333256860030815 |
| ASR latency p95 (s) | 2.0002993871807093 |
| ASR latency p99 (s) | 2.7431587776378725 |
| ASR RTF mean | 0.326753792483991 |
| ASR RTF median | 0.3122102430286813 |
| ASR total time (s) | 45.79461164306849 |
| ASR latency sum (s) | 1462.8782166480087 |
| ASR throughput (samples/s) | 23.75825366704863 |
| Audio processed (s) | 4786.8 |

## Immediate Comparisons

### Async Decode Off: Main vs Branch

| Metric | Main off | Branch off | Branch delta |
|---|---:|---:|---:|
| WER corpus (%) | 1.16 | 1.01 | -0.15 |
| Latency mean (s) | 0.893 | 1.046 | +0.153 |
| Latency p95 (s) | 1.256 | 1.281 | +0.025 |
| Latency p99 (s) | 1.462 | 2.379 | +0.917 |
| RTF mean | 0.2059 | 0.2499 | +0.0440 |
| Audio duration mean (s) | 4.439 | 4.400 | -0.039 |
| Audio throughput (s/s) | 39.682 | 33.594 | -6.088 |
| Output throughput (tok/s) | 1054.6 | 893.3 | -161.3 |
| Output tokens total | 128352 | 127286 | -1066 |
| Throughput (req/s) | 8.940 | 7.636 | -1.304 |

### Async Decode On: Main vs Branch

| Metric | Main on | Branch on | Branch delta |
|---|---:|---:|---:|
| WER corpus (%) | 1.16 | 1.04 | -0.12 |
| Latency mean (s) | 0.891 | 0.895 | +0.004 |
| Latency p95 (s) | 1.263 | 1.259 | -0.004 |
| Latency p99 (s) | 1.483 | 1.521 | +0.038 |
| RTF mean | 0.2073 | 0.2097 | +0.0024 |
| Audio duration mean (s) | 4.417 | 4.361 | -0.056 |
| Audio throughput (s/s) | 37.100 | 38.882 | +1.782 |
| Output throughput (tok/s) | 986.3 | 1034.5 | +48.2 |
| Output tokens total | 127751 | 126234 | -1517 |
| Throughput (req/s) | 8.400 | 8.916 | +0.516 |
