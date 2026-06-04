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

## Immediate Comparisons Before `14fd4574`

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

## Branch `feat/higgs-torch-profiler` After `14fd4574`, Async Decode Off

### TTS Speed

| Metric | Value |
|---|---:|
| Concurrency | 8 |
| Completed requests | 1088 |
| Failed requests | 0 |
| Latency mean (s) | 0.876 |
| Latency median (s) | 0.847 |
| Latency p95 (s) | 1.240 |
| Latency p99 (s) | 1.506 |
| RTF mean | 0.2052 |
| RTF median | 0.2003 |
| RTF p95 | 0.2488 |
| RTF p99 | 0.2908 |
| Audio duration mean (s) | 4.355 |
| Audio throughput (s/s) | 39.668 |
| Output throughput (tok/s) | 1055.5 |
| Output tokens/request-s | 147.0 |
| Output tokens mean | 116 |
| Output tokens total | 126081 |
| Prompt tokens mean | 157 |
| Prompt tokens total | 171225 |
| Throughput (req/s) | 9.108 |

### TTS WER

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| WER corpus micro-average | 0.0108 |
| WER corpus micro-average (%) | 1.08 |
| WER per-sample mean | 0.0103 |
| WER per-sample mean (%) | 1.03 |
| WER per-sample median | 0.0000 |
| WER per-sample std | 0.0381 |
| WER per-sample p95 | 0.0833 |
| WER per-sample max | 0.5714 |
| WER per-sample max (%) | 57.14 |
| WER corpus excl >50% | 0.0101 |
| WER corpus excl >50% (%) | 1.01 |
| >50% WER samples | 1 |
| >50% WER samples (%) | 0.1 |
| Latency mean (s) | 0.8761564338235294 |
| Latency p95 (s) | 1.23953 |
| RTF mean | 0.20531388303540954 |
| Throughput (req/s) | 9.108 |
| Audio duration mean (s) | 4.35533088235294 |

### ASR Speed

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| ASR concurrency | 32 |
| ASR latency mean (s) | 1.3007317043533584 |
| ASR latency median (s) | 1.27261909248773 |
| ASR latency p95 (s) | 1.9162601502146546 |
| ASR latency p99 (s) | 2.6553611097251997 |
| ASR RTF mean | 0.3162246125376549 |
| ASR RTF median | 0.2987183757353722 |
| ASR total time (s) | 44.30550964199938 |
| ASR latency sum (s) | 1415.1960943364538 |
| ASR throughput (samples/s) | 24.55676525992675 |
| Audio processed (s) | 4738.6 |

## Branch `feat/higgs-torch-profiler` After `14fd4574`, Async Decode On

### TTS Speed

| Metric | Value |
|---|---:|
| Concurrency | 8 |
| Completed requests | 1088 |
| Failed requests | 0 |
| Latency mean (s) | 0.894 |
| Latency median (s) | 0.857 |
| Latency p95 (s) | 1.274 |
| Latency p99 (s) | 1.496 |
| RTF mean | 0.2092 |
| RTF median | 0.2032 |
| RTF p95 | 0.2582 |
| RTF p99 | 0.3032 |
| Audio duration mean (s) | 4.359 |
| Audio throughput (s/s) | 38.929 |
| Output throughput (tok/s) | 1035.8 |
| Output tokens/request-s | 145.1 |
| Output tokens mean | 116 |
| Output tokens total | 126173 |
| Prompt tokens mean | 157 |
| Prompt tokens total | 171225 |
| Throughput (req/s) | 8.931 |

### TTS WER

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| WER corpus micro-average | 0.0129 |
| WER corpus micro-average (%) | 1.29 |
| WER per-sample mean | 0.0126 |
| WER per-sample mean (%) | 1.26 |
| WER per-sample median | 0.0000 |
| WER per-sample std | 0.0403 |
| WER per-sample p95 | 0.0909 |
| WER per-sample max | 0.4286 |
| WER per-sample max (%) | 42.86 |
| WER corpus excl >50% | 0.0129 |
| WER corpus excl >50% (%) | 1.29 |
| >50% WER samples | 0 |
| >50% WER samples (%) | 0.0 |
| Latency mean (s) | 0.8941730698529412 |
| Latency p95 (s) | 1.2743049999999998 |
| RTF mean | 0.20930354824335143 |
| Throughput (req/s) | 8.931 |
| Audio duration mean (s) | 4.358713235294118 |

### ASR Speed

| Metric | Value |
|---|---:|
| Evaluated / total | 1088 / 1088 |
| Skipped | 0 |
| ASR concurrency | 32 |
| ASR latency mean (s) | 1.3086088392907078 |
| ASR latency median (s) | 1.2882736469618976 |
| ASR latency p95 (s) | 1.9726151829469007 |
| ASR latency p99 (s) | 2.6911079552187585 |
| ASR RTF mean | 0.317307478194729 |
| ASR RTF median | 0.30140533514137025 |
| ASR total time (s) | 44.569917442975566 |
| ASR latency sum (s) | 1423.7664171482902 |
| ASR throughput (samples/s) | 24.411084032004958 |
| Audio processed (s) | 4742.28 |

## Immediate Comparisons After `14fd4574`

### Async Decode Off: Main vs Branch

| Metric | Main off | Branch off | Branch delta |
|---|---:|---:|---:|
| Completed requests | 1088 | 1088 | 0 |
| Failed requests | 0 | 0 | 0 |
| Latency mean (s) | 0.893 | 0.876 | -0.017 |
| Latency median (s) | 0.852 | 0.847 | -0.005 |
| Latency p95 (s) | 1.256 | 1.240 | -0.016 |
| Latency p99 (s) | 1.462 | 1.506 | +0.044 |
| RTF mean | 0.2059 | 0.2052 | -0.0007 |
| RTF median | 0.2004 | 0.2003 | -0.0001 |
| RTF p95 | 0.2530 | 0.2488 | -0.0042 |
| RTF p99 | 0.3091 | 0.2908 | -0.0183 |
| Audio duration mean (s) | 4.439 | 4.355 | -0.084 |
| Audio throughput (s/s) | 39.682 | 39.668 | -0.014 |
| Output throughput (tok/s) | 1054.6 | 1055.5 | +0.9 |
| Output tokens/request-s | 146.4 | 147.0 | +0.6 |
| Output tokens mean | 118 | 116 | -2 |
| Output tokens total | 128352 | 126081 | -2271 |
| Prompt tokens mean | 157 | 157 | 0 |
| Prompt tokens total | 171225 | 171225 | 0 |
| Throughput (req/s) | 8.940 | 9.108 | +0.168 |
| WER corpus (%) | 1.16 | 1.08 | -0.08 |
| WER per-sample mean (%) | 1.12 | 1.03 | -0.09 |
| WER per-sample median | 0.0000 | 0.0000 | 0.0000 |
| WER per-sample std | 0.0383 | 0.0381 | -0.0002 |
| WER per-sample p95 | 0.0909 | 0.0833 | -0.0076 |
| WER per-sample max (%) | 42.86 | 57.14 | +14.28 |
| WER corpus excl >50% (%) | 1.16 | 1.01 | -0.15 |
| >50% WER samples | 0 | 1 | +1 |
| ASR latency mean (s) | 1.3005136454489463 | 1.3007317043533584 | +0.0002180589044121 |
| ASR latency median (s) | 1.2749539801152423 | 1.27261909248773 | -0.0023348876275123 |
| ASR latency p95 (s) | 1.9546896129380893 | 1.9162601502146546 | -0.0384294627234347 |
| ASR latency p99 (s) | 2.669341992929112 | 2.6553611097251997 | -0.0139808832039123 |
| ASR RTF mean | 0.31541548159032795 | 0.3162246125376549 | +0.0008091309473269 |
| ASR RTF median | 0.29736808287238603 | 0.2987183757353722 | +0.0013502928629862 |
| ASR total time (s) | 44.29596303612925 | 44.30550964199938 | +0.00954660587013 |
| ASR latency sum (s) | 1414.9588462484535 | 1415.1960943364538 | +0.2372480880003 |
| ASR throughput (samples/s) | 24.562057700666564 | 24.55676525992675 | -0.005292440739814 |
| Audio processed (s) | 4829.44 | 4738.6 | -90.84 |

### Async Decode On: Main vs Branch

| Metric | Main on | Branch on | Branch delta |
|---|---:|---:|---:|
| Completed requests | 1088 | 1088 | 0 |
| Failed requests | 0 | 0 | 0 |
| Latency mean (s) | 0.891 | 0.894 | +0.003 |
| Latency median (s) | 0.850 | 0.857 | +0.007 |
| Latency p95 (s) | 1.263 | 1.274 | +0.011 |
| Latency p99 (s) | 1.483 | 1.496 | +0.013 |
| RTF mean | 0.2073 | 0.2092 | +0.0019 |
| RTF median | 0.2021 | 0.2032 | +0.0011 |
| RTF p95 | 0.2563 | 0.2582 | +0.0019 |
| RTF p99 | 0.3051 | 0.3032 | -0.0019 |
| Audio duration mean (s) | 4.417 | 4.359 | -0.058 |
| Audio throughput (s/s) | 37.100 | 38.929 | +1.829 |
| Output throughput (tok/s) | 986.3 | 1035.8 | +49.5 |
| Output tokens/request-s | 146.4 | 145.1 | -1.3 |
| Output tokens mean | 117 | 116 | -1 |
| Output tokens total | 127751 | 126173 | -1578 |
| Prompt tokens mean | 157 | 157 | 0 |
| Prompt tokens total | 171225 | 171225 | 0 |
| Throughput (req/s) | 8.400 | 8.931 | +0.531 |
| WER corpus (%) | 1.16 | 1.29 | +0.13 |
| WER per-sample mean (%) | 1.16 | 1.26 | +0.10 |
| WER per-sample median | 0.0000 | 0.0000 | 0.0000 |
| WER per-sample std | 0.0421 | 0.0403 | -0.0018 |
| WER per-sample p95 | 0.0909 | 0.0909 | 0.0000 |
| WER per-sample max (%) | 50.00 | 42.86 | -7.14 |
| WER corpus excl >50% (%) | 1.16 | 1.29 | +0.13 |
| >50% WER samples | 0 | 0 | 0 |
| ASR latency mean (s) | 1.299247111148468 | 1.3086088392907078 | +0.0093617281422398 |
| ASR latency median (s) | 1.2803384035360068 | 1.2882736469618976 | +0.0079352434258908 |
| ASR latency p95 (s) | 1.941616602079004 | 1.9726151829469007 | +0.0309985808678967 |
| ASR latency p99 (s) | 2.6448816516809166 | 2.6911079552187585 | +0.0462263035378419 |
| ASR RTF mean | 0.31602228623921363 | 0.317307478194729 | +0.0012851919555154 |
| ASR RTF median | 0.3016449776278489 | 0.30140533514137025 | -0.0002396424864787 |
| ASR total time (s) | 44.28446609410457 | 44.569917442975566 | +0.285451348870996 |
| ASR latency sum (s) | 1413.5808569295332 | 1423.7664171482902 | +10.185560218757 |
| ASR throughput (samples/s) | 24.568434396115286 | 24.411084032004958 | -0.157350364110328 |
| Audio processed (s) | 4805.4 | 4742.28 | -63.12 |
