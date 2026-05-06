# Qwen3 Omni V1 CI Threshold Calibration Report

This lightweight report records the first Qwen3 Omni V1 calibration run used for the local threshold update.

- Model: `qwen3-omni-v1`
- Repeats: 5
- Local artifact directory: `.tune-runs/20260506T193359Z_qwen3-omni-v1_all_r5`
- Raw logs and JSON outputs are intentionally kept local under `.tune-runs/` and are not included in git.

# CI Threshold Observation Report

## 1. MMMU Accuracy

— 2× NVIDIA H20 from precheck.json, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 60.00 |
| 2 | 50 | 50 | 62.00 |
| 3 | 50 | 50 | 64.00 |
| 4 | 50 | 50 | 56.00 |
| 5 | 50 | 50 | 56.00 |
| **Worst-of-5** | — | — | **56.00** |

## 2. MMMU Speed

— 2× NVIDIA H20 from precheck.json, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 0.683 | 51.00 | 10.885 |
| 2 | 50 | 50 | 0.711 | 52.50 | 10.561 |
| 3 | 50 | 50 | 0.691 | 53.80 | 11.082 |
| 4 | 50 | 50 | 0.743 | 53.40 | 10.009 |
| 5 | 50 | 50 | 0.685 | 54.30 | 10.839 |
| **Worst-of-5** | — | — | **0.683** | **51.00** | **11.082** |

## 3. MMMU TALKER Accuracy

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 10 | 10 | 70.00 |
| 2 | 10 | 10 | 70.00 |
| 3 | 10 | 10 | 70.00 |
| 4 | 10 | 10 | 70.00 |
| 5 | 10 | 10 | 70.00 |
| **Worst-of-5** | — | — | **70.00** |

## 4. MMMU TALKER Wer

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 10 | 10 | 25.00 | 1 |
| 2 | 10 | 10 | 15.09 | 1 |
| 3 | 10 | 10 | 13.36 | 1 |
| 4 | 10 | 10 | 18.45 | 2 |
| 5 | 10 | 10 | 18.65 | 2 |
| **Worst-of-5** | — | — | **25.00** | **2** |

## 5. MMMU TALKER Speed

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.128 | 3.30 | 40.518 | 0.3618 |
| 2 | 10 | 10 | 0.195 | 8.20 | 16.619 | 0.3612 |
| 3 | 10 | 10 | 0.193 | 7.80 | 17.456 | 0.3436 |
| 4 | 10 | 10 | 0.142 | 4.60 | 29.528 | 0.3316 |
| 5 | 10 | 10 | 0.153 | 5.40 | 25.268 | 0.3323 |
| **Worst-of-5** | — | — | **0.128** | **3.30** | **40.518** | **0.3618** |

## 6. MMSU Accuracy

— 2× NVIDIA H20 from precheck.json, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 2000 | 2000 | 70.25 |
| 2 | 2000 | 2000 | 69.55 |
| 3 | 2000 | 2000 | 69.60 |
| 4 | 2000 | 2000 | 70.20 |
| 5 | 2000 | 2000 | 69.60 |
| **Worst-of-5** | — | — | **69.55** |

## 7. MMSU Speed

— 2× NVIDIA H20 from precheck.json, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 2000 | 2000 | 29.399 | 7.60 | 0.271 |
| 2 | 2000 | 2000 | 29.574 | 7.70 | 0.270 |
| 3 | 2000 | 2000 | 29.818 | 7.70 | 0.268 |
| 4 | 2000 | 2000 | 29.949 | 7.70 | 0.266 |
| 5 | 2000 | 2000 | 30.137 | 7.80 | 0.264 |
| **Worst-of-5** | — | — | **29.399** | **7.60** | **0.271** |

## 8. MMSU TALKER Accuracy

— 2× NVIDIA H20 from precheck.json, max_samples=20, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 20 | 20 | 60.00 |
| 2 | 20 | 20 | 60.00 |
| 3 | 20 | 20 | 60.00 |
| 4 | 20 | 20 | 60.00 |
| 5 | 20 | 20 | 50.00 |
| **Worst-of-5** | — | — | **50.00** |

## 9. MMSU TALKER Wer

— 2× NVIDIA H20 from precheck.json, max_samples=20, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 20 | 20 | 2.63 | 0 |
| 2 | 20 | 20 | 3.07 | 3 |
| 3 | 20 | 20 | 3.05 | 1 |
| 4 | 20 | 20 | 3.80 | 0 |
| 5 | 20 | 20 | 3.82 | 2 |
| **Worst-of-5** | — | — | **3.82** | **3** |

## 10. MMSU TALKER Speed

— 2× NVIDIA H20 from precheck.json, max_samples=20, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 20 | 20 | 0.910 | 8.00 | 7.799 | 0.4178 |
| 2 | 20 | 20 | 0.322 | 4.90 | 11.887 | 0.3704 |
| 3 | 20 | 20 | 0.883 | 7.80 | 7.873 | 0.3681 |
| 4 | 20 | 20 | 0.382 | 6.80 | 8.784 | 0.3848 |
| 5 | 20 | 20 | 0.312 | 5.30 | 11.881 | 0.3654 |
| **Worst-of-5** | — | — | **0.312** | **4.90** | **11.887** | **0.4178** |

## 11. TTS Wer

— 2× NVIDIA H20 from precheck.json, max_samples=50, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | N/A | N/A | N/A | N/A |
| 2 | 50 | 50 | 2.14 | 1 |
| 3 | 50 | 50 | 3.01 | 0 |
| 4 | 50 | 50 | 3.55 | 0 |
| 5 | 50 | 50 | 1.42 | 0 |
| **Worst-of-5** | — | — | **3.55** | **1** |

## 12. TTS Speed

— 2× NVIDIA H20 from precheck.json, max_samples=50, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | N/A | N/A | 1.028 | 5.30 | 2.760 | 0.5896 |
| 2 | 50 | 50 | 4.045 | 7.70 | 1.914 | 0.5993 |
| 3 | 50 | 50 | 4.042 | 7.60 | 1.916 | 0.5811 |
| 4 | 50 | 50 | 3.381 | 7.40 | 1.973 | 0.5749 |
| 5 | 50 | 50 | 3.999 | 7.50 | 1.940 | 0.5912 |
| **Worst-of-5** | — | — | **1.028** | **5.30** | **2.760** | **0.5993** |

## 13. VIDEOAMME Accuracy

— 2× NVIDIA H20 from precheck.json, max_samples=30, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 30 | 30 | 66.67 |
| 2 | 30 | 30 | 66.67 |
| 3 | 30 | 30 | 66.67 |
| 4 | 30 | 30 | 66.67 |
| 5 | 30 | 30 | 66.67 |
| **Worst-of-5** | — | — | **66.67** |

## 14. VIDEOAMME Speed

— 2× NVIDIA H20 from precheck.json, max_samples=30, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 30 | 30 | 0.237 | 0.90 | 51.522 |
| 2 | 30 | 30 | 0.237 | 0.90 | 51.453 |
| 3 | 30 | 30 | 0.236 | 0.90 | 51.620 |
| 4 | 30 | 30 | 0.238 | 0.90 | 51.330 |
| 5 | 30 | 30 | 0.237 | 0.90 | 51.450 |
| **Worst-of-5** | — | — | **0.236** | **0.90** | **51.620** |

## 15. VIDEOAMME TALKER Accuracy

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 10 | 10 | 50.00 |
| 2 | 10 | 10 | 50.00 |
| 3 | 10 | 10 | 50.00 |
| 4 | 10 | 10 | 50.00 |
| 5 | 10 | 10 | 50.00 |
| **Worst-of-5** | — | — | **50.00** |

## 16. VIDEOAMME TALKER Wer

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 10 | 10 | 2.33 | 2 |
| 2 | 10 | 10 | 0.28 | 1 |
| 3 | 10 | 10 | 1.52 | 2 |
| 4 | 10 | 10 | 1.09 | 1 |
| 5 | 10 | 10 | 0.53 | 1 |
| **Worst-of-5** | — | — | **2.33** | **2** |

## 17. VIDEOAMME TALKER Speed

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.189 | 1.50 | 31.060 | 2.1748 |
| 2 | 10 | 10 | 0.240 | 1.50 | 29.983 | 3.1497 |
| 3 | 10 | 10 | 0.127 | 1.30 | 35.227 | 1.8913 |
| 4 | 10 | 10 | 0.233 | 1.50 | 30.461 | 7.0128 |
| 5 | 10 | 10 | 0.237 | 1.60 | 29.359 | 2.6414 |
| **Worst-of-5** | — | — | **0.127** | **1.30** | **35.227** | **7.0128** |

## 18. VIDEOMME Accuracy

— 2× NVIDIA H20 from precheck.json, max_samples=30, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 30 | 30 | 56.67 |
| 2 | 30 | 30 | 56.67 |
| 3 | 30 | 30 | 53.33 |
| 4 | 30 | 30 | 56.67 |
| 5 | 30 | 30 | 56.67 |
| **Worst-of-5** | — | — | **53.33** |

## 19. VIDEOMME Speed

— 2× NVIDIA H20 from precheck.json, max_samples=30, concurrency=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 30 | 30 | 0.237 | 2.20 | 51.043 |
| 2 | 30 | 30 | 0.237 | 2.20 | 51.242 |
| 3 | 30 | 30 | 0.229 | 2.10 | 53.480 |
| 4 | 30 | 30 | 0.236 | 2.20 | 51.264 |
| 5 | 30 | 30 | 0.235 | 2.20 | 51.721 |
| **Worst-of-5** | — | — | **0.229** | **2.10** | **53.480** |

## 20. VIDEOMME TALKER Accuracy

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 10 | 10 | 50.00 |
| 2 | 10 | 10 | 50.00 |
| 3 | 10 | 10 | 50.00 |
| 4 | 10 | 10 | 50.00 |
| 5 | 10 | 10 | 50.00 |
| **Worst-of-5** | — | — | **50.00** |

## 21. VIDEOMME TALKER Wer

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 10 | 10 | 1.96 | 0 |
| 2 | 10 | 10 | 3.12 | 0 |
| 3 | 10 | 10 | 0.56 | 0 |
| 4 | 10 | 10 | 1.68 | 0 |
| 5 | 10 | 10 | 2.83 | 0 |
| **Worst-of-5** | — | — | **3.12** | **0** |

## 22. VIDEOMME TALKER Speed

— 2× NVIDIA H20 from precheck.json, max_samples=10, max_tokens=256, concurrency=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Tok/s (aggregate) | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.242 | 1.50 | 29.112 | 3.6606 |
| 2 | 10 | 10 | 0.237 | 1.50 | 29.534 | 3.5555 |
| 3 | 10 | 10 | 0.244 | 1.50 | 28.714 | 3.8953 |
| 4 | 10 | 10 | 0.246 | 1.50 | 28.624 | 3.8892 |
| 5 | 10 | 10 | 0.245 | 1.50 | 28.454 | 3.7650 |
| **Worst-of-5** | — | — | **0.237** | **1.50** | **29.534** | **3.8953** |

## 23. Docs smoke

— 2× NVIDIA H20, docs smoke, 5 runs

| Run | Result |
|-----|--------|
| 1 | PASS |
| 2 | PASS |
| 3 | PASS |
| 4 | PASS |
| 5 | PASS |
| **Worst-of-5** | **PASS** |

## Provenance

- Model: qwen3-omni-v1
- Branch: calibrate-v1-thresholds-cuda-graph-20260506 @ ded5bba8 (dirty) — see `workspace.diff`
- Venv Python: /data/chenyang/.python/omni/bin/python (flag)
- sglang 0.5.8 · torch 2.9.1+cu128
- GPU: 2× NVIDIA H20
- tune-ci-thresholds v0.3.0
- Ran 2026-05-06T19:36:55Z – 2026-05-06T21:34:06Z
