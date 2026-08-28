# Incremental Higgs TTS qualification results

Candidate commit: _
Rust binary SHA-256: _
Host / CPUs: _
GPUs: _
Date: _
Evidence directory: _

Carry-forward baseline: `a1452e143406bfe94aa1fd2b5203ee68b4308e9d`

Carry-forward evidence: router-only, ASR, Qwen3-TTS, and Qwen3-Omni were
completed at the baseline and were not rerun. The incremental candidate has a
different binary hash because it preserves one additional speech metadata
header; routing and body relay mechanics are unchanged.

## Contract and configuration

| Gate | Result |
| --- | --- |
| Rust formatting / focused tests | |
| Benchmark termination-metadata tests | |
| Qualification harness tests | |
| Rust RR/LR config validation | |
| Two-worker Higgs YAML validation | |

## Rust policy screen

Values are `QPS; p95; p99`. Every point uses all 1,088 SeedTTS EN samples.

| Policy | c8 | c16 | c32 | Valid maximum | Selected K |
| --- | --- | --- | --- | --- | ---: |
| Round robin | | | | | |
| Least requests | | | | | |

Selected Rust policy: _
Selected Rust streaming policy: _
Selected Python non-stream policy at the same K: _
Selected Python streaming policy at the same K: _
Conditional policy repeats performed and reason: _
Direct-pair throughput: _

## Final AB/BA/AB comparison

| Mode | Candidate | Rounds | WAVs/trial | QPS | p95 / p99 | Audio TTFP p95 / p99 | Mean / p95 RTF | WER | CPU s/request | Peak RSS | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| WAV | Selected Python | 3 | 1,088 | | | — | | | | | |
| WAV | Selected Rust | 3 | 1,088 | | | — | | | | | |
| PCM stream | Selected Python | 3 | 1,088 | | | | | | | | |
| PCM stream | Selected Rust | 3 | 1,088 | | | | | | | | |

## Termination and failure ownership

| Scope | Candidate/policy | Sample | Finish reason | Completion tokens | Duration | Quality result | Classification |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| | | | | | | | |

Selected-candidate failures: _
Rejected-policy failures: _
Cross-router/model failures: _
Rust-attributable protocol failures: _

## Decision

Correctness: _
Throughput and tails: _
Host CPU/RSS efficiency: _
Worker saturation: _
Final decision: _
Open defects or inconclusive points: _
