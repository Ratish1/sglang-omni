# Rust router qualification results

Candidate commit: _
Rust binary: _
Host / CPUs: _
GPUs: _
Date: _

## Contract proof

| Gate | Result |
| --- | --- |
| Rust fmt / Clippy / all-target tests | |
| Package unit tests | |
| Rust config validation | |
| Launcher YAML validation | |

## Router-only

| Scenario | Concurrency | Direct req/s | Python RR req/s | Rust RR req/s | Rust/Python | Python CPU s | Rust CPU s | CPU efficiency ratio | Python p99 | Rust p99 | Failures | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Small JSON | | | | | | | | | | | | |
| 1 MiB upload | | | | | | | | | | | | |
| SSE relay | | | | | | | | | | | | |

| Sentinel | Candidate A | Candidate B | Throughput delta | p99 delta | Result |
| --- | --- | --- | ---: | ---: | --- |
| Variable duration | Rust round robin | Rust least requests | | | |
| Tracing filter median | Rust `error` | Rust `info` | | | |

## ASR — Qwen/Qwen3-ASR-1.7B

Selected knee: _
Direct-pair throughput: _

| Mode | Candidate | Rounds | Samples | samples/s | p95 / p99 | TTFT | WER | 429 / 5xx | CPU s/request | Peak RSS | Both workers | In-flight zero | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| Non-stream | Python RR | 3 | | | | — | | | | | | — | |
| Non-stream | Rust RR | 3 | | | | — | | | | | | | |
| SSE | Python RR | 3 | | | | | | | | | | — | |
| SSE | Rust RR | 3 | | | | | | | | | | | |
| Non-stream | Python least request | 1 | | | | — | | | | | | — | informational |

## Standalone TTS — Qwen/Qwen3-TTS-12Hz-1.7B-Base

Selected knee: _
Direct-pair throughput: _

| Mode | Candidate | Rounds | WAVs | QPS | p95 / p99 | TTFT | Mean / p95 RTF | Audio valid | 429 / 5xx | CPU s/request | Peak RSS | Both workers | In-flight zero | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| WAV | Python RR | 3 | | | | — | | | | | | | — | |
| WAV | Rust RR | 3 | | | | — | | | | | | | | |
| PCM stream | Python RR | 3 | | | | | | | | | | | — | |
| PCM stream | Rust RR | 3 | | | | | | | | | | | | |
| WAV | Python least request | 1 | | | | — | | | | | | | — | informational |

## Omni text/image — Qwen3-Omni FP8

| Concurrency | Candidate | Rounds | Completed | QPS | p95 / p99 | Accuracy | 429 / 5xx | CPU s/request | Peak RSS | Both workers | In-flight zero | Result |
| ---: | --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| 1 | Python RR | 1 | | | | | | | | | — | |
| 1 | Rust RR | 1 | | | | | | | | | | |
| 16 | Python RR | 3 | | | | | | | | | — | |
| 16 | Rust RR | 3 | | | | | | | | | | |
| 16 | Python least request | 1 | | | | | | | | | — | informational |

Direct-pair throughput at c16: _

## Omni audio output — Qwen3-Omni BF16

| Workload | Candidate | Rounds | Completed | QPS | p95 / p99 | TTFT / audio TTFP | Mean RTF | Audio valid | 429 / 5xx | CPU s/request | Peak RSS | Both workers | In-flight zero | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| SeedTTS-50 c16 | Python RR | 3 | | | | — | | | | | | | — | |
| SeedTTS-50 c16 | Rust RR | 3 | | | | — | | | | | | | | |
| Streaming TTFT | Python RR | 3 | | | | | | | | | | | — | |
| Streaming TTFT | Rust RR | 3 | | | | | | | | | | | | |
| SeedTTS-50 c16 | Python least request | 1 | | | | — | | | | | | | — | informational |

Direct-pair throughput at c16: _

## Decision

Correctness: _
Router-only proxy ceiling: _
H100 throughput/tails: _
Host CPU/RSS efficiency: _
Final decision: _
Open defects or inconclusive points: _
