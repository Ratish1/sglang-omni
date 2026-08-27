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

Full-corpus Rust screen (`samples/s; p95; p99`):

| Policy | c32 | c64 | c96 | Valid maximum | Selected K |
| --- | --- | --- | --- | --- | ---: |
| Round robin | | | | | |
| Least requests | | | | | |

Selected Rust policy: _
Selected Python policy: _
Selected common concurrency: _
Direct-pair throughput: _

| Mode | Candidate | Rounds | Samples | samples/s | p95 / p99 | TTFT | WER | 429 / 5xx | CPU s/request | Peak RSS | Both workers | In-flight zero | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| Non-stream | Selected Python | 3 | | | | — | | | | | | — | |
| Non-stream | Selected Rust | 3 | | | | — | | | | | | | |
| SSE | Selected Python | 3 | | | | | | | | | | — | |
| SSE | Selected Rust | 3 | | | | | | | | | | | |

## Standalone TTS — Qwen/Qwen3-TTS-12Hz-1.7B-Base

Full-corpus Rust screen (`QPS; p95; p99`):

| Policy | c16 | c32 | c64 | Valid maximum | Selected K |
| --- | --- | --- | --- | --- | ---: |
| Round robin | | | | | |
| Least requests | | | | | |

Selected Rust policy: _
Selected Python policy: _
Selected common concurrency: _
Direct-pair throughput: _

| Mode | Candidate | Rounds | WAVs | QPS | p95 / p99 | TTFT | Mean / p95 RTF | Audio valid | 429 / 5xx | CPU s/request | Peak RSS | Both workers | In-flight zero | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| WAV | Selected Python | 3 | | | | — | | | | | | | — | |
| WAV | Selected Rust | 3 | | | | — | | | | | | | | |
| PCM stream | Selected Python | 3 | | | | | | | | | | | — | |
| PCM stream | Selected Rust | 3 | | | | | | | | | | | | |

## Omni text/image — Qwen3-Omni FP8

Full-corpus Rust screen (`QPS; p95; p99`):

| Policy | c8 | c16 | c32 | Valid maximum | Selected K |
| --- | --- | --- | --- | --- | ---: |
| Round robin | | | | | |
| Least requests | | | | | |

Selected Rust policy: _
Selected Python policy: _
Selected common concurrency: _

| Candidate | Rounds | Completed | QPS | p95 / p99 | Accuracy | 429 / 5xx | CPU s/request | Peak RSS | Both workers | In-flight zero | Result |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- |
| Selected Python | 3 | | | | | | | | | — | |
| Selected Rust | 3 | | | | | | | | | | |

Direct-pair throughput: _

## Omni audio output — Qwen3-Omni BF16

Full-corpus Rust screen (`QPS; p95; p99`):

| Policy | c8 | c16 | c32 | Valid maximum | Selected K |
| --- | --- | --- | --- | --- | ---: |
| Round robin | | | | | |
| Least requests | | | | | |

Selected Rust policy: _
Selected Python policy: _
Selected common concurrency: _

| Workload | Candidate | Rounds | Completed | QPS | p95 / p99 | TTFT / audio TTFP | Mean RTF | Audio valid | 429 / 5xx | CPU s/request | Peak RSS | Both workers | In-flight zero | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| Full SeedTTS | Selected Python | 3 | | | | — | | | | | | | — | |
| Full SeedTTS | Selected Rust | 3 | | | | — | | | | | | | | |
| Full SeedTTS stream | Selected Python | 3 | | | | | | | | | | | — | |
| Full SeedTTS stream | Selected Rust | 3 | | | | | | | | | | | | |

Direct-pair throughput: _

## Decision

Correctness: _
Router-only proxy ceiling: _
H100 throughput/tails: _
Host CPU/RSS efficiency: _
Final decision: _
Open defects or inconclusive points: _
