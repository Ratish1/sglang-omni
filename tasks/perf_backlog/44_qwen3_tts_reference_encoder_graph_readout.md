# 44. Reference encoder graphs readout, 2026-09-14

Archive `runbook43-results.zip`, unpacked under `artifacts/re-session`. A is upstream
main 69ddc6baa from `tmp/main`, B is `perf/qwen3-tts-reference-encoder-graphs` at
c52eef547 from `tmp/re`, module launch, import paths archived, GPU 1 clear before every
boot, the B gate line present on all three B boots, no `disabled the runner` and no
recompile line, zero failed requests in every pass. The archive has no `A-nsys`
directory; the A rows of section 4 come from the box report and need the files.

## 1. Verdict

The slice does what doc 42 said it would. On the default launch the preprocessing
segment falls from 35.4 to 23.5 ms p50 and its p95 halves, and the first chunk lands
17 ms earlier on the mean and 56 ms earlier at p95, at unchanged throughput (closed loop
c16 is bound by the talker). Under early ids the branch runs 12.6 percent more
requests per second with a first chunk 46 ms earlier than main under the same patch,
and 12.8 ms earlier than main's unpatched control, so the #2123 gate (within 10 ms of
the control at early ids throughput) passes with margin. Quality is inside the bands.

## 2. Default launch, pass 2, streaming c16, full corpus

| read | A main | B branch | delta |
| --- | ---: | ---: | ---: |
| req/s | 15.88 | 15.89 | 0 |
| audio s/s | 66.05 | 66.07 | 0 |
| TTFC mean ms | 116.5 | 99.5 | -17.0 |
| TTFC p50 ms | 107.6 | 94.3 | -13.3 |
| TTFC p95 ms | 198.1 | 141.8 | -56.3 |
| TTFC p99 ms | 279.2 | 284.4 | +5.2 |
| inter chunk mean ms | 109.8 | 111.8 | +2.0 |
| preprocessing p50 / p95 ms | 35.4 / 106.9 | 23.5 / 46.8 | -11.9 / -60.1 |
| admission to first audio p50 / p95 ms | 106.3 / 191.4 | 92.8 / 139.7 | -13.5 / -51.7 |
| prefill p50 ms | 18.3 | 18.4 | |
| first frame to first audio p50 ms | 27.9 | 27.0 | |
| seeded c1 TTFC mean ms | 55.6 | 54.9 | -0.7 |
| seeded c1 RTF mean | 0.0974 | 0.0982 | |

Pass 1 TTFC mean 145.3 against 100.6. Quality on B pass 2: WER 1.05 percent, speaker
similarity 71.35 (doc 36: 1.05 and 71.53). The encoder's codes changed for every
reference (doc 41) and the bands did not move.

## 3. Early ids on both arms, pass 2, streaming c16

| read | A main + patch | B branch + patch | delta |
| --- | ---: | ---: | ---: |
| req/s | 16.96 | 19.10 | +12.6 percent |
| audio s/s | 70.4 | 79.2 | +12.5 percent |
| TTFC mean ms | 149.8 | 103.7 | -46.1 |
| TTFC p50 ms | 144.9 | 97.6 | -47.3 |
| TTFC p95 ms | 241.5 | 155.6 | -85.9 |
| TTFC p99 ms | 304.0 | 275.8 | -28.2 |
| latency mean s | 0.939 | 0.834 | -0.105 |
| RTF mean | 0.229 | 0.203 | |

B under early ids against the unpatched control of section 2: 103.7 against 116.5 ms
TTFC mean, 19.10 against 15.88 req/s. Early ids now costs the branch 4.2 ms of first
chunk for 20 percent more throughput; on main it cost 33 ms.

## 4. Nsight, one full pass capture per arm at c16, both scripts on the same export

| metric | A whole capture | A cohort window | B whole capture (69.8 s) | B cohort window (65.5 s) |
| --- | ---: | ---: | ---: | ---: |
| GR Active | 68.2 | 72.8 | 74.1 | 78.2 |
| SMs Active | 37.9 | 40.6 | 41.7 | 44.1 |
| SM Issue | 12.5 | 13.4 | 14.0 | 14.7 |
| Tensor Active | 2.3 | 2.5 | 2.3 | 2.4 |

The whole capture holds 4.3 s before the cohort (dataset staging, the warmup request)
that the cohort window drops; 78.2 x 65.5 / 69.8 is 73.4, which is the whole capture
mean, so the two scripts agree once they average the same samples. On the cohort
window the branch moves GR Active 72.8 to 78.2, SMs Active 40.6 to 44.1 and SM issue
13.4 to 14.7 (the A files are not in the archive).

## 5. What the comparison settled about the two scripts

Neither measured anything unreal; they averaged different windows of the same
counters. The teammate's window is defined by the benchmark's own log lines, so it is
the same cut on every arm and every machine, and that is the right default for a
whole pass capture. It still holds the cohort's ramp and drain, which a steady state
slice does not, so a number is only comparable with its window stated. Both scripts
now carry the same rules:

- `scripts/nsys_gpu_metrics.py` takes `--bench-log` and cuts the cohort window from
  the log through the export's `localTime` (the clock the log uses), or `--window`
  in session seconds, and prints the share of zero samples so an idle head or tail is
  visible in the mean.
- `patches/compute_sm_window_names_localtime.patch` for the teammate's script: metric
  ids looked up by name in `TARGET_INFO_GPU_METRICS` instead of fixed at 2 to 5 (the
  ids follow the metrics set and the driver), `localTime` instead of `utcTime` (the
  log is local time; on this box they are equal, on another they are not), the sample
  count and metric id printed with each mean, and one connection instead of two per
  metric. Checked on the doc 36 export: with the same window both scripts print the
  same four numbers.

Other things that move these numbers and belong next to them when quoted: the
`--trace` set (CUDA and OS runtime tracing add CPU overhead to the very threads whose
contention we measure; a GPU metrics only capture is the least disturbing), the
sampling frequency (100 Hz and 20 kHz give the same expectation over a minute), the
clock (1985 MHz on every capture here), and the other GPUs' tenants at boot time.

## 6. Tests

7 of 247 failed on the box, all in the test fakes, none in the runtime: the small Mimi
config built its decoder with 512 upsample groups on 16 channels, two engine tests'
fake tokenizers had no `model`, and one test's monkeypatched service factory refused
the new keyword. Fixed in a1f2b6249 on the branch, unrun.

## 7. Next

PR from a1f2b6249 with the section 2 table as the census and section 3 as the early
ids line, then #2123 rebased onto the merged main and remeasured, then #2126.
