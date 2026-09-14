# PR #2123 census table, measured 2026-09-14 (doc 44 session)

A is the #2172 branch at c52eef547 alone, B is the same tree plus this PR's runtime
change (`early_ids.patch`, identical to #2123's diff of `model_runner.py`). Both boots
on the doc 33 protocol, GPU 1, streaming c16, full seed-tts en corpus, pass 2. The
table goes into the PR after #2172 merges and #2123 rebases, when A is main.

### Census, default launch, streaming c16, full seed-tts corpus, H100, one boot per arm

| read | A | B | delta |
| --- | ---: | ---: | ---: |
| req/s | 15.89 | 19.10 | +20.2 percent |
| audio s/s | 66.07 | 79.24 | +19.9 percent |
| RTF mean | 0.2427 | 0.2030 | -16.4 percent |
| latency mean s | 1.001 | 0.834 | -0.167 |
| latency p99 s | 1.680 | 1.360 | -0.320 |
| TTFC mean ms | 99.5 | 103.7 | +4.2 |
| TTFC p50 ms | 94.3 | 97.6 | +3.3 |
| TTFC p95 ms | 141.8 | 155.6 | +13.8 |
| TTFC p99 ms | 284.4 | 275.8 | -8.6 |
| inter chunk mean ms | 111.8 | 90.5 | -21.3 |
| talker step cadence p50 ms | 12.9 | 10.6 | -2.3 |
| preprocessing p50 ms | 23.5 | 29.6 | +6.1 |
| admission to first audio p50 ms | 92.8 | 96.3 | +3.5 |

The first chunk moves 4 ms at 20 percent more throughput on a closed loop client; the
scheduler's own segments (request build to queue 13.0 to 9.0 ms, prefill 18.4 to 13.9
ms) get faster and the segments that share the lock and the GPU with a talker stepping
22 percent more often get slower. The doc 31 gate, first chunk within 10 ms of main's
control at early ids throughput: 103.7 against main's 116.5 ms, 12.8 ms below.

No Nsight rows for this pair: the profiler boots of the session were the default
launch arms only. An open loop pass at A's rate (`--request-rate 15.9`) on B is the
equal throughput first chunk read and is part of the remeasure after the rebase.
