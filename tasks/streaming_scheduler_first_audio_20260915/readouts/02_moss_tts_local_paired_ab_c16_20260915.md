# Readout 02: MOSS-TTS Local streaming c16, paired A/B, 2026-09-15

Archives: `artifacts/pair-stream-en-c16-20260915T080024Z.tar.gz` (runbook 02) and the earlier single
B boot `artifacts/moss-tts-local-regression-b-stream-en-c16-20260915T070256Z-readout.tar.gz`
(runbook 01, "B0" below).

## Verdict

The streaming scheduler change `442e559b4` does not regress MOSS-TTS Local. A (`1f6b6843e`) and B
(`442e559b4`) ran back to back on GPU 0 with the benchmark at its defaults. B matched or beat A on
every first audio metric in pass 1, and the vocoder's own first audio time is the same in both arms.
The earlier A against PR 1 head gap is inside the spread of identical code measured here. Qwen3-TTS
is not paired.

## Pass 1, the census

| metric | A | B | B - A |
|---|---|---|---|
| req/s | 11.49 | 13.08 | +13.8% |
| audio s/s | 53.665 | 62.75 | +16.9% |
| RTF mean | 0.2735 | 0.2622 | -4.1% |
| first audio mean s | 0.2602 | 0.2437 | -6.3% |
| first audio p50 s | 0.2571 | 0.2419 | -5.9% |
| first audio p95 s | 0.3768 | 0.3512 | -6.8% |
| first audio p99 s | 0.5559 | 0.4771 | -14.2% |
| C50 / C100 | 100 / 100 | 100 / 100 | 0 |
| latency mean s | 1.219 | 1.217 | -0.2% |

## Six runs, two trees

| run | tree | req/s | first audio p50 | p95 | p99 | latency mean | generated audio, total | runaways |
|---|---|---|---|---|---|---|---|---|
| A pass 1 | 1f6b6843e | 11.490 | 0.257 | 0.377 | 0.556 | 1.219 | 5,081.8 s | 3 |
| A pass 2 | 1f6b6843e | 13.283 | 0.248 | 0.370 | 0.501 | 1.189 | 4,854.1 s | 2 |
| B pass 1 | 442e559b4 | 13.080 | 0.242 | 0.351 | 0.477 | 1.217 | 5,219.4 s | 4 |
| B pass 2 | 442e559b4 | 12.160 | 0.256 | 0.380 | 0.534 | 1.242 | 5,034.2 s | 2 |
| B0 pass 1 | 442e559b4 | 12.233 | 0.258 | 0.411 | 0.580 | 1.264 | 5,553.8 s | 5 |
| B0 pass 2 | 442e559b4 | 12.342 | 0.278 | 0.436 | 1.047 | 1.198 | 4,786.7 s | 1 |

Pass 2 runs carry the event recorder. A runaway is an output at the 2048 token cap (163.84 s of
audio). The same tree reads 11.49 and 13.28 req/s on two boots of one arm.

## Vocoder side, pass 2 (`scripts/vocoder_event_timings.py`)

| timing | A | B | B0 |
|---|---|---|---|
| first audio sent - chunk 1 received, p50 / p95 / p99 ms | 32.9 / 61.5 / 73.1 | 32.3 / 61.4 / 72.1 | 35.3 / 111.2 / 198.4 |
| later audio chunks, all, p95 ms | 93.1 | 96.8 | 111.1 |
| receive -> next audio sent, p95 / max ms | 81.7 / 185.2 | 90.8 / 234.5 | 99.9 / 537.7 |
| transport, tts_engine sent -> vocoder receive, p95 ms | 14.7 | 15.5 | 42.7 |

## Where first audio goes, per request (server side, pass 2)

| part | A p50 / p95 / p99 ms | B p50 / p95 / p99 ms | B0 p50 / p95 / p99 ms | share of variance A / B / B0 |
|---|---|---|---|---|
| preprocessing (dispatch to complete) | 102.8 / 204.7 / 356.8 | 105.7 / 228.3 / 362.2 | 110.4 / 218.9 / 759.5 | 78.6 / 78.3 / 75.6 % |
| AR prefill and the first 6 frames | 107.1 / 156.7 / 178.8 | 111.1 / 170.2 / 196.5 | 117.8 / 188.2 / 231.3 | 17.2 / 20.6 / 17.0 % |
| vocoder own | 32.9 / 61.5 / 73.1 | 32.3 / 61.4 / 72.1 | 35.3 / 111.2 / 198.4 | 4.2 / 1.1 / 7.4 % |
| delivery to the coordinator | 0.5 / 1.3 / 1.6 | 0.5 / 1.4 / 1.9 | 0.5 / 1.4 / 1.8 | 0 / 0 / 0 % |
| admission to first audio | 246.2 / 383.8 / 544.7 | 254.0 / 397.0 / 537.4 | 276.4 / 449.1 / 1036.9 | |

The first decode needs 6 frames (1 + 5) against a threshold of 5, so the AR row includes the wait for
the second code chunk.

## Run to run variance, what these archives show

1. Preprocessing carries about 80 percent of the per request first audio variance, and its tail is
   the widest of any part (p99 357 to 760 ms).
2. Sampling is unseeded, so each run generates different audio: 4,787 to 5,554 s in total per run,
   about plus or minus 1.3 s per prompt between two runs of the same tree (p5 to p95), and 1 to 5
   runaways per run, each holding one of the 16 concurrency slots for about 30 s.
3. Other workloads on the host: during B0, GPUs 2 to 5 held about 72 GB each and GPUs 4 and 5 ran at
   99 to 100 percent; B0 has the widest vocoder tail and the widest preprocessing tail. During the
   pair, the other GPUs were idle for A and GPUs 6 and 7 reached 100 percent for B. CPU load was not
   recorded.
4. The p99 over 1,088 requests rests on about 11 requests.

A pass 1 (11.49 req/s, 53.7 audio s/s against 56 to 63 for the other runs) is not explained by these
archives.

## Corrections to readout 01 (box copy)

- It put the added first audio on the vocoder hop (106.4 ms). That segment includes the wait for the
  AR's second code chunk (p50 78 ms); the vocoder's own share is 35 ms at p50, and with no A profile
  no hop delta could be read.
- It called the negative transport p50 clock skew. Matched per chunk, transport is never negative
  (minimum 0.13 ms); the negative comes from how `first_chunk_anatomy.py` pairs events.

## Next

- The preprocessing tail is the largest and most variable part of MOSS-TTS Local first audio; its
  origin is traced next.
- Future A/B runs: CPU load recorded with the GPU logs, arms interleaved A, B, B, A, runaways quoted
  separately.

## Provenance

| arm | head.txt | import path |
|---|---|---|
| A | 1f6b6843ed6accd3e8ee48109445b2e9c23f85a7 | /sgl-workspace/wt/stream-a/sglang_omni/__init__.py |
| B | 442e559b40b5040965ec876650b32da05d31769f | /sgl-workspace/wt/stream-b/sglang_omni/__init__.py |
| B0 | 2654dcecb (runtime tree 442e559b4) | /sgl-workspace/wt/cosyvoice-analysis/sglang_omni/__init__.py |

Server command for both pair arms: `python -m sglang_omni.cli serve --model-path
OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 --port 8000 --host localhost
--tts_engine.engine.max_running_requests 64 --tts_engine.engine.cuda_graph_max_bs 64`, as launched
by the benchmark. GPU 0 for every run.
