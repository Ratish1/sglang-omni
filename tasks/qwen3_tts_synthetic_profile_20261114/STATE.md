# Qwen3-TTS at c1, c16 and c64 (from 11-15)

Goal: faster first audio and decode at c1, c16 and c64, and ahead of Pipeshift's H100 bar at every load.

- The C++ persistent predictor is set aside. On its own branch it measured 15 % faster than CuTe DSL at 1 row and 2 % slower at 16, and it covers no 64-row decode. Numbers are in `BRANCH_RESULTS.md`.
- Upstream main `bac76771a` has no Qwen3-TTS commit after the profiled `13364eb45`.

## External bar: Pipeshift (read 10-27, restored here 11-15)

The 10-27 notes lived in `tasks/qwen3-tts-perf/STATE.md`, which is gone from the Mac. The text below is restored from the session record. The bar kit is on `feat/omni-step-profiler` under `tasks/qwen3-tts-perf/box/bar_20261027/`.

**Their setup** (`pipeshift.com/all-posts/qwen3-tts-benchmark-h100`):
- Qwen3-TTS 1.7B CustomVoice in one preset voice, over the 1,088 SeedTTS EN sentences.
- Nari Labs' open client, unmodified: full text over HTTP, audio streamed back.
- Poisson arrivals at 1, 4, 8, 12, 20, 30 and 40 req/s; 30 s warm-up and 120 s measured; three seeds; one H100 SXM per run.
- Time to first audio is measured to the first audible sound: from the request's send to the first point where DC-removed RMS over 20 ms windows (10 ms hop) reaches -45 dBFS for two windows in a row.
- A load where more than 5 % of requests fail gets no p95.

**Their p95 time to first audio**, ms, with the median in brackets:

| req/s | Pipeshift | Nari Labs | SGLang-Omni | vLLM-Omni tuned |
|---|---|---|---|---|
| 1 | 24.6 (12.7) | 28.4 (18.7) | 45.7 (32.9) | 154.1 (139.5) |
| 4 | 24.6 (13.6) | 32.0 (21.3) | 53.6 (36.8) | 167.3 (144.6) |
| 8 | 25.7 (14.2) | 34.5 (23.8) | 57.8 (40.5) | 183.2 (153.1) |
| 12 | 26.0 (14.6) | 38.6 (26.5) | 64.5 (43.3) | 203.3 (167.3) |
| 20 | 28.4 (15.8) | 54.3 (33.5) | 121.4 (54.4) | 282.8 (215.6) |
| 30 | 30.7 (18.1) | failed 45.6 % | 925.5 (142.2) | 4,394.0 (2,135.6) |
| 40 | 33.7 (20.2) | failed 87.6 % | failed 49.3 % | failed 14.1 % |

**Our last run of their bar**: main `e9c058462` (10-28), one H100, their client unmodified, three seeds. Audible p95 at 1 to 40 req/s:

| req/s | 1 | 4 | 8 | 12 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| Our p95 | 28.8 | 29.5 | 28.8 | 30.3 | 30.8 | 34.6 | 97.6 |
| Our p50 | 17.0 | 18.3 | 19.2 | 19.5 | 20.4 | 21.7 | 23.5 |

Every request succeeded. Up to 30 req/s we trailed their p95 by 2 to 5 ms, and their p50 by 4 to 5 ms. At 40 req/s, 80 ms of leading silence made the whole gap (first-byte p95 29.4 ms).

**What they name against us, and its status on main `bac76771a`:**

| Item | Where it lives | Status |
|---|---|---|
| Leading silence (80 ms at p95) once more than 24 requests stream | #1901's bootstrap-frame trim, gated by `suppress_bootstrap_max_streams` = 24 (`streaming_vocoder.py:1608`) | Open. Held `fix/qwen3-tts-load-independent-opening` (10-27, base `72b776c9c`) needs rebuilding from main |
| HTTP 503 at 40 req/s | Fixed 64 running and 64 queued (`config.py:37-38`). SGLang sizes the running bound from KV and leaves the queue unbounded | Open |
| Saturation from 20 to 30 req/s | Decode throughput at high batch (ranked below) | Open |
| p50 4 to 5 ms behind at low load | The first-audio path | Open. Held `perf/qwen3-tts-first-chunk-no-wait` measured 1 to 2 ms off first audio (10-27); it waits on its SeedTTS throughput pair |

Open differences from Nari's engine (read 10-27, `nari_reference_20261027.md`):
- The prompt mode. CustomVoice forces `non_streaming_mode = True` (`request_builders.py:489`), which is official's default; Nari honours the client's `false`.
- No host wait on the first two frames.
- One deadline order across talker, predictor and codec.
- FP8 talker gate and up projections.

## Ranked by measured size on main (synthetic profile, H100, `RESULTS_MAIN13.md`)

Decode step:

| ms per step | c1 | c16 | c64 |
|---|---|---|---|
| Step | 4.48 | 5.61 | 9.28 |
| Code predictor | 2.41 | 2.51 | 4.22 |
| Talker | 1.99 | 2.78 | 3.86 |
| Rest, mostly host | 0.08 | 0.32 | 1.20 |

- **Predictor:** CuTe DSL (opt-in) covers 32 rows at most, and nothing custom covers c64.
- **Talker at c1:** about 1.0 of 1.8 ms kernel time is eight small kernels a layer.
- **Host at c64:** `write_feedback_buffers` 1.3 ms mean, `get_next_batch_to_run` 1.2 ms, and 64 vocoder messages a step.
- **Vocoder:** beside vocoder follow-ups, the talker runs +99 % and the predictor +177 % at c64 (10-24 ledger).

First audio (Base, 20 s references):

| | c1 | c64 |
|---|---|---|
| Total | 33 ms | about 390 ms |
| Preprocessing | 18.6 ms | 68.3 ms a request |
| Of it, speaker mel on the CPU | 12.0 ms | 18.1 ms |
| Reference encode | one clip at a time, fp32 | 64 clips in series, about 430 ms |

- The CPU mel is official's bits. A device mel is a numerics change, gated on SIM and WER.
- CustomVoice has no reference to preprocess. Its c1 first audio is host hops: on 10-27, 18.1 to 18.9 ms p50 from preprocessing inbox to the chunk, of which the first vocoder decode took 5.2 ms (a 2 ms batch window, then 3.2 ms of host work).

## Order of work

1. Baseline of upstream main `bac76771a` on one H100:
   - `served_baseline.py`: SeedTTS EN, full set, Base streaming and whole requests at c1, c16 and c64, CustomVoice streaming at c1, c16 and c64.
   - Pipeshift's bar with `bar_20261027/queue_baseline.sh`, three seeds.
2. Then the items by measured size in each regime. Each item gets its own branch from main, a design with file:line flow and its SGLang counterpart, a component bench, and the served A/B at c1, c16 and c64, plus the bar when it moves CustomVoice streaming.
3. Bar items first where they are fixes of a named defect: the load-independent opening, admission sized as SGLang sizes it, and the first-chunk wait.
