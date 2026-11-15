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

**The GIL at c64** (`gil_owners.py` on `main13/attribution/decode_bs64`; main `13364eb45`, which is today's Qwen3-TTS code; under the profiler, 294 steps over 3,061 ms, about 10.4 ms a step):

| Pipeline process thread | Holds, ms a step | Waits, ms a step |
|---|---|---|
| Talker scheduler | 3.01 | 2.75 |
| Vocoder follow-up workers (2) | 1.37 | 3.98 |
| Unnamed threads | 0.78 | 1.31 |
| Vocoder message thread | 0.71 | 0.72 |

- The lock is held 56 % of the window. Other threads hold it 2.86 ms a step, and that is time the talker scheduler waits.
- **The scheduler's own hold**, 2.14 ms a step under `execute`:
  - `prepare_and_forward` 1.34, of which `write_feedback_buffers` 0.57 (a Python loop over rows);
  - `finalize` 0.45;
  - `post_process_outputs` 0.37;
  - `emit_stream_output` 0.33 (64 stream messages);
  - `get_next_batch_to_run` and `process_batch_result` 0.22 each.
- **The vocoder's hold:**
  - `handle_message` 64 times a step at 8.9 µs, 0.57 ms a step;
  - `vocoder.followup[streams=8]` about 0.95 ms a batch, about 0.48 ms of it in `launch_decode_plans`.

First audio (Base, 20 s references):

| | c1 | c64 |
|---|---|---|
| Total | 33 ms | about 390 ms |
| Preprocessing | 18.6 ms | 68.3 ms a request |
| Of it, speaker mel on the CPU | 12.0 ms | 18.1 ms |
| Reference encode | one clip at a time, fp32 | 64 clips in series, about 430 ms |

- The CPU mel is official's bits. A device mel is a numerics change, gated on SIM and WER.
- CustomVoice has no reference to preprocess. Its c1 first audio is host hops: on 10-27, 18.1 to 18.9 ms p50 from preprocessing inbox to the chunk, of which the first vocoder decode took 5.2 ms (a 2 ms batch window, then 3.2 ms of host work).

## Baseline of main `bac76771a`, one H100 (11-15, `served_baseline.py`, SeedTTS EN 1,088 prompts, seed 1234)

| Cell | First audio mean / median / p95, ms | RTF mean | Throughput, req/s | Max underrun p95 |
|---|---|---|---|---|
| Base stream c1 | 18.4 / 19.7 / 22.5 | 0.0656 | 3.70 | 0 |
| Base stream c16 | 30.6 / 27.7 / 43.8 | 0.1034 | 37.6 | 0 |
| Base stream c64 | 51.6 / 39.8 / 133.1 | 0.2255 | 68.0 | 0 |
| CustomVoice stream c1 | 12.0 / 11.8 / 13.2 | 0.0636 | 3.38 | 0 |
| CustomVoice stream c16 | 18.2 / 17.9 / 21.9 | 0.0927 | 36.5 | 0 |
| CustomVoice stream c64 | 28.4 / 21.8 / 121.2 | 0.1790 | 74.7 | 0 |

| Whole-request cell | Latency mean / median / p95, ms | RTF mean | Throughput, req/s |
|---|---|---|---|
| Base c1 | 265 / 259 / 382 | 0.0645 | 3.78 |
| Base c16 | 398 / 388 / 571 | 0.0970 | 40.0 |
| Base c64 | 773 / 762 / 1,139 | 0.1887 | 80.8 |

- SeedTTS references are a few seconds long, so Base's c1 first audio is about 18 ms. The 12 ms CPU speaker mel of the 20 s synthetic references does not appear at this length.

**Pipeshift's bar on main `bac76771a`**: seeds 0, 1 and 2 (`baseline_main/bar/main_s{0,1,2}`), median of seeds, ms. Every request succeeded, with no underrun. Seeds agree within 1 ms except the p95 at 1 req/s (21.9, 29.3, 24.7).

| req/s | 1 | 4 | 8 | 12 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| Audible p95 | 24.7 | 31.5 | 30.9 | 31.1 | 32.4 | 37.0 | 100.1 |
| Pipeshift p95 | 24.6 | 24.6 | 25.7 | 26.0 | 28.4 | 30.7 | 33.7 |
| Audible p50 | 19.5 | 19.5 | 20.5 | 20.4 | 21.3 | 22.6 | 25.4 |
| Pipeshift p50 | 12.7 | 13.6 | 14.2 | 14.6 | 15.8 | 18.1 | 20.2 |
| First byte p95 | 21.5 | 22.7 | 23.5 | 24.2 | 26.3 | 29.2 | 30.8 |
| Leading silence p95 | 0 | 10 | 10 | 10 | 10 | 10 | 80 |

- **At 40 req/s** the leading silence p95 is 80 ms and the first-byte p95 is 30.8 ms. The 24-stream gate is the whole gap.
- **Why Nari's client sees a later first byte than the SeedTTS client** (18.6 against 11.8 ms at c1): its request sends `language: "English"`, so the bootstrap trim applies to Ryan. That holds the first chunk for one more frame, one more talker and predictor step, about 4.5 ms at c1. The SeedTTS client sends no language, so the trim stays off.
- **The audible floor** is prefill, two decode steps and the first vocoder chunk. On top of it (10-27 split):
  - the 2 ms first-chunk wait;
  - 3.2 ms of host work in the first vocoder decode;
  - preprocessing and admission, about 3 ms.
- A faster c1 decode step counts twice in audible first audio.

## A/B on one H100 (11-15, after main's baseline, one run each)

`perf/qwen3-tts-first-chunk-from-queue` `5d18dc578`, served streaming cells (`ab_20261115/firstchunk/served`):

| Cell | First audio mean / median / p95, main → branch | RTF, main → branch |
|---|---|---|
| Base c1 | 18.4 / 19.7 / 22.5 → 17.0 / 18.8 / 20.5 | 0.0656 → 0.0652 |
| Base c16 | 30.6 / 27.7 / 43.8 → 27.4 / 26.6 / 39.9 | 0.1034 → 0.1014 |
| Base c64 | 51.6 / 39.8 / 133.1 → 49.3 / 38.4 / 128.7 | 0.2255 → 0.2190 |
| CustomVoice c1 | 12.0 / 11.8 / 13.2 → 11.0 / 10.9 / 11.5 | 0.0636 → 0.0636 |
| CustomVoice c16 | 18.2 / 17.9 / 21.9 → 17.4 / 17.4 / 20.3 | 0.0927 → 0.0926 |
| CustomVoice c64 | 28.4 / 21.8 / 121.2 → 27.5 / 21.5 / 110.2 | 0.1790 → 0.1818 |

- First audio is lower in every cell, on mean, median and p95. No cell had an underrun.
- The RTF changes are within ±3 % and go both ways. Main's repeat at the end of the queue gives the drift band.

## Order of work

1. Baseline of upstream main `bac76771a` on one H100:
   - `served_baseline.py`: SeedTTS EN, full set, Base streaming and whole requests at c1, c16 and c64, CustomVoice streaming at c1, c16 and c64.
   - Pipeshift's bar with `bar_20261027/queue_baseline.sh`, three seeds.
2. Then the items by measured size in each regime. Each item gets its own branch from main, a design with file:line flow and its SGLang counterpart, a component bench, and the served A/B at c1, c16 and c64, plus the bar when it moves CustomVoice streaming.
3. Bar items first where they are fixes of a named defect: the load-independent opening, admission sized as SGLang sizes it, and the first-chunk wait.
   - Built 11-15 from main, in the A/B queue: `fix/qwen3-tts-opening-at-any-load` `7bbc943b2` (tree `cda960bb0`) and `perf/qwen3-tts-first-chunk-from-queue` `5d18dc578` (tree `16cc6d127`).
   - **Prompt mode:** `fix/qwen3-tts-honor-prompt-mode` `e72d22744` (tree `0fa6e1c70`), queued after the served profile, with Pipeshift's benchmark over three seeds and WER at 1 req/s against main.
     - Official `generate_custom_voice` and `generate_voice_design` default `non_streaming_mode=True` and accept `False`. `generate_voice_clone` defaults to `False`.
     - On main, `CreateSpeechRequest` drops the field, and the builder forces `True` for both named-voice tasks (since #527, kept by #1852).
     - The branch declares the field on the four request types, forwards it, and drops the forcing. The default is unchanged.
     - Drama3 still refuses the field through its `supported_speech_fields`.
   - **Admission is a policy question for the user.** #1449 set the bound to fast-reject past running plus queued (load shedding). SGLang leaves the queue unbounded. On the bar our peak in flight was 47, under the 128 cap, so it causes no failure at today's throughput.
4. c64 decode, by GIL share. Each is a host-only change, so seeded outputs must stay byte-identical. Gate on c16 and c64 RTF and throughput.
   1. The talker scheduler's per-row Python: `write_feedback_buffers` re-gathers rows of last step's batched `output_embeds`, and `post_process_outputs` and `finalize` also loop over rows.
   2. Stream messages from engine to vocoder: one per request a step on both sides. This is `OmniScheduler`'s shared path, so other models must keep main's behaviour.
   3. The vocoder's follow-up planning: `launch_decode_plans` per batch of 8 streams.
   4. Then the predictor at c64 (4.22 ms a step, no custom backend covers it).
