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

**Served SeedTTS on main `bac76771a`, Base streaming** (`served_profile.py --mode attribution`, `served_profile_main/attribution`, 3 s windows during steady load). The profiler inflates Python, so read shares, not absolute times.

| Per talker step | c1 | c16 | c64 |
|---|---|---|---|
| Step (window / steps) | 5.3 ms (580) | 7.3 ms (419) | 13.0 ms (112, of which 46 prefills) |
| Scheduler holds the GIL | 1.44 ms | 2.33 ms | 3.68 ms |
| Scheduler waits for it | 0.23 ms | 1.51 ms | 4.36 ms |
| Other pipeline threads hold it | 0.29 ms | 1.75 ms | 4.61 ms |

At c64 the other holders, per step:
- vocoder follow-up and initial workers 1.79 ms;
- preprocessing pool 1.24 ms, that is 1.6 to 1.9 ms per request in `preprocess_qwen3_tts_payload`, at 68 req/s;
- unnamed threads 1.05 ms;
- vocoder message thread 0.52 ms (4,186 messages at 11 µs).

Other readings:
- **Talker stream at c64:** its kernels total 826 ms over 112 steps, about 7.4 ms a step against a 13.0 ms step, so it waits on the host about 40 % of the time.
- **The fp32 reference encoder** adds 170 ms of kernels on its own stream in the c64 window.
- **At c1** the scheduler's own Python is the whole host cost. The first vocoder chunk holds the GIL 0.65 ms (`vocoder.initial[streams=1]`), and a one-stream follow-up 0.3 to 0.37 ms.

**GIL owners by function, served c64** (`served_profile.py --mode pyspy`, `py-spy --gil --threads --subprocesses`, a 30 s window inside a three-pass c64 load; `pyspy_digest.py`, `served_profile_main/pyspy/base_stream_c64.pyspy.txt`). py-spy samples about 46 Hz on the pipeline process's 251 threads, so read shares.

Pipeline process, 1,138 samples holding the lock:

| Holder | Share |
|---|---|
| Talker scheduler | 50 % |
| Vocoder follow-up and initial workers and the vocoder scheduler | 27 % |
| Preprocessing pool | 17 % |
| Main thread stream routing (`drain_outbox`, `route_stream_run`, ZMQ sends) | 5 % |

Within the scheduler (inclusive, of 564 samples):
- **`execute` 40 %:**
  - forward host side 16 %;
  - `before_decode` 12 %;
  - `sample_next_token_ids` 8 % (SGLang's `top_k_top_p_min_p_sampling_from_probs_torch`);
  - finalize 6 %.
- **`process_batch_result` 15 %:**
  - `emit_stream_output` 7 %, of which `stream_output_builder:2195`, a per-row `.to(long)`, is 3 %;
  - `result_adapter` 4 %.
- **`get_next_batch_to_run` 12 %**, mostly the coalesced prefill pick.
- **`process_input_requests` 3 %:** `build_sglang_qwen3_tts_request` and `adopt_prepared_tensors`.

The hottest innermost frames are per-row: the stream output builder, `decode_row`'s device and dtype check, `apply_sglang_qwen3_tts_result`, `prepare_decode_buffers:1354-1355`, `append_decode_input_history`, `reset_padding`.

The API process is separate (217 samples on its own GIL): HTTP streaming and PCM encoding.

**Unseeded repeat** (`served_profile_main/pyspy`, kit `6fe6d831b`; the seeded capture above moved to `pyspy_seeded`). The pipeline process holds the lock for 1,072 samples:

| Holder | Share |
|---|---|
| Scheduler | 50 % |
| Vocoder workers and scheduler | 23 % |
| Preprocessing pool | 21 % |
| Main-thread routing | 5 % |

Within the scheduler:

| Part | Share |
|---|---|
| `before_decode` | 16 % |
| Forward host side | 11 % |
| Sampling and logit shaping | 9 % |
| `emit_stream_output` | 7 % |
| Coalesced prefill pick | 7 % |
| `update_running_batch` | 3 % |

The innermost per-row frames are the same as seeded: `decode_row`, `stream_output_builder:2195`, `prepare_decode_buffers:1354`.

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

**These served cells, the first-chunk and opening served cells, and the first two py-spy captures ran seeded** (`--seed 1234`, carried over from the #3477 review driver). Our A/B protocol is unseeded.

- A seeded batch samples the talker's layer 0 through SGLang's torch `top_k_top_p_min_p_sampling_from_probs_torch`, which sorts the full vocab per row. Unseeded batches do not take it.
- Both arms of each served A/B were seeded, so the comparisons hold. The absolute numbers include that sampler.
- Pipeshift's bar (Nari's client) was unseeded throughout.
- The drivers are unseeded from kit `6fe6d831b`.

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

The same branch on Pipeshift's bar, seeds 0 to 2, median of seeds, main → branch, ms. Every request succeeded, with no underrun.

| req/s | 1 | 4 | 8 | 12 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| Audible p50 | 19.5 → 17.1 | 19.5 → 18.9 | 20.5 → 19.7 | 20.4 → 20.1 | 21.3 → 20.7 | 22.6 → 22.0 | 25.4 → 25.0 |
| First byte p50 | 19.5 → 17.1 | 19.4 → 18.6 | 20.3 → 19.4 | 20.1 → 19.8 | 20.9 → 20.5 | 22.1 → 21.5 | 22.4 → 22.0 |
| First byte p95 | 21.5 → 21.2 | 22.7 → 21.8 | 23.5 → 22.8 | 24.2 → 24.3 | 26.3 → 25.7 | 29.2 → 28.7 | 30.8 → 30.7 |
| Audible p95 | 24.7 → 29.0 | 31.5 → 30.1 | 30.9 → 30.2 | 31.1 → 30.5 | 32.4 → 31.4 | 37.0 → 38.2 | 100.1 → 99.0 |

`fix/qwen3-tts-opening-at-any-load` `7bbc943b2` on Pipeshift's bar at the rates where more than 24 streams are live, seeds 0 to 2, median of seeds (all three seeds in brackets for the p95 at 40), main → branch, ms. Every request succeeded, with no underrun and no stall.

| req/s | 20 | 30 | 40 |
|---|---|---|---|
| Audible p95 | 32.4 → 31.9 | 37.0 → 34.1 | 100.1 → 37.2 (36.8 to 37.6) |
| Audible p50 | 21.3 → 21.2 | 22.6 → 22.6 | 25.4 → 24.1 |
| First byte p50 | 20.9 → 20.9 | 22.1 → 22.3 | 22.4 → 23.7 |
| First byte p95 | 26.3 → 26.5 | 29.2 → 29.1 | 30.8 → 32.1 |
| Leading silence p95 | 10 → 10 | 10 → 10 | 80 → 10 |

- **At 40 req/s** the gap to Pipeshift's p95 (33.7) goes from 66.4 to 3.5 ms.
- **The cost** is the extra first-chunk frame above 24 streams: about 1.3 ms of first byte at 40 req/s.
- **No continuity cost:** #1901 gated the trim on stream count for continuity, and at these loads the branch shows no underrun and no stall.

`fix/qwen3-tts-honor-prompt-mode` `e72d22744` on Pipeshift's bar (Nari's client sends `non_streaming_mode: false`), seeds 0 to 2, median of seeds, main → branch, ms. Every request succeeded, with no underrun.

| req/s | 1 | 4 | 8 | 12 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|---|
| Audible p50 | 19.5 → 16.8 | 19.5 → 19.1 | 20.5 → 19.9 | 20.4 → 20.4 | 21.3 → 21.3 | 22.6 → 22.5 | 25.4 → 25.3 |
| First byte p95 | 21.5 → 21.1 | 22.7 → 22.0 | 23.5 → 23.0 | 24.2 → 24.5 | 26.3 → 26.5 | 29.2 → 29.0 | 30.8 → 30.8 |

- The shorter prefill shows at 1 req/s (−2.7 ms, outside the 1.2 ms band at that rate) and fades by 12 req/s, where prefills share steps.
- **WER at 1 req/s, seed 0, Qwen3-ASR** (`bar_wer.py`): main 2.11 % (1.85 % without its one sample over 50 %), branch 1.43 % (none over 50 %), over 136 requests each.
- A full SeedTTS EN WER with `non_streaming_mode: false` is the gate before a PR.

**First-chunk branch, unseeded ABAB** (`ab_unseeded_20261115/{main_1,firstchunk_1,main_2,firstchunk_2}`, streaming SeedTTS EN, 1,088 prompts per cell, every request completed).

| Cell | First audio mean, main → branch (pair 1 / pair 2), ms | RTF, main_1 / branch_1 / main_2 / branch_2 |
|---|---|---|
| Base c1 | 17.5 → 17.2 / 17.4 → 17.0 | 0.0642 / 0.0642 / 0.0640 / 0.0641 |
| Base c16 | 27.5 → 27.0 / 27.8 → 26.7 | 0.0993 / 0.0998 / 0.0991 / 0.0996 |
| Base c64 | 52.5 → 50.6 / 52.1 → 49.4 | 0.2162 / 0.2186 / 0.2157 / 0.2164 |
| CustomVoice c1 | 11.6 → 10.8 / 11.5 → 10.9 | 0.0625 / 0.0625 / 0.0626 / 0.0626 |
| CustomVoice c16 | 17.8 → 17.2 / 17.7 → 17.1 | 0.0907 / 0.0910 / 0.0910 / 0.0911 |
| CustomVoice c64 | 28.3 → 26.9 / 27.7 → 26.6 | 0.1758 / 0.1770 / 0.1770 / 0.1770 |

- First audio mean and median are lower in every cell in both pairs. Main against main moves at most 0.6 ms.
- RTF stays within main's own repeat band (up to 0.7 %). It leans 0.3 to 1.1 % higher at c16 and c64, which this repeat does not resolve. The seeded −2.9 % at Base c64 was noise.

**Tracing the first-chunk branch's RTF lean** (py-spy `--gil`, `firstchunk_profile/{main,firstchunk}`, served unseeded c64 and c16):
- The first-chunk worker's lock samples rise 58 → 69 at c64 and 36 → 44 at c16. The follow-up workers are level (199 → 206, 124 → 113). Without the 2 ms wait, each first chunk is a batch of its own and pays the batch's fixed host cost alone.
- The largest single piece of that cost is not batching. `Qwen3TTSCodecStateArena.acquire` → `zero_slot` (`codec_state_arena.py:224-227`) takes 280 of the worker's 690 ms inclusive. Every acquire built, in Python, a new list of views of the slot's rows of every state buffer and K/V ring before its one `_foreach_zero_` launch.
- The storage tensors are created once in the constructor and only written in place; gather and eager states replace tensors in their own copies. So the views can be built once per slot.
- **Fix:** `perf/qwen3-tts-codec-slot-views` `45f189833` (tree `2b0274b2d`) keeps `slot_buffers`, built in the constructor; `zero_slot` launches over the slot's cached list. The existing tests `test_arena_slot_reuse_starts_from_a_cold_state` and `test_arena_acquire_zeroes_only_the_acquired_slot` hold the contract.
- **Measurement:** served unseeded streaming at c1, c16 and c64, run in alternating order, twice: main, slot views alone, and `test/qwen3-tts-first-chunk-slot-views` `3e360e05e` (tree `8fd9a022b`), the first-chunk branch with slot views (`ab_slotviews_20261115`).

  | Cell | Slot views vs main, round 1 / round 2 | First chunk plus slot views vs main, round 1 / round 2 |
  |---|---|---|
  | Base c64 first audio mean | −2.6 / −0.8 ms | −3.0 / −2.4 ms |
  | CustomVoice c64 first audio mean | −1.4 / −1.5 ms | −1.8 / −0.6 ms |
  | Base c64 RTF | −0.5 % / +5.0 % | +3.7 % / +2.4 % |
  | CustomVoice c64 RTF | −2.6 % / −1.7 % | −0.3 % / +5.1 % |
  | c1 first audio mean | within ±0.4 ms (CustomVoice round 2: +1.5) | within ±1 ms |

  - **Main against main** in this run: RTF within 0.3 %; CustomVoice c64 throughput 73.97 against 69.03 req/s, so c64 throughput is too noisy to read.
  - Round 2 of both branch arms is noisier (CustomVoice c1 13.2 ms, Base c64 RTF +5 %) while main's second run stayed level.
  - **Reading:** slot views lowers c64 first audio in both rounds, and its RTF is inside the noise. It does not remove the first-chunk branch's Base c64 RTF lean, which reads +2.4 to +3.7 % here (+0.3 to +1.1 % in the earlier ABAB).
  - **Next step for that lean:** a lean under 3 % gets a kernel census and loop ledger at c64, main against the branch: vocoder initial batch counts, graph replays and talker idle per step. Not more pairs.

**Census at served c64, unseeded** (`firstchunk_census/{main,firstchunk}`, nsys timing mode, probe labels, `census_digest.py`):

| | Main | First-chunk branch |
|---|---|---|
| Window | 1,846 ms, 134 talker steps, 13.8 ms a step | 1,278 ms, 90 steps, 14.2 ms a step |
| Decode step | 101 at 7.96 ms | 75 at 7.75 ms |
| Prefill step | 33 at 13.27 ms | 15 at 19.03 ms |
| Initial vocoder batches | 33 batches, 175 streams | 16 batches, 132 streams |
| Initial worker's share of the window (in batches) | 527 ms, 28 % | 441 ms, 35 % |
| One-stream initial batch | 7.3 ms | 14.3 ms |
| Initial batches of 9 to 16 streams | 43 to 65 ms | 22 to 57 ms |
| Follow-up batches of 8 streams | 4.8 ms | 5.0 ms |

- A Base first chunk decodes the reference's codes as left context, so it is a wide decode.
- **Main's initial worker** (CUDA runtime table): 43 graph launches for 33 batches, 1,027 eager launches for staging, and 123 `cudaEventSynchronize` calls totalling 96.7 ms. The follow-up workers spent 2.4 and 5.6 ms in syncs.
**Graph overlap at served c64** (`CUPTI_ACTIVITY_KIND_GRAPH_TRACE` joined to the launching thread):

| Talker graph | Arm | Graphs | Mean span | Overlapped by first-chunk graphs | Overlapped by any vocoder graph | Mean when first-chunk overlap > 20 % | Otherwise |
|---|---|---|---|---|---|---|---|
| Decode (graph 685) | Main | 81 | 3,281 µs | 12.1 % | 37.8 % | 3,664 µs (n=14) | 3,201 µs |
| Decode (graph 685) | Branch | 48 | 3,046 µs | 7.0 % | 28.7 % | 3,458 µs (n=4) | 3,009 µs |
| Predictor (graph 439) | Main | 81 | 4,643 µs | 4.2 % | 27.4 % | 4,604 µs (n=10) | 4,649 µs |
| Predictor (graph 439) | Branch | 48 | 4,770 µs | 4.7 % | 31.6 % | 6,896 µs (n=3) | 4,629 µs |

- Vocoder graphs (first chunks and follow-ups) run beside the talker's critical-path graphs on the same SMs for 27 to 38 % of their time.
- A talker graph that a first-chunk graph overlaps runs 14 % (talker decode, main) to 49 % (predictor, branch) longer.
- **The defect the first-chunk branch exposes:** nothing orders vocoder work against the talker's critical path. The 2 ms wait only moved some of it aside. It is the same mechanism as the 10-24 ledger's "+99 % talker, +177 % predictor beside follow-ups at c64", and as Nari's one deadline order (10-27 notes).
- **The fix belongs there:** vocoder decodes scheduled around the talker's step, not concurrently with its graphs. The first-chunk branch then goes on top.
- **Next target, c64 first audio:** the initial worker's wide reference-context replays sharing the card with the talker, and its event waits. Which event each sync waits on is next. The timing mode's kernel-per-stream figures are near zero because graph trace mode records graphs whole, so read graph spans for device time.

**Same-card drift band.** The opening branch runs CustomVoice SeedTTS cells where its mechanism cannot act: the SeedTTS client sends no language, so the Ryan and English trim never engages. These cells are therefore an A/A of main's behaviour about 2.5 h apart.

| CustomVoice cell | First audio mean / median / p95, main → same behaviour | RTF | Throughput |
|---|---|---|---|
| c1 | 12.0 / 11.8 / 13.2 → 11.6 / 11.5 / 12.2 | 0.0636 → 0.0635 | 3.383 → 3.388 |
| c16 | 18.2 / 17.9 / 21.9 → 17.8 / 17.7 / 21.3 | 0.0927 → 0.0923 | 36.55 → 36.72 |
| c64 | 28.4 / 21.8 / 121.2 → 28.7 / 22.3 / 119.8 | 0.1790 → 0.1800 | 74.72 → 74.27 |

- **The band:** first audio within 0.5 ms on median, RTF and throughput within 0.6 %.
- **Against it, the first-chunk branch:** its first-audio drops of 0.9 to 3.2 ms on mean clear the band. Its c64 RTF changes (Base −2.9 %, CustomVoice +1.6 %) go both ways and need a repeat before either is read.

**Bar drift band.** Main's repeat, seed 0, run 3.5 h after the first (`ab_20261115/bar/main_aa_s0`):
- From 4 to 40 req/s, p50 moved within ±0.4 ms and first byte p95 within ±0.8 ms.
- At 1 req/s, the p50 moved −1.2 ms and the audible p95 swung from 21.9 to 29.8 ms, on 136 requests.

Against it:
- The first-chunk branch's bar p50 gains (0.3 to 0.8 ms from 4 to 40 req/s) sit at the band's edge, all in one direction.
- The opening branch's 40 req/s gain (63 ms) is far outside it.

- The audible p95 at 1 req/s comes from 136 requests. Main's own seeds gave 21.9, 29.3 and 24.7.
- Its tail there is the model's 10 ms opening quiet (leading silence p95 0 → 10 ms), which unseeded sampling varies. The first-byte p95 at that rate fell.

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
   - **Admission: the user chose SGLang's unbounded queue (11-15).** #1449 had set a fast reject past 64 running plus 64 queued.
     - `fix/qwen3-tts-unbounded-waiting-queue` `bf88f7d8c` (tree `f51eb68ba`) drops `max_queued_requests` from both Qwen3-TTS defaults (`config.py` `generation_admission_defaults` and `engine_builder.py` `generation_defaults`). The coordinator then sets no in-flight cap (`mp_runner.py:79-82`), and the scheduler treats `None` as unbounded (`omni_scheduler.py:1557`, `:444`).
     - An operator's `max_queued_requests` still bounds both. The NPU example configs keep theirs.
     - Tests cover both bounds. The tuning doc is updated.
     - On the bar our peak in flight is 47, so no rate up to 40 req/s can change. The served check is Nari's client at 80 req/s, main against the branch.
     - **Measured at 80 req/s, seed 0** (`overload_20261115`):

       | Arm | Succeeded | Failures |
       |---|---|---|
       | Main | 5.4 % (515), all with underruns | 7,014 HTTP 503 |
       | Branch | 0 % | 1,971 timeouts at 120 s |

       - The branch's first byte reached p50 19.5 s and p95 49.3 s, and its largest playback stall was 29.3 s.
       - With no bound, every arrival is preprocessed in the pipeline process, under the same lock as the scheduler and vocoder, so the backlog starves the running streams.
       - SGLang's queue holds tokens only. Ours holds preprocessed requests, and the backlog keeps preprocessing more.
       - Both arms collapse at 80 while 40 is clean. A knee search at 50, 60 and 70 req/s for both arms runs next.
     - Held, not for a PR, until the queue stops preprocessing ahead of engine capacity.
     - **Knee, seed 0** (`overload_20261115/knee`):

       | Arm | 50 req/s | 60 req/s | 70 req/s |
       |---|---|---|---|
       | Main | 100 % succeed, first byte p95 29 ms | 4.5 % succeed (325), 4,678 HTTP 503, every success underruns | 90.9 % succeed, first byte about 1 s, 1 underrun |
       | Unbounded | 100 % | 0 %; the client dispatched only 33.8 of 60 req/s with thousands open, so this arm is partly client-bound | — |

     - **Root cause of the collapse** (main's `serve.log`): `Qwen3-TTS incremental Codec state arena is full (64 slots); this request uses the left-context decoder`.
       - Counts per minute of the log: 4 in the 50 req/s run, then 2,078, 3,020 and 1,745 per minute during the 60 req/s run, and 300 to 600 per minute at 70 req/s.
       - The arena holds `DEFAULT_QWEN3_TTS_CODEC_STATE_SLOTS = 64` stream slots (`streaming_vocoder.py:86`, from #1997), the same as the engine's running bound.
       - A slot is taken at a stream's first decode (`streaming_vocoder.py:1884`) and released at its final decode (`release_stream_resources`, `:1996`). A stream whose generation ended keeps its slot while the vocoder decodes its tail.
       - Once the vocoder falls behind, more than 64 streams are live. Each one over the limit takes the left-context decoder (`:1886-1893`), which costs more per chunk. That slows the vocoder, so more streams stay live: a feedback loop.
       - In-flight then sits at the 128 cap, the excess gets 503, and the admitted streams underrun.
       - The served cells at c1, c16 and c64 and the bar up to 40 req/s logged 0 arena-full fallbacks. The loop only starts past about 50 req/s open-loop.
     - **Fix at the root:** a stream that finds the arena full waits for a slot, keeping the incremental path, instead of taking the costlier decoder. A full vocoder then holds new streams' first audio back (backpressure) rather than slowing every stream. The unbounded queue is reassessed on top of that fix.
     - **Proof of the mechanism:** py-spy `--gil` on main at 60 req/s (`overload_20261115/pyspy/main_rps60.pyspy.txt`). The two follow-up workers hold the lock for 312 and 306 samples, against 85 and 72 in the served c64 cell. Nearly all of it is in `decode_group` (`streaming_vocoder.py:3348`), the left-context path only fallback streams take.
     - **Branch** `fix/qwen3-tts-wait-for-codec-slot` `73b3e655c` (tree `424e1fec5`), from main:
       - **Waiting instead of fallback:** with async decode, a full arena sets `awaiting_codec_slot` and returns no plan, not the fallback.
       - **The waiter list:** `plan_stream_decode` records the stream in `codec_slot_waiters` (oldest first, under `state_lock`).
       - **Wake-up:** every slot that returns to the arena (`release_codec_slot`, `finish_codec_slots`) schedules the oldest live waiter's first decode.
       - **End of generation:** `handle_stream_done` keeps a waiting stream on the async path, so it is not decoded whole on the message thread.
       - **Synchronous decoder:** keeps the fallback.
       - **Stats:** `slot_waits` is added.
       - **Tests:** a full arena holds the stream and a release reschedules it; a waiting stream whose generation ends stays on the async path.
       - **Normal load:** the served cells never fill the arena (0 fallbacks), so normal load takes main's path.
     - **Measured on Pipeshift's bar, seed 0, main → branch** (`overload_20261115/knee/slotwait_s0`; main at 80 from `overload_20261115/main_s0`):

       | req/s | Succeeded | First byte p50 / p95, ms | Underrun requests | Largest stall, ms | HTTP 503 (measured) |
       |---|---|---|---|---|---|
       | 60 | 4.5 % → 100 % | 47.0 / 95.3 → 19.1 / 26.5 | 2,529 → 0 | 548 → 0 | 4,678 → 0 |
       | 70 | 90.9 % → 100 % | 908 / 1,034 → 22.8 / 185 | 1 → 3 | 0.3 → 4.8 | 765 → 0 |
       | 80 | 5.4 % → 93.8 % (9,010) | 47.6 / 94.1 → 768 / 867 | 2,587 → 0 | 521 → 0 | 7,014 → 591 |

       - Past capacity (about 75 req/s on this load) the branch serves at capacity with clean audio and turns away only the excess.
       - The arena-full fallback no longer fires: 0 warnings, since the warning only remains on the synchronous path.
     - `test/qwen3-tts-slot-wait-unbounded` `1e0d175e2` (tree `06d1c6f24`) puts the unbounded-queue commit on top, for measurement only.

       | Arm | 70 req/s | 80 req/s |
       |---|---|---|
       | Slot fix alone | 100 %, first byte p50 / p95 22.8 / 185 ms | 93.8 % (591 HTTP 503), first byte 768 / 867 ms, 0 underrun |
       | Slot fix plus unbounded queue | 100 %, first byte 22.6 / 170 ms, 1 underrun | **100 % (9,601)**, first byte 7.8 / 13.1 s, 11 underruns, largest stall 51 ms |

       - With the slot fix, the unbounded queue behaves as SGLang's does: every request is served, and under sustained overload the wait grows by the excess, about 5 req/s here.
       - Without the slot fix the same queue served 0 %.
     - **PR order:** two mechanisms, two PRs. The slot wait goes first; the unbounded queue is stacked on it and depends on it.
     - **Opened as #4077, ready for review (11-16).** Head `f9bc7891b` (sglang_omni tree `4bc1fa14d`) is main `ea8216220` plus the PR's four files. Body: `PR_CODEC_SLOT_WAIT_BODY.md`. Results: `artifacts/qwen3-tts-synthetic-20261114/codec_slot_20261116/` (host-221, two H100s on NUMA node 1, servers pinned to half a socket each, clients on node 0).
       - **Design at the head:** `build_incremental_plan` owns the waiting queue (enter where the flag is set, leave where the slot is acquired). The planning-failure handler and `release_stream_resources` are the only other exits. Waiting streams are counted against `Qwen3TTSCodecStateArena.free_slots()` (the free list), not `num_slots - active_slots()`, which counts retired slots as free. With an empty queue nothing takes a lock.
       - **Overload, three seeds, same-time pair** (`bar_overload`): 60 req/s 1.1-1.7 % → 97.3-100 %; 70 req/s 84-87 % → 98-100 %; 80 req/s 23-79 % → 83-93 %. Fewer underruns at every rate. The branch's failures are HTTP 503 at the 128 in-flight cap. Main's own 80 req/s swung from 70 % to 23 % between two runs, so readings near the knee vary by run.
       - **Normal load:** every cell inside the same-node A/A band, first audio equal or better everywhere (`pair_final`, `c16_rounds_*`, `whole16_rounds_fix`, `c64_rounds_*`). WER 1.06 % → 1.06 %, SIM 71.47 → 71.54. `slot_waits` 0 and `exhausted` 0 in every normal-load server.
       - **A lean that did not repeat:** the first full pair read streaming c16 RTF +1.6 % in both orders (A/A in the next block: +0.5 %). The ABAB timing census (`c16_census`) put the fix arm's extra talker EXTEND time in `cudaMalloc` stalls of 26 to 47 ms (allocator growth on the talker and a preprocessing pool thread), with equal GPU time and less GIL wait. Four warm c16 rounds in both orders then read RTF -0.4 % against the same-block A/A's -0.1 %.
       - **Lesson for the pair protocol:** the first cell after a boot carries allocator growth and lazy loads, so a lean only in the first cell is read again on warm servers (`seedtts_pair.py --rounds N --rounds-cell CELL`) next to a same-block A/A before it is called a cost.
       - **The c64 RTF lean, profiled (11-17, new lease host-221 GPUs [4 5], `artifacts/qwen3-tts-synthetic-20261114/codec_slot_c64_20261117/`).** Warm rounds read c64 RTF +0.64 % on the fix (7 of 8 readings worse) against the A/A's +0.26 % (5 of 8), and c16 -0.41 % (7 of 8 better). `loop_pair.py` ran the loop probe and `ar_loop_ledger.py` on both arms at once, four blocks:
         - Own device time per step equal in every block (6.70 to 6.78 ms at c64).
         - Block 1 (main `/workspace`, fix `/data`): fix period +0.07 / +0.50 ms, extra in `loop.result`. Block 3 (both on `/data`): +0.40 / +0.17 ms, extra in `mr.forward` host time. Lock-traced block 2: fix -0.29 / -0.24 ms. The extra time moves between ranges from block to block, and the fix's median step is up in two orders and down in two.
         - A/A ledger (main against main, both on `/data`): period -0.31 / -0.08 ms, client RTF +3.1 / +0.5 %, as large as any fix reading.
         - Lock trace (`gil_owners.py`): the engine holds the lock 557 us per `process_batch_result` call on both arms; no thread group holds it longer on the fix in both orders.
         - The fix's added normal-path work measures 24 to 34 ns per call, about 6 calls per stream: about 0.16 us per 11 ms engine step at c64, 2,000 times below the step differences read.
         - Read: run-to-run variance of the c64 step tail, not a cost of the fix. No code path in the change can produce it.
4. c64 decode, by GIL share. Each is a host-only change, so seeded outputs must stay byte-identical. Gate on c16 and c64 RTF and throughput.
   1. The talker scheduler's per-row Python: `write_feedback_buffers` re-gathers rows of last step's batched `output_embeds`, and `post_process_outputs` and `finalize` also loop over rows.
   2. Stream messages from engine to vocoder: one per request a step on both sides. This is `OmniScheduler`'s shared path, so other models must keep main's behaviour.
   3. The vocoder's follow-up planning: `launch_decode_plans` per batch of 8 streams.
   4. Then the predictor at c64 (4.22 ms a step, no custom backend covers it).
