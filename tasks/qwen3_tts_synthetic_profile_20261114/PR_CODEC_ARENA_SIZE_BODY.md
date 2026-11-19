## Problem

With asynchronous decode, the streaming vocoder keeps each stream's incremental codec state in an arena row, from the stream's first decode to its last. The arena had 64 stream rows, a constant equal to the engine's running bound, and the startup check compared the rows against that bound.

The engine's bound does not limit the streams the vocoder holds. A stream's last decode comes after its engine slot frees, so the vocoder holds the running streams plus every stream still decoding its tail, up to every request the coordinator admits: 128 with the defaults (64 running plus 64 queued). Requests bind to replicas round-robin with no cap per replica, so one vocoder replica may hold the whole cap.

A stream that found the 64 rows taken decoded on the left-context path, which costs more per chunk. The vocoder fell further behind, more streams stayed live past their generation, and more took the slow path. Past about 60 req/s of Poisson arrivals (1.7B CustomVoice), almost every request underran and the rest got HTTP 503 at the in-flight cap.

## Change

- `Qwen3TTSPipelineConfig.stage_factory_kwargs("vocoder")` passes the coordinator's own in-flight cap, `resolve_coordinator_max_in_flight`, as `codec_state_slots`, and as `max_in_flight` for the startup check. The arena and the coordinator read one formula. An operator's `--vocoder.factory.codec_state_slots` still wins.
- `resolve_coordinator_max_in_flight` and `engine_stage_capacity` move unchanged from the pipeline runner to `sglang_omni/config/topology.py`, so the model config reads the cap from the config layer.
- The scheduler builds the arena, and the graph runners that read it, only when it decodes asynchronously on the stateful decoder: those asynchronous workers are the only code that takes rows. A synchronous stateful decode (the default on Intel XPU, CPU and Apple MPS, `async_decode: false`, and deterministic inference) keeps each stream's state on the stream. Those servers now allocate no arena, where main allocated 64 idle rows (175 MiB on 1.7B), and their model info reports `codec_state` as not enabled. The left-context rollback keeps no Codec state and allocated no arena on main either, also when its decodes run on the asynchronous workers. The Ascend NPU examples turn asynchronous decode on, so their arenas move from 64 rows to their caps: 9, 32, 9 and 9 (not run on NPU).
- The 64-slot default is gone from the factory and the scheduler. A server whose engine queue is unbounded has no cap, so with asynchronous stateful decode the scheduler refuses to start, after the checkpoint loads, unless `codec_state_slots` is set, and the startup check then warns that streams past those slots take the left-context path. The synchronous and rollback modes start uncapped as before. The startup line names the mode, and the warning when the slots are below the cap stays.
- The slot count comes from free-form factory config, so the scheduler refuses a set count that is not a positive integer, in every mode as main did. Main truncated it with `int()` first, so 96.0 or 1.5 now refuse where main took 96 or 1, and a fraction such as 0.5 never reaches the arena as zero rows.
- Whole requests keep their own, equal set of rows, so stream load never changes a whole request's audio. With the defaults the arena holds 128 stream rows and 128 whole-request rows, 701 MiB instead of 350 MiB. On an H100 the KV pool goes from 554,006 to 551,255 tokens (59.17 to 58.88 GiB), still above the 524,288 the running bound can demand.
- The arena grows with the cap: 5.48 MiB on 1.7B for every request `max_queued_requests` adds, taken from the KV pool since the vocoder loads first. A deep engine queue should set `codec_state_slots` below the cap, which the startup warning names.
- Every vocoder replica takes the whole cap, so with replicated engines each replica's arena grows with the replica count: 1.37 GiB with two replicas and 2.74 GiB with four. A cap per replica at the coordinator would bound one replica's share, and is a separate change.

This replaces the waiting queue this PR first proposed. With the arena sized to the cap, no stream waits for a row.

What the sizing covers: the cap counts a request until its result, or until the server publishes its abort or failure. The vocoder returns that stream's row only once it has handled the abort and any decode of the stream already running has finished. A request admitted in that moment can find the rows full and decode on the left-context path to its end. Closing the moment would need the coordinator to keep counting an aborted request until the vocoder confirms its cleanup, a change to shared coordinator code for a separate PR. Sizing slack for it would cost about 219 MiB (the decodes the workers can have in flight) and still miss the abort's hop to the vocoder.

## Results

Everything below ran on this head and on main `2a66cb083`, one H100, each server alone on the card in alternating order, pinned to the card's half of the socket with the client on the other socket.

Overload: Nari Labs' open client, unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured, seeds 0, 1 and 2.

| req/s | completed, main → branch | requests with underruns | first byte p50, ms | left-context decodes, main (branch 0) |
|---|---|---|---|---|
| 60 | 29.7 / 100 / 1.3 % → 100 / 100 / 100 % | 1,746 / 23 / 2,255 → 1 / 7 / 6 | 24 / 24 / 51 → 20 / 21 / 21 | 832 / 88 / 1,359 |
| 70 | 86.8 / 87.0 / 86.4 % → 100 / 100 / 99.8 % | 13 / 23 / 10 → 15 / 4 / 4 | 978 / 989 / 981 → 23 / 26 / 104 | 330 / 331 / 337 |
| 80 | 77.5 / 74.7 / 75.5 % → 91.9 / 93.7 / 92.1 % | 18 / 20 / 10 → 4 / 6 / 1 | 990 / 1,047 / 1,017 → 796 / 767 / 788 | 343 / 327 / 330 |

- Sampled every 2 s, the branch's arena held at most 58 to 67 of its 128 rows, while main's 64 rows were full at every rate.
- Main is unstable at 60 req/s (29.7, 100 and 1.3 % across seeds): once its rows run out, whether it recovers depends on the arrival sequence. At 80 req/s main's first byte covers only the requests it completed.
- Past capacity the branch serves at capacity, and the excess gets HTTP 503 at the in-flight cap.

Normal load: SeedTTS EN, 1.7B Base, the full 1,088 prompts, unseeded, two orders.

| cell | first audio mean / median / p95, ms, main → branch | RTF |
|---|---|---|
| streaming c16 | 30.2 / 29.3 / 43.7 → 29.6 / 28.8 / 42.4 | -1.0 % |
| streaming c64, three warm rounds per order | 53.3 / 40.9 / 169.2 → 53.7 / 41.1 / 156.0 | +0.9 % |
| whole c16 (latency mean) | 375 → 374 ms | 0.0 % |

- Main against main on the same card, three warm c64 rounds per order: first audio mean +0.2 ms, median +0.6 ms, RTF +2.6 %. The branch's c64 differences sit inside that. The first c64 pass after each boot read mean +1.3 ms and median +0.15 ms, which the warm rounds do not repeat.
- 54 seeded requests over Base, CustomVoice and VoiceDesign, 27 streamed and 27 whole, are byte-identical to main, so the arena size changes no audio.
- WER (streaming c16, Qwen3-ASR): 1.04 / 1.09 → 1.01 / 1.06 %. SIM 71.51 / 71.53 → 71.40 / 71.34. With byte-identical seeded audio, both differences are unseeded sampling.

Every mode on a real 1.7B CustomVoice server, one stream and one whole request each: an uncapped asynchronous server, `codec_state_slots` none, 0.5 or 0 refuse to start with the error naming the slot count. The uncapped synchronous and rollback servers serve both with no arena, and the default serves both with 128 and 128 rows.

## Tests

- `test_codec_arena_holds_every_request_the_coordinator_admits`: the slot count the vocoder factory builds with equals the coordinator's cap. Cases: the defaults, running and queued overrides, an unbounded queue, a factory-path override, and replicated shared and separate processes. On main every case fails, with 64 against the cap.
- `test_operator_codec_slots_replace_the_cap_but_not_the_check`: `--vocoder.factory.codec_state_slots` below the cap sets the slots, and the startup check still compares them with the cap.
- `test_qwen3_tts_uncapped_server_builds_rows_only_for_the_async_workers`: an unbounded engine queue, through the config, factory and real scheduler: the rollback with asynchronous workers, `async_decode: false` and deterministic inference start with no arena, and an explicit slot count builds an arena of that size. `test_qwen3_tts_async_vocoder_needs_a_slot_count`: asynchronous decode without a count refuses to start, uncapped or with the count set to none. `test_qwen3_tts_vocoder_refuses_a_slot_count_that_is_not_a_positive_integer`: 0.5, 0 and a boolean with asynchronous decode, and -5 with synchronous decode, are refused through the same path.
- `test_qwen3_tts_vocoder_without_an_arena_names_its_decode_path` and `test_qwen3_tts_vocoder_warns_when_streams_can_outnumber_codec_slots`: the startup line for each mode, and the warning for no cap or a cap above the slots.
