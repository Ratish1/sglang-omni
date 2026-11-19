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
- Whole requests keep their own, equal set of rows, so stream load never changes a whole request's audio. With the defaults the arena holds 128 stream rows and 128 whole-request rows, 701 MiB instead of 350 MiB. On an H100 the KV pool goes from 553,996 to 551,255 tokens (58.88 GiB), still above the 524,288 the running bound can demand.
- The arena grows with the cap: 5.48 MiB on 1.7B for every request `max_queued_requests` adds, taken from the KV pool since the vocoder loads first. A deep engine queue should set `codec_state_slots` below the cap, which the startup warning names.
- Every vocoder replica takes the whole cap, so with replicated engines each replica's arena grows with the replica count: 1.37 GiB with two replicas and 2.74 GiB with four. A cap per replica at the coordinator would bound one replica's share, and is a separate change.

This replaces the waiting queue this PR first proposed. With the arena sized to the cap, no stream waits for a row.

What the sizing covers: the cap counts a request until its result, or until the server publishes its abort or failure. The vocoder returns that stream's row only once it has handled the abort and any decode of the stream already running has finished. A request admitted in that moment can find the rows full and decode on the left-context path to its end. Closing the moment would need the coordinator to keep counting an aborted request until the vocoder confirms its cleanup, a change to shared coordinator code for a separate PR. Sizing slack for it would cost about 219 MiB (the decodes the workers can have in flight) and still miss the abort's hop to the vocoder.

## Results

Overload: Nari Labs' open client, unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured. Seed 0 is a same-time pair, one H100 per server, each pinned to its card's half of the socket. Seeds 1 and 2 ran each arm alone on one H100, in alternating order.

| req/s | completed, main → branch (seeds 0 / 1 / 2) | requests with underruns | first byte p50, ms | left-context decodes on main (seeds 1 / 2) |
|---|---|---|---|---|
| 60 | 1.1 / 1.0 / 2.1 % → 100 / 100 / 100 % | 2,284 / 2,269 / 2,381 → 4 / 1 / 0 | 52 / 51 / 47 → 21 / 20 / 20 | 1,183 / 1,397 |
| 70 | 85.6 / 88.0 / 89.9 % → 100 / 100 / 100 % | 13 / 23 / 15 → 5 / 4 / 4 | 999 / 969 / 926 → 67 / 28 / 23 | 343 / 358 |
| 80 | 4.0 / 76.5 / 77.5 % → 91.5 / 95.4 / 95.1 % | 2,229 / 31 / 8 → 0 / 0 / 2 | 62 / 1,013 / 989 → 799 / 722 / 729 | 351 / 345 |

- The branch took the left-context path 0 times. Sampled every 2 s, its arena held at most 57 to 61 rows at 60 req/s and 65 to 66 at 70 and 80 req/s, while main's 64 rows were full at every rate.
- Main's 80 req/s swings between seeds (4.0 % against 77 %): once its rows run out, whether it recovers depends on the arrival sequence. At 80 req/s main's first byte covers only the requests it completed.
- Past capacity the branch serves at capacity, and the excess gets HTTP 503 at the in-flight cap.
- 128 clients that each drop their stream after the first chunk and send a new one at once, two rounds: no acquire found the arena full, and every stream ended. The coordinator refused 37 and 40 of the new streams with HTTP 503 at the cap, because it still counted the dropped ones.

Normal load, the same 128-row arena: SeedTTS EN, 1.7B Base, the full 1,088 prompts, unseeded, both card orders.

| cell | first audio mean / median / p95, ms, main → branch | RTF |
|---|---|---|
| streaming c1 | 18.8 / 20.6 / 22.8 → 18.85 / 20.55 / 23.15 | -0.2 % |
| streaming c16 | 30.6 / 29.55 / 43.8 → 30.5 / 29.4 / 43.9 | -0.3 % |
| streaming c64, three warm passes per order | 54.25 / 41.27 / 169.3 → 54.45 / 41.7 / 153.7 | -0.4 % |
| whole c1, c16 | | -0.4 %, +0.2 % |

- The first c64 pass after boot read RTF +1.0 % in both orders. The three warm passes per order read -0.4 %.
- 54 seeded requests over Base, CustomVoice and VoiceDesign, 27 streamed and 27 whole, are byte-identical to main, so the arena size changes no audio.
- WER (streaming c16, Qwen3-ASR): 0.95 / 1.67 → 1.04 / 1.09 %. Main's second order has one sample over 50 %.

## Tests

- `test_codec_arena_holds_every_request_the_coordinator_admits`: the slot count the vocoder factory builds with equals the coordinator's cap. Cases: the defaults, running and queued overrides, an unbounded queue, a factory-path override, and replicated shared and separate processes. On main every case fails, with 64 against the cap.
- `test_operator_codec_slots_replace_the_cap_but_not_the_check`: `--vocoder.factory.codec_state_slots` below the cap sets the slots, and the startup check still compares them with the cap.
- `test_qwen3_tts_uncapped_server_builds_rows_only_for_the_async_workers`: an unbounded engine queue, through the config, factory and real scheduler: the rollback with asynchronous workers, `async_decode: false` and deterministic inference start with no arena, and an explicit slot count builds an arena of that size. `test_qwen3_tts_async_vocoder_needs_a_slot_count`: asynchronous decode without a count refuses to start, uncapped or with the count set to none. `test_qwen3_tts_vocoder_refuses_a_slot_count_that_is_not_a_positive_integer`: 0.5, 0 and a boolean with asynchronous decode, and -5 with synchronous decode, are refused through the same path.
- `test_qwen3_tts_vocoder_without_an_arena_names_its_decode_path` and `test_qwen3_tts_vocoder_warns_when_streams_can_outnumber_codec_slots`: the startup line for each mode, and the warning for no cap or a cap above the slots.
