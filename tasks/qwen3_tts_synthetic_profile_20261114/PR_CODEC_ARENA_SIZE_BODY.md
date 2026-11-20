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
- Whole requests keep rows of their own, and their count now follows what one replay takes. A whole request holds its row from its first window replay to its last, and a replay takes at most the largest batch the graphs capture, 8 with the defaults. Main sized this set to the stream rows, 64, so it would have grown to 128 with the stream rows at the cap. Only the initial worker takes and returns these rows, in arrival order, so a whole request never waits for a stream's row and never sends a stream to the left-context path. With the defaults the arena is 128 stream rows and 8 whole-request rows, 372.4 MiB on 1.7B, 21.9 MiB more than main's 64 and 64.
- The two sets stay apart because the two kinds of request take rows on opposite terms: a stream never waits and falls back to the left-context path, and a whole request always waits so its audio never depends on load. One shared set would make a whole request wait for rows that the follow-up workers and abort handling free on other threads, which needs a waiting queue with a claim per whole request kept across replays, cancellation and every release. SGLang avoids the same problem by allocating, freeing and keeping the waiting queue in one scheduler loop. Separate rows need none of that, for 8 rows.
- The arena grows with the cap: 2.74 MiB on 1.7B for every request the cap adds. In the default layout the vocoder shares the engine's process and GPU and allocates the arena before the engine measures free memory, so at the default memory fraction most of it comes out of the KV pool. A vocoder on its own GPU takes it from that GPU. A deep engine queue can set `codec_state_slots` below the cap, which the startup warning names.
- Every vocoder replica takes the whole cap, so with replicated engines each replica's arena grows with the replica count: 723.0 MiB with two replicas and 1.39 GiB with four. A cap per replica at the coordinator would bound one replica's share, and is a separate change.

This replaces the waiting queue this PR first proposed. With the arena sized to the cap, no stream waits for a row.

What the sizing covers: the cap counts a request until its result, or until the server publishes its abort or failure. The vocoder returns a stream's row only once it has handled the abort and any decode of the stream already running has finished. A stream admitted in that moment can find the rows full and decode on the left-context path to its end. Closing the moment would need the coordinator to keep counting an aborted request until the vocoder confirms its cleanup, a change to shared coordinator code for a separate PR. Sizing slack for it would cost about 219 MiB (the decodes the workers can have in flight) and still miss the abort's hop to the vocoder.

## Results

The overload and SeedTTS tables ran on `d37692113` and main `2a66cb083`, one H100 at a time, each server alone on the card in alternating order, pinned to the card's half of the socket with the client on the other socket. This head adds only the two fixes above for a waiting whole request and a raised stream wait. The overload client sends only streams, so no whole request waits there and a stream asks for a row exactly as on `d37692113`. With the defaults no SeedTTS request finds every row held either, and the pair against `d37692113` below stays within run-to-run noise.

Overload: Nari Labs' open client, unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured, seeds 0, 1 and 2.

| req/s | completed, main → branch | requests with underruns | first byte p50, ms | left-context decodes, main (branch 0) |
|---|---|---|---|---|
| 60 | 31.5 / 40.7 / 1.2 % → 100 / 100 / 100 % | 1,823 / 1,394 / 2,243 → 209 / 50 / 99 | 23 / 22 / 53 → 25 / 21 / 21 | 884 / 651 / 1,312 |
| 70 | 54.2 / 80.5 / 83.3 % → 99.9 / 98.6 / 99.6 % | 795 / 248 / 53 → 5 / 87 / 23 | 1,073 / 1,083 / 1,029 → 47 / 32 / 105 | 499 / 308 / 333 |
| 80 | 76.7 / 74.2 / 73.4 % → 83.4 / 90.1 / 88.6 % | 40 / 84 / 48 → 20 / 0 / 7 | 994 / 1,057 / 1,047 → 908 / 819 / 826 | 356 / 337 / 332 |

- Sampled every 2 s, the branch's arena held at most 65 to 66 of its 128 rows, while main's 64 rows were full at every rate.
- Main is unstable at 60 req/s (31.5, 40.7 and 1.2 % across seeds): once its rows run out, whether it recovers depends on the arrival sequence. At 80 req/s main's first byte covers only the requests it completed.
- Past capacity the branch serves at capacity, and the excess gets HTTP 503 at the in-flight cap.
- Requests with underruns at 60 req/s move a lot between runs of the same code. On one card, interleaved, seeds 0 to 5: the shared rows 118 / 121 / 79 / 0 / 32 / 29, a second set of rows for whole requests (`4c41ba4c8`) 3 / 103 / 26 / 46 / 43 / 23, and that second set against itself on seeds 0 to 2, 23 / 4 / 70 against 75 / 23 / 42. Across those runs the second set spans 3 to 103 (median 34.5) and the shared rows 0 to 121 (median 55.5), a difference a rank test puts at p 0.3. At 80 req/s, seeds 0 to 2, the shared rows completed 91.6 / 96.2 / 87.4 % against 83.7 / 93.3 / 87.9 %.

Normal load: SeedTTS EN, 1.7B Base, the full 1,088 prompts, unseeded, two orders.

| cell | first audio mean / median / p95, ms, main → branch | RTF |
|---|---|---|
| streaming c16 | 32.5 / 29.9 / 55.0 → 31.3 / 29.3 / 52.4 | -0.9 % |
| streaming c64 | 54.8 / 41.8 / 164.0 → 54.7 / 41.8 / 121.8 | +0.3 % |
| streaming c64, three warm rounds per order | 55.8 / 42.0 / 164.5 → 54.4 / 41.4 / 157.1 | -3.0 % |
| whole c16 (latency mean) | 381 → 376 ms | -0.9 % |

- 54 seeded requests over Base, CustomVoice and VoiceDesign, 27 streamed and 27 whole, are byte-identical to main, so neither the arena size nor whole requests sharing it changes audio.
- WER (streaming c16, Qwen3-ASR): 1.02 / 1.24 → 1.08 / 1.01 %. SIM 71.03 / 71.49 → 71.36 / 71.47. With byte-identical seeded audio, both differences are unseeded sampling.
- The KV pool keeps main's size. On one H100 at the default memory fraction, 1.7B Base holds 546,401 to 546,404 tokens on main and on this head alike. 1.7B CustomVoice holds 554,009 on this head, against 551,103 with a second set of rows.

This head against `d37692113`, the same corpus, one H100, each server alone, both orders:

| cell | first audio mean / median / p95, ms, `d37692113` → this head | RTF |
|---|---|---|
| streaming c1 | 20.8 / 20.5 / 33.0 → 21.4 / 20.5 / 34.4 | +0.8 % |
| streaming c16 | 37.1 / 30.6 / 69.4 → 32.2 / 29.8 / 52.4 | -7.2 % |
| streaming c64 | 57.3 / 44.2 / 164.8 → 57.2 / 43.3 / 163.7 | -1.5 % |
| whole c16 (latency mean) | 397 → 396 ms | +0.3 % |
| whole c1 (latency mean) | 250 → 249 ms | +0.2 % |

- The server that ran second in an order was faster at c64 and whole c16 in both orders, and `d37692113`'s first c16 cell after the first boot read 41.5 ms, so these differences are order and noise. At c1, `d37692113` alone read 20.2 and 21.3 ms in the two orders.
- WER 1.06 / 1.14 → 1.10 / 0.98 %, SIM 71.35 / 71.26 → 71.57 / 71.42. The KV pool holds 546,401 tokens on both.

Streams and whole requests at once on a real server, on this head:

- **1 row, 1.7B Base and 1.7B CustomVoice, six rounds each:** each round sent 16 streams, 2 whole requests, 6 clients that cancel before audio and 3 that drop after the first chunk. Every request that should finish did, and every row came back. Refused acquires equal left-context fallbacks, so a whole request waiting for a row counts none.
- **4 rows, continuous arrivals, 1.7B Base:** for 60 s, 8 stream clients that each send a new stream as the last ends, and a whole request every second, this head and `d37692113` alternating, four runs each. Every request finished and every row came back. Whole requests averaged 1.27 / 1.00 / 1.11 / 1.04 s on this head against 0.97 / 1.07 / 1.01 / 1.04 s, a difference an exact rank test puts at p 0.4, and the two slowest runs are the two that completed the fewest streams in their minute. About half the streams decoded on the left-context path on both heads. At this arrival rate `d37692113` did not starve a whole request either, so the waiting-order test under Tests, which forces the interleaving, is what shows the fix.
- **The defaults, 128 rows, both models, three rounds each:** the 1-row mix four times over, 108 clients at once, gave no refused acquire, no fallback, and every row back.

Every mode on a real 1.7B CustomVoice server, one stream and one whole request each:

- An uncapped asynchronous server, or `codec_state_slots` set to none, 0.5 or 0, refuses to start, with the error naming the slot count.
- The uncapped synchronous and rollback servers serve both with no arena.
- The default serves both with 128 rows.

## Tests

- `test_codec_arena_holds_every_request_the_coordinator_admits`: the slot count the vocoder factory builds with equals the coordinator's cap. Cases: the defaults, running and queued overrides, an unbounded queue, a factory-path override, and replicated shared and separate processes. On main every case fails, with 64 against the cap.
- `test_operator_codec_slots_replace_the_cap_but_not_the_check`: `--vocoder.factory.codec_state_slots` below the cap sets the slots, and the startup check still compares them with the cap.
- `test_qwen3_tts_uncapped_server_builds_rows_only_for_the_async_workers`: an unbounded engine queue, through the config, factory and real scheduler: the rollback with asynchronous workers, `async_decode: false` and deterministic inference start with no arena, and an explicit slot count builds an arena of that size. `test_qwen3_tts_async_vocoder_needs_a_slot_count`: asynchronous decode without a count refuses to start, uncapped or with the count set to none. `test_qwen3_tts_vocoder_refuses_a_slot_count_that_is_not_a_positive_integer`: 0.5, 0 and a boolean with asynchronous decode, and -5 with synchronous decode, are refused through the same path.
- `test_qwen3_tts_vocoder_without_an_arena_names_its_decode_path`, `test_qwen3_tts_vocoder_factory_checks_coverage_against_the_in_flight_cap` and `test_qwen3_tts_vocoder_warns_when_streams_can_outnumber_codec_slots`: the startup line for each mode, the check against the in-flight cap, and the warning for no cap or a cap above the slots.
- `test_qwen3_tts_whole_requests_keep_the_rows_one_replay_takes`: the arena keeps as many whole-request rows as the largest graph batch, 8 next to 128 stream rows and 2 next to 2. With the whole-request rows at the stream count, the 128-row case kept 128.
- `test_qwen3_tts_streams_and_whole_requests_keep_to_their_own_rows`: with one stream row free and a whole request waiting for a whole-request row, a new stream takes the stream row on the stateful decoder, and the whole request finishes on a row another whole request returns. With one shared set of rows, the stream fell back to the left-context path and the whole request waited for rows streams held.
