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

Everything below ran on this head and on main `2a66cb083` on one H100, each server alone on the card with the first server alternating, pinned to the card's half of the socket with the client on the other socket.

Overload: Nari Labs' open client, unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured, seeds 0, 1 and 2.

| req/s | completed, main → this head | requests with underruns | first byte p50, ms | left-context decodes, main (this head 0) |
|---|---|---|---|---|
| 60 | 0.4 / 0.9 / 0.6 % → 100 / 100 / 98.7 % | 2,067 / 2,085 / 1,972 → 55 / 105 / 340 | 59 / 55 / 56 → 21 / 21 / 326 | 1,213 / 1,258 / 1,183 |
| 70 | 82.5 / 0.2 / 81.7 % → 89.9 / 94.7 / 98.1 % | 149 / 2,008 / 79 → 271 / 154 / 78 | 1,032 / 60 / 1,051 → 895 / 795 / 438 | 316 / 1,152 / 304 |
| 80 | 66.2 / 72.0 / 0.4 % → 85.1 / 88.1 / 75.9 % | 301 / 155 / 1,962 → 16 / 28 / 143 | 1,170 / 1,080 / 57 → 882 / 840 / 1,007 | 310 / 332 / 1,119 |

- Sampled every 2 s, this head held at most 62 to 67 of its 128 stream rows, while main's 64 stream rows were full in every cell.
- Main is unstable past 60 req/s: once its rows run out, whether it recovers depends on the arrival sequence, and in five of the nine cells it completed under 1 %. In those cells its first byte covers only the requests it completed.
- Past capacity this head serves at capacity, and the excess gets HTTP 503 at the in-flight cap.
- The overload client sends only streams, and their path is main's except for the row count.

Normal load: SeedTTS EN, 1.7B Base, the full 1,088 prompts, unseeded, two orders.

| cell | first audio mean / median / p95, ms, main → this head | RTF |
|---|---|---|
| streaming c1 | 20.9 / 20.5 / 34.5 → 21.3 / 20.6 / 35.4 | +0.6 % |
| streaming c16 | 35.8 / 30.7 / 67.2 → 33.4 / 30.5 / 52.8 | -5.1 % |
| streaming c64 | 58.2 / 44.0 / 177.8 → 61.0 / 44.6 / 208.8 | +1.9 % |
| streaming c64, three warm rounds per order | 59.9 / 43.2 / 203.1 → 58.8 / 44.2 / 182.5 | -0.3 % |
| whole c1 (latency mean) | 250 → 251 ms | -0.2 % |
| whole c16 (latency mean) | 397 → 401 ms | +0.7 % |
| whole c64, three warm rounds per order (latency mean) | 828 → 834 ms | +0.7 % |

- At c64 the server that ran second in an order was the slower one in both orders, by up to 30 % in RTF, so a single c64 cell moves with the order. Over six warm rounds each, this head's c64 first audio and RTF match main's.
- At whole c64 the window graphs ran 3,094 and 3,068 replays on main and 3,081 and 3,062 on this head, and the tail graphs 2,379 and 2,350 against 2,383 and 2,361, for the same requests. Eight rows keep the replays as full as main's 64. Main's latency ranged 805 to 864 ms over its six rounds and this head's 795 to 882.
- 54 seeded requests over Base, CustomVoice and VoiceDesign, with each `non_streaming_mode` value, 27 streamed and 27 whole, are byte-identical to main.
- WER (streaming c16, Qwen3-ASR): 1.05 / 1.01 → 1.05 / 1.04 %. SIM 71.43 / 71.32 → 71.28 / 71.21. With byte-identical seeded audio, both differences are unseeded sampling.
- On one H100 at the default memory fraction, 1.7B Base holds 546,401 and 546,475 KV tokens on main and 546,226 on this head.

Streams and whole requests at once on a real server, on this head:

- **1 stream row, 1.7B Base and 1.7B CustomVoice, six rounds each:** each round sent 16 streams, 2 whole requests, 6 clients that cancel before audio and 3 that drop after the first chunk. Every request that should finish did, and every row came back. Refused acquires equal left-context fallbacks.
- **4 stream rows, continuous arrivals:** for 60 s, 8 stream clients that each send a new stream as the last ends, and a whole request every second. About half the streams decoded on the left-context path, and every whole request finished on its own rows: 1.7B Base in 0.80 s median and 1.08 s at most, beside 586 streams, and 1.7B CustomVoice in 0.94 and 1.43 s, beside 485. Every row came back.
- **The defaults, 128 stream rows, both models, three rounds each:** the 1-row mix four times over, 108 clients at once, gave no refused acquire, no fallback, and every row back.

Every mode on a real 1.7B CustomVoice server, one stream and one whole request each:

- An uncapped asynchronous server, or `codec_state_slots` set to none, 0.5 or 0, refuses to start, with the error naming the slot count.
- The uncapped synchronous and rollback servers serve both with no arena.
- The default serves both with 128 stream rows and 8 whole-request rows.

## Tests

- `test_codec_arena_holds_every_request_the_coordinator_admits`: the slot count the vocoder factory builds with equals the coordinator's cap. Cases: the defaults, running and queued overrides, an unbounded queue, a factory-path override, and replicated shared and separate processes. On main every case fails, with 64 against the cap.
- `test_operator_codec_slots_replace_the_cap_but_not_the_check`: `--vocoder.factory.codec_state_slots` below the cap sets the slots, and the startup check still compares them with the cap.
- `test_qwen3_tts_uncapped_server_builds_rows_only_for_the_async_workers`: an unbounded engine queue, through the config, factory and real scheduler: the rollback with asynchronous workers, `async_decode: false` and deterministic inference start with no arena, and an explicit slot count builds an arena of that size. `test_qwen3_tts_async_vocoder_needs_a_slot_count`: asynchronous decode without a count refuses to start, uncapped or with the count set to none. `test_qwen3_tts_vocoder_refuses_a_slot_count_that_is_not_a_positive_integer`: 0.5, 0 and a boolean with asynchronous decode, and -5 with synchronous decode, are refused through the same path.
- `test_qwen3_tts_vocoder_without_an_arena_names_its_decode_path`, `test_qwen3_tts_vocoder_factory_checks_coverage_against_the_in_flight_cap` and `test_qwen3_tts_vocoder_warns_when_streams_can_outnumber_codec_slots`: the startup line for each mode, the check against the in-flight cap, and the warning for no cap or a cap above the slots.
- `test_qwen3_tts_whole_requests_keep_the_rows_one_replay_takes`: the arena keeps as many whole-request rows as the largest graph batch, 8 next to 128 stream rows and 2 next to 2. With the whole-request rows at the stream count, the 128-row case kept 128.
- `test_qwen3_tts_streams_and_whole_requests_keep_to_their_own_rows`: with one stream row free and a whole request waiting for a whole-request row, a new stream takes the stream row on the stateful decoder, and the whole request finishes on a row another whole request returns. With one shared set of rows, the stream fell back to the left-context path and the whole request waited for rows streams held.
