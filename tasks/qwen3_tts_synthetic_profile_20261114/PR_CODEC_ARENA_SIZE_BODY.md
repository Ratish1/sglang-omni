## Problem

With asynchronous decode, the streaming vocoder keeps each stream's incremental codec state in an arena row, from the stream's first decode to its last. The arena had 64 stream rows, a constant equal to the engine's running bound, and the startup check compared the rows against that bound.

The engine's bound does not limit the streams the vocoder holds. A stream's last decode comes after its engine slot frees, so the vocoder holds the running streams plus every stream still decoding its tail, up to every request the coordinator admits: 128 with the defaults (64 running plus 64 queued). Requests bind to replicas round-robin with no cap per replica, so one vocoder replica may hold the whole cap.

A stream that found the 64 rows taken decoded on the left-context path, which costs more per chunk. The vocoder fell further behind, more streams stayed live past their generation, and more took the slow path. Past about 60 req/s of Poisson arrivals (1.7B CustomVoice), almost every request underran and the rest got HTTP 503 at the in-flight cap.

## Change

- `Qwen3TTSPipelineConfig.stage_factory_kwargs("vocoder")` passes the coordinator's own in-flight cap, `resolve_coordinator_max_in_flight`, as `codec_state_slots`, and as `max_in_flight` for the startup check. The arena and the coordinator read one formula. An operator's `--vocoder.factory.codec_state_slots` still wins.
- `resolve_coordinator_max_in_flight` and `engine_stage_capacity` move unchanged from the pipeline runner to `sglang_omni/config/topology.py`, so the model config reads the cap from the config layer.
- The scheduler builds the arena, and the graph runners that read it, only when it decodes asynchronously on the stateful decoder: those asynchronous workers are the only code that takes rows. A synchronous stateful decode (the default on Intel XPU, CPU and Apple MPS, `async_decode: false`, and deterministic inference) keeps each stream's state on the stream. Those servers now allocate no arena, where main allocated 64 idle rows (175 MiB on 1.7B), and their model info reports `codec_state` as not enabled. The left-context rollback keeps no Codec state and allocated no arena on main either, also when its decodes run on the asynchronous workers. The Ascend NPU examples turn asynchronous decode on, so their arenas move from 64 rows to their caps: 9, 32, 9 and 9 (not run on NPU).
- The 64-slot default is gone from the factory and the scheduler. A server whose engine queue is unbounded has no cap, so with asynchronous stateful decode the vocoder factory refuses to start unless `codec_state_slots` is set, and the startup check then warns that streams past those slots take the left-context path. The synchronous and rollback modes start uncapped as before. The factory and the scheduler decide the asynchronous mode with one function, `is_async_decode`, so the factory refuses exactly the modes whose scheduler would build an arena. The startup line names the mode, and the warning when the slots are below the cap stays.
- The slot count comes from free-form factory config, so the factory refuses a set count that is not a positive integer, in every mode as main did. Main truncated it with `int()` first, so 96.0 or 1.5 now refuse where main took 96 or 1, and a fraction such as 0.5 never reaches the arena as zero rows.
- Both refusals come before the factory loads the checkpoint, so a server that cannot start loads no tokenizer and builds no decoder.
- Whole requests keep rows of their own, as many as the stream rows, as main pairs them. A whole request holds its row from its first window replay to its last and counts in the same in-flight cap as a stream, so with the cap's rows no whole request waits for one. A replay decodes at most the largest batch the graphs capture, 8 with the defaults, and the whole requests past it wait for a later replay, not for a row. Only the initial worker takes and returns these rows, in arrival order, so a whole request never waits for a stream's row and never sends a stream to the left-context path. With the defaults the arena is 128 stream rows and 128 whole-request rows, 701.1 MiB on 1.7B, 350.6 MiB more than main's 64 and 64.
- The two sets stay apart because the two kinds of request take rows on opposite terms: a stream never waits and falls back to the left-context path, and a whole request always waits so its audio never depends on load. One shared set would make a whole request wait for rows that the follow-up workers and abort handling free on other threads, which needs a waiting queue with a claim per whole request kept across replays, cancellation and every release. SGLang avoids the same problem by allocating, freeing and keeping the waiting queue in one scheduler loop.
- Whole-request rows follow how many requests hold state, not how many one replay decodes. With only one replay's batch of rows, eight long requests holding them made short ones wait behind every long window, and requests of different widths split into replays of one row each.
- The arena grows with the cap: 5.48 MiB on 1.7B for every request the cap adds, half for stream rows and half for whole-request rows. In the default layout the vocoder shares the engine's process and GPU and allocates the arena before the engine measures free memory, so at the default memory fraction most of it comes out of the KV pool. A vocoder on its own GPU takes it from that GPU. A deep engine queue can set `codec_state_slots` below the cap, which the startup warning names.
- Every vocoder replica takes the whole cap, so with replicated engines each replica's arena grows with the replica count: 1.37 GiB with two replicas and 2.74 GiB with four. A cap per replica at the coordinator would bound one replica's share, and is a separate change.

This replaces the waiting queue this PR first proposed. With the arena sized to the cap, no stream waits for a row.

What the sizing covers: the cap counts a request until its result, or until the server publishes its abort or failure. The vocoder returns a stream's row only once it has handled the abort and any decode of the stream already running has finished. A stream admitted in that moment can find the rows full and decode on the left-context path to its end. Closing the moment would need the coordinator to keep counting an aborted request until the vocoder confirms its cleanup, a change to shared coordinator code for a separate PR. Sizing slack for it would cost about 219 MiB (the decodes the workers can have in flight) and still miss the abort's hop to the vocoder.

## Results

Everything below ran on this head and on main `2a66cb083` on one H100, each server alone on the card with the first server alternating, pinned to the card's half of the socket with the client on the other socket.

Overload: Nari Labs' open client, unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured, seeds 0, 1 and 2.

| req/s | completed, main → this head | requests with underruns | first byte p50, ms | left-context decodes, main (this head 0) |
|---|---|---|---|---|
| 60 | 0.5 / 0.5 / 1.8 % → 100 / 100 / 100 % | 1,984 / 2,068 / 2,324 → 56 / 140 / 45 | 56 / 57 / 54 → 22 / 25 / 22 | 1,129 / 1,197 / 1,370 |
| 70 | 79.7 / 77.9 / 80.2 % → 99.4 / 96.8 / 97.6 % | 127 / 153 / 170 → 38 / 14 / 3 | 1,091 / 1,128 / 1,067 → 323 / 715 / 673 | 306 / 310 / 318 |
| 80 | 67.6 / 68.3 / 77.8 % → 78.6 / 84.7 / 88.5 % | 336 / 292 / 16 → 182 / 102 / 77 | 1,153 / 1,152 / 982 → 958 / 881 / 811 | 299 / 312 / 353 |

- Sampled every 2 s, this head held at most 64 to 66 of its 128 stream rows, while main's 64 stream rows were full in every cell.
- Main collapses at 60 req/s: once its rows run out it completes under 2 % on every seed, and its first byte there covers only the requests it completed.
- Past capacity this head serves at capacity, and the excess gets HTTP 503 at the in-flight cap.
- The overload client sends only streams, and their path is main's except for the row count.

Normal load: SeedTTS EN, 1.7B Base, the full 1,088 prompts, unseeded, two orders.

| cell | first audio mean / median / p95, ms, main → this head | RTF |
|---|---|---|
| streaming c1 | 21.1 / 20.4 / 34.5 → 21.8 / 20.6 / 35.8 | +0.8 % |
| streaming c16 | 35.0 / 30.2 / 64.0 → 32.3 / 29.1 / 54.3 | -5.4 % |
| streaming c64 | 57.2 / 42.5 / 144.2 → 58.5 / 43.5 / 206.7 | +0.3 % |
| streaming c64, three warm rounds per order | 60.5 / 43.9 / 207.1 → 62.8 / 44.5 / 192.3 | +3.6 % |
| whole c1 (latency mean) | 250 → 250 ms | 0.0 % |
| whole c16 (latency mean) | 394 → 385 ms | -2.4 % |
| whole c64, three warm rounds per order (latency mean) | 812 → 835 ms | +2.7 % |

- The c64 rounds move more between rounds of one server than between the servers: main's first audio mean ranged 53.7 to 70.3 ms over its six warm rounds and this head's 54.7 to 76.0, and whole c64 latency 784 to 852 against 788 to 890 ms. A rank test puts the streaming difference at p 0.7 and the whole one at p 0.4. With a row for every whole request on both servers, the window graphs ran 3,107 and 3,082 replays on main and 3,078 and 3,091 on this head, and the tail graphs 2,399 and 2,355 against 2,369 and 2,396, for the same requests.
- At c1 one stream decodes at a time on the same row and graphs as main, and the median matches. Main's own c1 mean read 20.5 to 21.3 ms across runs on this card.
- 54 seeded requests over Base, CustomVoice and VoiceDesign, with each `non_streaming_mode` value, 27 streamed and 27 whole, are byte-identical to main.
- WER (streaming c16, Qwen3-ASR): 1.05 / 0.99 → 0.90 / 0.95 %. SIM 71.57 / 71.47 → 71.33 / 71.48. With byte-identical seeded audio, both differences are unseeded sampling.
- On one H100 at the default memory fraction, 1.7B Base holds 546,401 KV tokens on main and 543,736 on this head, 0.5 % fewer for the larger arena.

Whole-request rows against the replay batch, on the real 1.7B Base codec with captured window graphs, 128 stream rows and replays of at most 8: eight 384-frame whole requests and then seven 16-frame ones finish the short ones on the second replay, 18.7 ms in, with this head's 128 whole-request rows, and the same with main's 64. In the review's run with one replay's batch of 8 rows, the short ones waited behind the long ones' windows and finished on the seventh replay, 92.6 ms in. Requests that end on different window widths take 3 replays with 128 or 64 rows, and took 10 with 8 in that run, eight of them a single row.

Streams and whole requests at once on a real server, on this head:

- **1 stream row, 1.7B Base and 1.7B CustomVoice, six rounds each:** each round sent 16 streams, 2 whole requests, 6 clients that cancel before audio and 3 that drop after the first chunk. Every request that should finish did, and every row came back. Refused acquires equal left-context fallbacks.
- **4 stream rows, continuous arrivals:** for 60 s, 8 stream clients that each send a new stream as the last ends, and a whole request every second. About half the streams decoded on the left-context path, and every whole request finished on its own rows: 1.7B Base in 0.83 s median and 1.09 s at most, beside 558 streams, and 1.7B CustomVoice in 0.91 and 1.61 s, beside 518. Every row came back.
- **The defaults, 128 stream rows, both models, three rounds each:** the 1-row mix four times over, 108 clients at once, gave no refused acquire, no fallback, and every row back.

Every mode on a real 1.7B CustomVoice server, one stream and one whole request each:

- An uncapped asynchronous server, or `codec_state_slots` set to none, 0.5 or 0, refuses to start, with the error naming the slot count, before the vocoder loads its tokenizer or builds its decoder.
- The uncapped synchronous and rollback servers serve both with no arena.
- The default serves both with 128 stream rows and 128 whole-request rows.

## Tests

- `test_codec_arena_holds_every_request_the_coordinator_admits`: the slot count the vocoder factory builds with equals the coordinator's cap. Cases: the defaults, running and queued overrides, an unbounded queue, a factory-path override, and replicated shared and separate processes. On main every case fails, with 64 against the cap.
- `test_operator_codec_slots_replace_the_cap_but_not_the_check`: `--vocoder.factory.codec_state_slots` below the cap sets the slots, and the startup check still compares them with the cap.
- `test_qwen3_tts_uncapped_server_builds_rows_only_for_the_async_workers`: an unbounded engine queue, through the config, factory and real scheduler: the rollback with asynchronous workers, `async_decode: false` and deterministic inference start with no arena, and an explicit slot count builds an arena of that size. `test_qwen3_tts_async_vocoder_needs_a_slot_count`: asynchronous decode without a count refuses to start, uncapped or with the count set to none. `test_qwen3_tts_vocoder_refuses_a_slot_count_that_is_not_a_positive_integer`: 0.5, 0 and a boolean with asynchronous decode, and -5 with synchronous decode, are refused through the same path. Both run with a checkpoint whose load fails, so each refusal must come before the load.
- `test_qwen3_tts_vocoder_without_an_arena_names_its_decode_path`, `test_qwen3_tts_vocoder_factory_checks_coverage_against_the_in_flight_cap` and `test_qwen3_tts_vocoder_warns_when_streams_can_outnumber_codec_slots`: the startup line for each mode, the check against the in-flight cap, and the warning for no cap or a cap above the slots.
- `test_qwen3_tts_whole_requests_keep_as_many_rows_as_streams`: the arena keeps as many whole-request rows as stream rows, 128 and 2. With the whole-request rows at one replay's batch, the 128-row case kept 8.
- `test_qwen3_tts_streams_and_whole_requests_keep_to_their_own_rows`: with one stream row free and a whole request waiting for a whole-request row, a new stream takes the stream row on the stateful decoder, and the whole request finishes on a row another whole request returns. With one shared set of rows, the stream fell back to the left-context path and the whole request waited for rows streams held.
- `test_qwen3_tts_whole_requests_decode_on_reserved_rows_beside_a_stream` (accelerator lane, a small real codec through the vocoder factory, captured window and tail graphs, 2 stream rows and 2 whole-request rows): a live stream decodes on a stream row while three whole requests share the two whole-request rows, one waiting for a row another returns. Every whole request matches one stateful decode, the stream keeps its row until its abort returns it, and every row comes back. It reads the rows the factory, scheduler and arena set up, so a whole-request row list built from stream rows fails it.
- `test_real_base_whole_requests_on_reserved_slots_match_one_stateful_decode` (real 1.7B Base checkpoint, CUDA): the same flow and checks on the real codec.
