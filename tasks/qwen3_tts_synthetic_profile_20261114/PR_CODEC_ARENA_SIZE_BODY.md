## Problem

The streaming vocoder keeps each stream's incremental codec state in an arena row, from the stream's first decode to its last. The arena had 64 stream rows, the engine's default running bound, and the startup check compared the rows against that bound.

The engine bound does not limit the streams the vocoder holds. A stream's last decode comes after its engine slot frees, so under load the vocoder holds the 64 running streams plus every stream still decoding its tail, up to the 128 requests the coordinator admits (64 running plus 64 queued).

Every stream past the 64th rows took the left-context decoder, which costs more per chunk. The vocoder fell further behind, more streams stayed live past their generation, and more took the fallback. Past about 60 req/s of Poisson arrivals (1.7B CustomVoice), almost every request underran and the rest got HTTP 503 at the in-flight cap.

## Change

`Qwen3TTSPipelineConfig.stage_factory_kwargs("vocoder")` derives the streams a vocoder replica can hold at once, the admitted requests that reach it: `max_running_requests` plus `max_queued_requests` (from the engine overrides, or the admission defaults the coordinator's cap reads), times the engine replicas when the vocoder runs in its own process. It passes that as `codec_state_slots`, so an operator's `--vocoder.factory.codec_state_slots` still wins, and to the startup check, which now warns when the slots are below it. With the defaults the arena holds 128 stream rows and 128 for whole requests, 701 MiB instead of 350 MiB.

## Results

One H100 pair, main against this branch at the same time, one server per card, each pinned to its card's half of the socket. Nari Labs' open client unmodified (`tts_bench` `a2d65e1`): 1.7B CustomVoice, Ryan, English, the SeedTTS EN sentences, Poisson arrivals, 30 s warm-up and 120 s measured.

| req/s | completed, main → branch | requests with underruns | first byte p50, ms |
|---|---|---|---|
| 60 | 1.1 % → 100 % | 2,284 → 4 | 52 → 21 |
| 70 | 85.6 % → 100 % | 13 → 5 | 999 → 67 |
| 80 | 4.0 % → 91.5 % | 2,229 → 0 | 62 → 799 |

At 80 req/s main's first byte covers only the 4 % it completed; past capacity the branch serves at capacity and the excess gets HTTP 503 at the in-flight cap. No stream took the left-context decoder (main: 1,614 times at 60 req/s).

Below the knee every stream fits either arena, so normal load takes main's path with more rows free. SeedTTS EN, 1.7B Base, the full 1,088 prompts, unseeded, both card orders:

| cell | first audio mean / median / p95, ms, main → branch | RTF |
|---|---|---|
| streaming c1 | 18.8 / 20.6 / 22.8 → 18.85 / 20.55 / 23.15 | -0.2 % |
| streaming c16 | 30.6 / 29.55 / 43.8 → 30.5 / 29.4 / 43.9 | -0.3 % |
| streaming c64, three warm passes | 54.25 / 41.27 / 169.3 → 54.45 / 41.7 / 153.7 | -0.4 % |
| whole c1, c16 | | -0.4 %, +0.2 % |

WER 0.95 / 1.67 → 1.04 / 1.09 % (main's second order has one sample over 50 %), SIM 71.27 / 71.32 → 71.57 / 71.41.

## Tests

- `test_codec_arena_holds_every_request_the_coordinator_admits`: the default arena equals the coordinator's in-flight cap.
- `test_vocoder_receives_the_streams_one_replica_may_hold`: the derived slots for running and queued overrides, engine and vocoder replicas.
