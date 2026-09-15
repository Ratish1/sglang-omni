# Streaming scheduler follow ups after #2169, #2170, #2171

Written 2026-09-14. Gates first, then the fixes in the order they ship, then the measurements that
decide each. Every fix here is an open row of MECHANICS.md; this file is the design side.

## Gates that every fix below must pass

1. Continuity is 100 percent for every streaming model at the concurrency the model is served
   at: C50 and C100 at 100.0 on the full corpus, zero failures. A fix that buys first audio or
   throughput with a continuity point is withdrawn.
2. First audio, req/s, audio s/s and RTF within 2 percent of the previous B, or better.
3. A and B are different trees. #2169 was squash merged as 442e559b4 at 23:33 IST on
   2026-09-14, and `git diff perf/cosyvoice3-vocoder-serving-loop upstream/main` is empty. An "A
   upstream main" booted from a checkout fetched after that time is the PR 1 tree; its pair with
   the PR 1 head measures run to run noise, not the PR. Record `head.txt` of both arms.
4. One boot per arm, same day, other GPUs recorded; a delta inside 2 percent is repeated once
   before it is believed.

## The fixes, in order

### 1. Capacity of the packed step (PR 2 follow up, origin: launch bound)

- Origin: `solve_flow_euler_packed`, 10 Euler steps by 22 blocks, about 18,100 launches per Flow
  call at a 280 ms host floor; a 16 row step is about 0.6 s, so the vocoder serves about 27 audio
  seconds per second (16 rows by hop 25 tokens by 40 ms per token per 0.6 s).
- Fix: CUDA graphs for the packed step keyed by total tokens after packing, the same key the
  layout varies in. Capture keys are derived from the token bucket, never from a table of
  observed (batch, frames) pairs (config.py FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES
  is a symptom of the padded layout and goes away with it).
- Evidence that decides: step wall at 16 rows before and after; audio s/s at c16; first audio at
  c16, which is the step in flight a first hop waits behind.
- Numerics: replay against the eager packed call, bit identical (E2 form).

### 2. One axis ranking with waiting time (F4, origin: ordering)

- Where: `select_step_participants`, models/fun_cosyvoice3/streaming_vocoder.py, PR 2 head
  `ranked = started + unstarted`.
- What it does today: every started hop, whatever its buffer, ranks before every new stream.
  Under 16 runnable streams this decides nothing. Over 16, a new stream gets its first hop only when
  a started stream finishes: first audio equals another request's length.
- PR 1's rule was one axis with an unstarted stream at slack 0.0, so a new stream went ahead of
  every started stream with buffer and behind every underrunning one. PR 2 split the axis to hold
  continuity at c16 and paid first audio (0.91 to 1.83 s mean).
- Design: one axis. A started stream's slack is buffered audio minus wall time since first emit.
  An unstarted stream's slack is minus the time since it became runnable. One second of waiting for
  first audio and one second of silence mid stream are the same second of silence. No constant.
- Risk against gate 1: at c16 a new stream now goes ahead of a started stream with positive slack,
  which is the behaviour PR 2 moved away from. The measurement decides; if C50 drops at c16 the rule
  is not shipped and fix 1 alone carries first audio.
- Evidence that decides: CosyVoice streaming c16 and c32, stack head against stack head plus this
  rule: C50, C100, failures, first audio p50 p95 p99, req/s, audio s/s.

### 3. dots.tts census (F11 verification)

- MOSS-TTS Local at c16 on the full corpus is measured as a pair, 1f6b6843e against 442e559b4, with no
  regression (readout 02 of `streaming_scheduler_first_audio_20260915`). dots.tts is the other
  coalescing scheduler (batch 4, distinct requests). One streaming c16 pair, main at 1f6b6843e against
  442e559b4.

### 4. Remaining open rows of MECHANICS.md, unchanged

Upsample encoder syncs (2 per call), HiFT full history per hop, preprocessing at c16, runaway
generations (client contract, not ours).

## Measurement matrix

| run | arms | point | reads |
|---|---|---|---|
| CosyVoice c32 baseline | 1f6b6843e vs stack head fd088faca | streaming c32 | first audio p50 p95 p99, C50, C100, failures, req/s |
| CosyVoice fix 1 | stack head vs fix 1 head | streaming c16, c32 | step wall, audio s/s, first audio, C50 |
| CosyVoice fix 2 | fix 1 head vs fix 2 head | streaming c16, c32 | C50, C100, first audio, req/s |
| dots.tts | 1f6b6843e vs 442e559b4 | streaming c16 | req/s, first audio, C50, C100, failures |
| MOSS-TTS Local repeat | 1f6b6843e vs 442e559b4, `head.txt` archived | streaming c16 | measured 2026-09-15, no regression (readout 02) |

## Mechanics of #2169 for the other streaming models (read 2026-09-14)

The base hunks and what each does for a model whose `_has_ready_work` is the base False (every
model but CosyVoice):

| hunk | main | #2169 | effect on MOSS-TTS Local and dots.tts |
|---|---|---|---|
| `start` loop | `_next_message` | `_has_ready_work()` False, so `_next_message` | none |
| `_next_message` | pending, then `inbox.get(0.1)` | `_get_batch_message(timeout=0.1)`: pending, then `inbox.get(0.1)` | none |
| `_collect_new_request_batch` | inbox only; non new messages appended to pending | pending then inbox; deferred restored to the front | none on a streaming run: a streaming payload returns the batch of one before any read |
| `_collect_stream_chunk_batch` | inbox only | pending then inbox | on a streaming run pending holds at most the one chunk the distinct request rule pushed back, read first by both versions; a batch is cut at a non chunk between parked chunks only when the new request collector parked them, which a streaming payload never does |
| `_handle_stream_done` | `on_stream_done` under the lock, messages to the outbox, state cleared | same, through `_complete_stream_request`, None path unused | none; the lock scope is identical |
| `_pump_streams` | inline loop | `_pump_one_step` in a loop, same returns | none |

Two independent whole file reads (MOSS-TTS Local, dots.tts) against these hunks on 2026-09-14
CONFIRMED every row: `_has_ready_work` is overridden only by fun_cosyvoice3 (grep over
sglang_omni), so the loop runs main's body; `is_streaming_payload` returns the new request batch
of one before any read; the only writers of `_pending_messages` on a streaming run are the two
`appendleft` sites of the chunk collector, each followed by `break`, so the deque holds at most one
message and is empty on every collector entry; `on_stream_done` never returns None for either model
and `_complete_stream_request` re-enters the same RLock on the same thread; the pump runs the same
select, build, run sequence. Neither model's tests changed. One latent item, unreachable: the pump
would loop again if `on_step_failure` returned an empty list; the base returns one id per
participant and neither model overrides it.

Consequence for the MOSS-TTS Local c16 result of 2026-09-14 (req/s 12.945 to 12.425, first audio
p95 0.380 to 0.555 s, C50 100 to 99.17, RTF p99 0.392 to 0.469): no hunk can move these on the
MOSS path. Measured on 2026-09-15 as a pair on one GPU, `head.txt` 1f6b6843e against 442e559b4
(`../../streaming_scheduler_first_audio_20260915/readouts/02_moss_tts_local_paired_ab_c16_20260915.md`):
B read 13.08 req/s against A's 11.49, first audio p95 0.351 against 0.377 s, and the vocoder's own
first audio p95 61.4 against 61.5 ms. The same tree read 11.49 and 13.28 req/s on two boots, so the
2026-09-14 gap is inside the run to run spread of identical code.
