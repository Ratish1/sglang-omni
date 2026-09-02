# PR draft for perf/scheduler-observed-reservation (9769867a0)

Title: [Scheduler] Reserve KV by the largest recent finished output instead of sglang's guess

## Motivation

SGLang admits a request only while the KV pool holds, for every running
row, min(max_new_tokens, 4096) times new_token_ratio, its guess at how much
of the cap an output will use. The guess starts at 0.7, decays to 0.098
over 600 decode steps, and returns to 0.7 at every idle iteration. Omni's
stages set caps far above their outputs: the Qwen3-Omni talker caps at
4096 frames and a voice clone answer is about 45 frames. On a starved pool
the guess is the limit. The bf16 colocated H100 profile (the CI's TTS
stage) leaves the talker 20136 KV tokens after its bf16 weights, and 0.7
times 4096 per row admits 6 of the 16 concurrent requests. The rest wait a
second each for a finish.

## Change

The scheduler measures the fraction instead of guessing it.
RecentFinishedOutputTracker (sglang_omni/scheduling/finished_output_tracker.py)
keeps, for one cohort of max_running_requests finishes, the fraction of the
clipped cap each finished request used, and OmniScheduler pushes the
largest into SGLang's tracker before every admission. Two hook lines in
omni_scheduler.py, one at the top of get_new_batch_prefill and one at
request finish in stream_output. It has the shape of SGLang's own
RecentPrefillBatchSizeTracker: a window, a max, pushed at the admission
site.

- Until the first request finishes SGLang's guess stays in place.
- The fraction divides by min(cap, CLIP_MAX_NEW_TOKENS), the same clip
  the PrefillAdder applies, and an output past the clip counts as 1.0, so
  the ratio stays within (0, 1] as SGLang keeps it.
- The candidate's own reservation, the retract path and every cap are
  untouched. A request longer than the last cohort's maximum takes
  SGLang's retract like any other overrun.
- The window is one cohort, so an outlier leaves after one turnover of
  the batch.

Nothing is hard coded: the clip is SGLang's constant and the window is
the engine's max_running_requests.

## Where it changes anything

Only where a stage's pool is smaller than rows times 0.7 times the
clipped cap plus the prompts. Where the pool is larger the row limit
binds first and both versions admit the same rows.

| profile | talker pool, tokens | rows at which the guess starts binding | at the default 32 rows |
|---|---|---|---|
| H100 bf16 colocated (CI TTS stage) | 20136 | 6 | binds, the gain below |
| H100 fp8 colocated | 120769 | 38 | no change |
| H200 bf16 colocated | about 223k derived | about 72 | no change |

The same rule covers every other stage the scheduler serves (the per
stage bound is rows times min(cap, 4096) at ratio 1.0, tasks doc 09
section 3). Under continuous saturated traffic SGLang's decay reaches its
floor after 600 steps and the two versions converge. The difference lives
in the first 600 steps after every idle gap.

## Measurements

Qwen3-Omni bf16 colocated H100, voice clone (SeedTTS 50), one worker,
base 68c88dae6 against the branch:

| concurrency | qps | latency p50 s | talker running max | admission wait p50 ms |
|---|---|---|---|---|
| c1 | 2.237 to 2.214 | 0.422 to 0.428 | 1 | |
| c16 | 6.737 to 8.387 | 2.297 to 1.812 | 9 to 16 | 935 to 4 |
| c32 | 7.595 to 10.852 | 3.712 to 2.391 | 9 to 32 | 1533 to 314 |

WER 0.009 to 0.016 and UTMOS 4.43 to 4.47 on both sides at every
concurrency. Same sample similarity of the base's wav against the
branch's wav scores 82 to 84, the run to run baseline of the branch
against itself is 84. Forced talker retractions (SGLANG_TEST_RETRACT on
the talker stage only, 9 retractions) leave WER and UTMOS in the same
band and the replay path runs as SGLang's.

Neutral where the pool does not bind: fp8 colocated H100 with voice clone
at c1, c16 and c32, MMSU (2000 clips) and Video-MME with the talker at
c16 (same completed counts, same accuracy, no retraction, qps within run
variance). Video-AMME on fp8 at c16, accuracy 33 and 32 of 50 with one
flip under different batch composition. MiniMax Music 3 at c16 with seeded
requests, 16 of 16 in both arms, five outputs byte identical, no
retraction.

## Tests

tests/unit_test/scheduling/test_observed_output_reservation.py: the
tracker's fractions (largest recent, clipped cap, past the clip, window
turnover, requests without a cap) and the admission push (guess kept
until a finish, observed value pushed before upstream admission, SGLang's
writes between admissions overwritten). The coalesce gate stub carries
the tracker.
