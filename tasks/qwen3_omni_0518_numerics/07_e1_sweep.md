# 07. E1 sweep, talker admission cap (2026-09-01)

Bundle artifacts/e1-sweep.tar.gz. Host: one H100 80GB HBM3 (driver
580.126.20), GPU 0, bf16 colocated config
(examples/configs/qwen3_omni_colocated_h100_bf16.yaml), voice clone
(seed-tts-eval-50-arrow en, 50 requests per bench plus warmup). Arms
interleaved base, e1, base, e1, base, e1, a fresh server boot per arm and
repeat, the GPU checked idle before every boot (0 to 2 percent, 0 MiB).
Base is upstream/main e7d876b28, e1 is perf/talker-admission-cap
112b7bf0c (one commit on e7d876b28). Unit tests per arm:
tests/unit_test/qwen3_omni 682 passed on base, 686 on e1 (the four new
cap tests), 2 skipped on both.

## 1. Bench, medians over three boots (range in brackets)

| conc | metric | base | e1 | change |
|---|---|---|---|---|
| c1 | qps | 2.191 [2.160, 2.254] | 2.267 [2.208, 2.316] | +3.5% |
| c1 | latency p50 s | 0.425 [0.409, 0.442] | 0.409 [0.405, 0.452] | -3.8% |
| c1 | RTF mean | 0.1364 [0.1356, 0.1391] | 0.1347 [0.1283, 0.1370] | -1.2% |
| c16 | qps | 6.626 [6.533, 6.631] | 8.239 [8.102, 8.288] | +24.3% |
| c16 | latency mean s | 2.268 [2.265, 2.289] | 1.830 [1.818, 1.854] | -19.3% |
| c16 | latency p50 s | 2.371 [2.277, 2.410] | 1.832 [1.792, 1.841] | -22.7% |
| c16 | latency p95 s | 3.520 [3.500, 3.662] | 2.906 [2.884, 2.954] | -17.4% |
| c16 | RTF mean | 0.6886 [0.6630, 0.6912] | 0.5427 [0.5419, 0.5569] | -21.2% |
| c16 | RTF p95 | 1.017 [0.957, 1.098] | 0.714 [0.698, 0.730] | -29.8% |
| c16 | audio s per s | 23.55 [23.15, 23.71] | 28.93 [28.75, 29.04] | +22.9% |
| c32 | qps | 7.404 [7.306, 7.608] | 10.963 [10.805, 11.366] | +48.1% |
| c32 | latency mean s | 3.543 [3.500, 3.616] | 2.398 [2.379, 2.521] | -32.3% |
| c32 | latency p50 s | 3.708 [3.629, 3.819] | 2.318 [2.271, 2.323] | -37.5% |
| c32 | latency p95 s | 5.222 [5.220, 5.531] | 3.557 [3.422, 3.792] | -31.9% |
| c32 | RTF mean | 1.117 [1.116, 1.135] | 0.730 [0.728, 0.749] | -34.7% |
| c32 | RTF p95 | 1.976 [1.921, 2.199] | 0.995 [0.946, 1.105] | -49.7% |
| c32 | audio s per s | 26.44 [26.24, 26.92] | 39.39 [38.51, 39.67] | +49.0% |

At c16 and c32 no e1 boot overlaps any base boot on qps, latency mean,
latency p50 or RTF mean. At c1 every difference is inside the overlap of
the two ranges, as expected: the cap binds on no request.

## 2. Mechanism, from the request profiler events (medians over boots)

| conc | quantity | base | e1 |
|---|---|---|---|
| c16 | talker running, max | 9 | 16 |
| c16 | talker running, time weighted mean | 6.1 | 10.3 |
| c16 | talker admission wait p50 ms | 928 | 4 |
| c16 | talker admission wait p95 ms | 1428 to 1541 | 27 to 51 |
| c16 | talker generation p50 ms | 769 | 809 |
| c16 | thinker generation p50 ms | 194 | 233 |
| c16 | admission to first audio p50 ms | 1693 | 894 |
| c16 | admission to complete p50 ms | 2195 | 1802 |
| c32 | talker running, max | 9 | 32 |
| c32 | talker running, time weighted mean | 6.1 | 16.6 |
| c32 | talker admission wait p50 ms | 1535 | 10 |
| c32 | talker admission wait p95 ms | 3162 to 3307 | 69 to 79 |
| c32 | talker generation p50 ms | 485 | 694 |
| c32 | thinker generation p50 ms | 193 | 241 |
| c32 | admission to first audio p50 ms | 2196 | 1174 |
| c32 | admission to complete p50 ms | 2562 | 1715 |

The base talker never exceeds 9 running at either concurrency, which is
the 06 section 4 arithmetic for a 4096 token reservation in a 21373
token pool. With the cap (64 plus 32 per thinker text token, 224 to 992
here) the talker runs the full concurrency and the admission wait is
gone. Per request generation is slower with more rows in the batch
(talker plus 5 percent at c16, plus 43 percent at c32, thinker plus 20
to 24 percent because the GPU is shared with a talker that is now busy
two to three times as often), and the request still finishes 18 to 33
percent sooner because the second in the queue is gone.

## 3. Checks

- Cap never binds: over the 450 e1 requests the largest ratio of
  emitted frames (audio seconds times 12.5) to the request's cap is
  0.18.
- Outputs at c1 are not byte comparable between two boots with this
  bench: a request without an explicit seed draws its talker seed from
  its request id (request_builders.py `_build_talker_request_data`), so
  the per sample audio lengths in the base1_c1 to e1_1_c1 compare differ
  on both arms alike (latency ratio per sample median 1.013, audio
  seconds 171.4 against 169.6). A codes comparison needs the bench to
  pass a fixed seed per sample (run_bench.py sets seed None), which is a
  tooling item before E2's proof.
- No failed requests in 18 benches, no server errors (the only ERROR
  line is the nixl import notice).

## 4. What this means for the plan

E1 is accepted on the bf16 colocated config. The remaining c16 latency
(1.83 s mean for 3.4 s of audio) is now the thinker generation (233 ms),
the talker generation (809 ms at bs up to 16) and the code2wav first
audio delay, which is where E2 (step host syncs), E4 (prefill and
request build) and the thinker side items apply. The fp8 colocated
config (talker fraction 0.12, so a larger talker pool and a different
base cap) has not been measured, and its earlier boots on that host
died at image_encoder startup with an external SIGKILL, which needs a
host side look first.

## 5. Redesign after review (2026-09-01)

The measured branch lowered `max_new_tokens` itself, which changes the
model's official stop (4096, the HF `talker_max_new_tokens` default) with a
constant fitted on one dataset. That is the wrong seam: the stop was never
the problem, the reservation was. vLLM-Omni keeps the talker at
`max_tokens: 4096` (vllm_omni/deploy/qwen3_omni_moe.yaml:49) and its
scheduler never reserves for it: `max_tokens` appears in vllm/v1/core/sched/
scheduler.py only as the stop check (:489-492), blocks are allocated for the
tokens being generated (`allocate_slots(request, num_new_tokens, ...)`,
kv_cache_manager.py:344) and a request is preempted when blocks run out
(scheduler.py:569-593). SGLang reserves `min(max_new_tokens, 4096) x
new_token_ratio` and retracts when the reservation was too small, so the
equivalent is a small reservation with the official ceiling intact.

The branch now reverts the cap (50c6d424c) and sets
`schedule_conservativeness` 0 for the talker in
`configure_talker_server_args` (48bbc40fa, after a 0.1 step at f74c4fcda).
SGLang uses that argument in one place, the initial `new_token_ratio`
(0.7 x conservativeness, new_token_ratio_tracker.py:20-32). At 0 the
reservation for running rows is off: a request is admitted while its own
ceiling fits the free pool (the candidate charge stays min(max_new_tokens,
4096) unscaled, schedule_policy.py:1219-1237), running rows hold only what
they use, and the pool fills to `max_running_requests`. When the pool does
fill, `check_decode_mem` fails and `retract_decode` releases the youngest
rows back to the waiting queue (scheduler.py:3491-3526,
schedule_batch.py:2816-2865), which the talker supports: retracted rows are
replayed from `_decode_input_history` (talker_model_runner.py:328-340) and
the base runner skips retracted rows and resets penalty state on a shrunk
`output_ids` (base.py:363-368, 895-933). This is vLLM's policy for the same
talker (allocate per generated token, preempt on exhaustion) expressed
through SGLang's knob, with no fitted constant.

The admission call chain, so the seam is not in doubt:
`QwenTalkerScheduler.get_next_batch_to_run` (talker_scheduler.py:110-137,
readiness only) -> `OmniScheduler.get_next_batch_to_run`
(omni_scheduler.py:1317-1329, hands running_batch to upstream) ->
`Scheduler.get_next_batch_to_run` (sglang scheduler.py:3015-3149, prefill
wins over decode at :3121-3131) -> `OmniScheduler.get_new_batch_prefill`
(omni_scheduler.py:1331-1352, only coalescing, then `_Upstream`) ->
`_get_new_batch_prefill_raw` builds `PrefillAdder(..., self.new_token_ratio_tracker.current, ...)`
(scheduler.py:3257-3262) -> `add_one_req` (schedule_policy.py:1219-1237).
The tracker is built once from `schedule_conservativeness`
(scheduler.py:1255, new_token_ratio_tracker.py:20-32), decays per decode
step (:3554) and is reset on idle (:4074). Omni owns none of the policy,
it owns the server args the policy reads, which is where the change is.

What the measurement covered and did not: the E0 profile and the sweep used
the repository's SeedTTS voice clone benchmark (run_bench.py seedtts-vc,
benchmark_omni_seedtts, 50 requests, warmup), not raw requests, on one
workload: 5 to 29 thinker text tokens, 1 to 7 s of audio. The reservation
arithmetic does not depend on the workload but the size of the gain does,
so the sweep on the reservation branch adds a long output arm (texts of 60
to 120 tokens, 20 to 40 s of audio, where the reservation is closer to the
truth and the gain smaller) and, once the fp8 boot is fixed, the video to
speech workload on the fp8 colocated config.

The mechanism is the same arithmetic the cap exercised, so the sweep above
is the prediction for the new branch, not its proof. To do on f74c4fcda:

- the same three boot interleaved sweep at c1, c16 and c32, plus WER,
  speaker similarity and UTMOS on the outputs of both arms, and
- one boot with the talker pool shrunk (`--talker_ar.engine.max_total_tokens
  4096`) at c32 to force retracts, reading the retract count from the
  server log and the WER and similarity of that run against the normal one,
  which is the proof that the safety valve keeps the audio intact.
