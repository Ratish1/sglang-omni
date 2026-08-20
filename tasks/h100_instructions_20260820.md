# H100 session instructions (2026-08-20)

Everything needed rides on this branch under tasks/. Full protocol, cells,
and gates: tasks/h100_test_plan_20260820.md. Benchmark client and corpus
protocol: tasks/prefill_launch_first_impl_log_20260817.md section 9.

## Arms on the H100

- Arm A (baseline): git checkout 24e9a3552. This is the base of this
  branch, so A and B differ by exactly the three launch-first commits.
- Arm B: git checkout prefill-launch-first.
- Arm B+C (phase 3 only): from arm B, apply the two coalesce-removal
  commits shipped here as patch files:
    git checkout -b b-plus-c prefill-launch-first
    git am -3 tasks/0001-stamp-wait_queue_entry_time-at-enqueue.patch \
              tasks/0002-remove-prefill-coalescing.patch
  The patches were cut on 99388dca4, a nearby base; -3 resolves the drift.
  Fallback if am conflicts get noisy: stay on arm B and zero the knobs by
  CLI with --prefill-coalesce-requests 0 on the launched model.

## Order

Run phases strictly in order per tasks/h100_test_plan_20260820.md:

1. Phase 1 diagnosis, no gates: E-B3 (make_sync_probe.py, Qwen3-ASR c32,
   arms A and B), E-B4a (probe_host_launch_first.py, ArkASR c16, A and B),
   E-B4b (make_loop_counters.py, ArkASR c16, A and B).
2. Phase 2: code B4 from the phase 1 findings, re-measure ArkASR c16,
   Whisper c2, Qwen3-ASR c16, Qwen3-ASR c32.
3. Phase 3: the knob-deletion matrix, arm A vs arm B+B4+C.
4. Phase 4: correctness sweeps (c64 disconnect burst, strict-mem-check
   tail run, optional nsys).

## Probe script hygiene

make_sync_probe.py, make_loop_counters.py, and probe_host_launch_first.py
each patch sglang_omni/scheduling/omni_scheduler.py in place. Apply one at
a time, revert with git checkout -- sglang_omni/scheduling/omni_scheduler.py
before the next, and never commit their output. Per cell record req/s,
p50/p95/p99 request latency, TTFT when the client reports it, and WER.
