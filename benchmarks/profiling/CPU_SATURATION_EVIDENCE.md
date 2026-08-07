# Fun-ASR CPU saturation: evidence boundary

This is the compact, living record for the H100 investigation. Percentages are
screening estimates unless the row says matched.

## Proven

- Unbound CPU64 reduces throughput by 35–46%, increases migrations and runnable
  delay, and lowers GPU utilization while GPU clocks stay fixed.
- Protecting the server's GPU-local physical cores reduces the matched remote
  CPU64 cost to 5.6%. The large collapse is therefore primarily logical-CPU
  scheduling competition, not GPU throttling, request rejection, or batch
  collapse.
- SMT-sibling pressure produces a smaller 6.9% matched loss with negligible
  direct runqueue delay. SMT/shared execution resources and effective frequency
  are secondary mechanisms.
- The affected host path is distributed: `fun-asr-audio-e`, `sched-asr`, and
  `omni-request-bu` all gain about 5–5.6 CPU ms/request under CPU64. No single
  thread has been shown to own the whole regression.
- Fresh quiet runs can alternate between roughly 12–18 and 42–59 QPS despite
  completing every request. This bimodality is now evidence to capture, not a
  precondition that must disappear before profiling.
- The first bounded non-injected Nsight pair completed with valid request,
  system, and capture integrity, but its quiet and CPU64 control drift was
  36.49% and 8.48%. Its exact throughput and per-owner cross-arm deltas are
  therefore not causal effect estimates. Directionally, on-CPU service rose
  across scheduler, request builder, and audio encoder, most strongly in the
  request builder.
- At overload, the checked-in 16-entry request-build backlog is an independent
  admission boundary. It is not the cause of accepted-request slowdown below
  that boundary.

## Architectural facts and live hypotheses

Fun-ASR performs CPU request building, waits on one batched pre-LM audio-encoder
service, then admits LM-ready work to one Omni scheduler. Completed request
builds are drained in submission order, so a slow head can hold ready followers.

These facts make three hypotheses credible but not yet proven:

1. OS preemption/migration delays several serial host-dispatch stages.
2. Extra CPU service comes from cache/SMT effects in audio preparation, Python,
   allocation/copying, or synchronization.
3. Ordered request-build drain or the single pre-LM service amplifies variance
   into head-of-line delay before LM admission.

It is **not proven** that scheduler-loop computation is the root cause, that
speech needs a separate LM token scheduler, or that the router is required to
reproduce the single-server mechanism.

## Remaining decision

The bounded Nsight pair selected a distributed host path rather than one sole
owner. Its export did not contain usable sched-out state/block metadata, so it
cannot distinguish runnable starvation from blocked waiting. Nsight CPU
changes are also not exact kernel migrations; procfs `se.nr_migrations` is the
authoritative counter.

The next non-redundant test is low-overhead aggregate semantic telemetry around
the distributed pipeline:

- request-build submissions, completions, ordered drains, and ready-to-drain
  delay;
- head-of-line episode count/duration and the number of later-ready followers;
- pre-LM enqueue, batch publish, completion, and scheduler admission delay;
- scheduler-loop drain budget, work count, and elapsed time.

Collect fixed-window counters/histograms rather than per-request event traces.
This directly tests whether ordered drain or the single pre-LM boundary
amplifies OS interference. Another affinity screen would only repeat the
already-proven mitigation.

No production repair is justified until a candidate removes the selected
mechanism in an unprofiled matched A/B while preserving output, admission,
latency, and lifecycle integrity.
