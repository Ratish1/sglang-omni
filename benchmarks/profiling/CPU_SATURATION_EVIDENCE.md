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

The non-injected Nsight pair must distinguish, per semantic thread:

- on-CPU work and native sampled stacks;
- runnable-but-off-CPU delay versus blocked dependency time;
- migrations and simultaneous activity among builder, encoder, and scheduler.

That selects one narrow follow-up. Request-level ordering, Python-line
ownership, and CPU-to-CUDA launch correlation require targeted semantic
telemetry after the owner is selected; they cannot be inferred from aggregate
Nsight tables.

No production repair is justified until a candidate removes the selected
mechanism in an unprofiled matched A/B while preserving output, admission,
latency, and lifecycle integrity.
