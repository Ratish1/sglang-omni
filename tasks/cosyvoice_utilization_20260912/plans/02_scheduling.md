# Acoustic scheduling and batch formation

Open PRs #2086 and #2110 both change pending-message FIFO handling in the streaming scheduler. Reconcile their different implementations and request-order behavior before changing coalescing or admission. These are prerequisite correctness investigations, not implementations of the wait controls in S1. See [report 17](../reports/17_open_prs.md).

Owners: `FunCosyVoice3StreamingVocoderScheduler`, generic streaming scheduler state/lifecycle, buffered admission, `adaptive_flow_requests_grouping`, config/factory. Evidence: reports 02, 03, 06, 12.

Buffered and streaming policies are separate. Buffered admission is bounded by nominal mel cost/size/wait, then splits into Flow groups and independent HiFT groups. Streaming groups by first-hop key or `(hop, offset)` and has a hardcoded 30 ms peer wait independent of factory `max_batch_wait_ms`; it does not apply the buffered cost cap. Changing one wait flag cannot be assumed to affect both paths.

## S1: expose and bound the existing streaming policy

Trigger: measured singleton fragmentation or peer-wait gaps with eligible peers. First PR adds explicit factory/config fields for first-hop and follow-up wait windows, retaining 30 ms defaults, and documents which stage each controls. Preserve the current early return when no peer can join; consume ingress/abort messages through the existing path. This yields a controlled experiment, not an automatic reduction of all waits.

Select one policy at a time from actual arrivals: shorten a wait if its added delay exceeds batching benefit; coalesce if larger realized batches improve per-audio cost within TTFA/continuity budgets. Keep first-hop priority, one-hop-per-step fairness, same readiness/lookahead and finalization semantics. Do not fix “batch size” by padding more requests or blocking on peers that have no compatible hop/offset.

A separate PR may enforce a streaming cost budget once measured shapes show memory or long-prefix monopolization. Cost must use **actual cumulative prompt+generated prefix**, CFG factor and padded maximum, not only new hop length. Retain one oversized singleton progress path and ensure deferred requests remain selectable without token loss. Add bounded aging only if the trace demonstrates starvation; define priority and ready-time accounting explicitly.

For buffered mode, existing grouping DP already minimizes group count under its constraints. Tune admission/merge thresholds before replacing that algorithm. Record useful frames, padded frames, selected group size, queue residence, and resulting HiFT partition; one metric for admitted batch size is insufficient.

Remote proof: c1 no-peer behavior; c16 mixed short/long target and reference lengths; staggered compatible peers; first-hop arrivals during follow-up work; final-done overtaking queued chunks; abort during a peer wait; oversized singleton and cost-limit cases. Compare complete token/audio coverage and final count, TTFA, tail latency, playback continuity, throughput and peak memory. Batch composition can change floating arithmetic and sampling timing; use frozen acoustic inputs for parity and full SeedTTS for acceptance. Each field/policy is independently reversible; no SGLang scheduler replacement is needed.
