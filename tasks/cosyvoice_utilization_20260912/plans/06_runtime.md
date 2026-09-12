# Placement, memory and transport

Owners: process planning/stage workers/local dispatch, `SGLModelRunner` KV budget configurator, `stage_kv_budget`, admission/abort and output transport. Evidence: reports 04–07, 10, 13.

Default stage edges are local Python references. No relay optimization is justified by the existence of SHM/CUDA-IPC code. Capture actual process IDs, CUDA contexts/streams, CPU runnable/waiting state and kernel overlap before changing placement.

R1 memory experiment: measure model/ONNX/KV/graph/vocoder allocations in startup order and peak acoustic workspace at representative longest prefixes. The current 0.85 static fraction is resolved before vocoder load; it is not a complete shared-GPU budget. Use existing explicit stage KV-byte contracts where a budget can be derived. Ensure enough request rows/tokens for c16 and intended max length; no silent shrink, retraction storm or OOM fallback. Graph buckets must fit alongside peak Flow/HiFT memory, not just idle model footprints. This can be a deployment/config PR with no allocator replacement.

R2 placement experiment: keep preprocessing+AR together; compare a separate vocoder process only if shared-interpreter or CUDA-context/stream contention is evidenced. Account for new transfer copies, context scheduling and memory replication. MPS, extra streams or process replicas are separate hypotheses, not required ingredients. A GPU normally runs distinct CUDA streams subject to dependencies and resources; a Python thread count is not a GPU overlap guarantee.

Open [PR #1933](https://github.com/sgl-project/sglang-omni/pull/1933) already moves the Flow vocoder to its own process. Reconcile and measure that head using [the dated overlap audit](../reports/17_open_prs.md) before writing another placement change. The independent stage-memory assessment still applies, including the changed startup/transfer behavior of a separated worker.

If placement wins, expose the topology/config and retain the existing transfer/lifetime protocol. State ownership spans stream-before-payload ingress, first-chunk conditioning, final stream_done then payload, abort, relay ACK/cleanup and coordinator terminal delivery. Prove these transitions with H100 request lifecycles and output hashes. Do not combine process separation with a new payload schema and scheduler rewrite in one PR.

Unbounded queues/backpressure are a separate operational contract. Add global in-flight or stage cost admission only if measured queue/memory growth or overload behavior requires it; define rejection scope, fairness, abort and completion cleanup before implementation. The user-reported c16 SM number by itself does not prove an overload issue. Roll back placement/config independently and restore source-default transport selection.
