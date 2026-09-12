# Optional profiler-control hardening

Open PRs #1995 and #1216 already contain the same rank-initialization patch. PR #1304 also covers that fix plus process ownership and scheduler-thread start/stop, while retaining unacknowledged control. Reconcile/reuse those changes rather than submitting another rank or ownership implementation; acknowledgments and configuration forwarding remain separate uncovered scopes. See [report 17](../reports/17_open_prs.md).

This is independent of the initial NSYS diagnostic slice. Source: reports 08 and 13. Promote it only if reusable native Torch profiling is required; do not block complete process-tree NSYS capture on it.

Omni currently broadcasts start/stop over PUSH with no acknowledgment. The request config is dropped, and colocated stages share a process-local TorchProfiler singleton. A direct repeated `TorchProfiler.start` can use `rank` before assignment; normal stage handlers skip starts while active, so the direct-call defect is not established as a default serving bottleneck. Trace-export callback and explicit stop can both request export; actual double-export behavior depends on the pinned Torch callback lifecycle and must be reproduced.

Separate PRs:

1. Fix rank initialization and make start/stop/export ownership explicit and idempotent for one process. Preserve process-level sharing across stage handlers; one run owns one export and completion notification. Validate callback behavior against the installed Torch source/version before removing either export path.
2. Add request IDs and per-process acknowledgments to control. Deduplicate colocated stages by profiler owner, aggregate every start/stop/error, expose accepted versus ready versus exported states, and reject incompatible concurrent runs. Upstream SGLang's returned-control-result path is useful precedent, but its tokenizer checks only element zero; do not reuse that as aggregate-success logic.
3. Forward a validated profiler configuration and select bounded activities/shape/stack options. Define which scheduler threads are actually captured by Kineto and qualify CPU cross-thread coverage. Preserve an independent process-tree NSYS route for full CUDA/API timing.

Remote gates: repeated start, incompatible run, wrong-run stop, worker failure, two colocated stage handlers, export failure, asynchronous gzip completion, all stage processes and one-process topology. Control-plane success must mean the documented state, not “message sent.” Existing profile event JSONL uses a wall clock; cross-tool alignment still requires a documented shared clock/anchor. These lifecycle fixes are observability work, not a claimed SM-utilization optimization.
