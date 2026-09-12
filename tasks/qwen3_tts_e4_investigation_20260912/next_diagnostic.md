**Next H100 diagnostic: measure the first-audio dependency before choosing a fix**

The minimum missing comparison is **unmodified control versus early IDs**, on one common source base, with the normal colocated layout. E4a/E4b/E4c need not be repeated. After locating that difference, run the combined early-ID/nonblocking stack through the same measurement. A trace containing only the early-ID candidate cannot identify time introduced by its diff.

**Freeze the variants**

This directory includes source-only copies of the reviewed changes:

| Variant | Construction |
|---|---|
| Control | `645b472cdb2d7b93a1825a5bfa6a603b62b03936` |
| Early IDs | Same base + [early_ids.patch](early_ids.patch) |
| Combined, after the first diagnosis | Same base + early-ID patch + [nonblocking.patch](nonblocking.patch) |

The patches were applied in sequence to a disposable Git index against this base; both checks passed. [patch_validation.json](patch_validation.json) contains their hashes. No real checkout was modified. These patches contain the existing PR changes, not an additional proposed fix. The nonblocking patch is the tested product change through `898dc3234`; the later comment/test-wait-only followup is irrelevant to this performance comparison.

On the remote machine, copy this task directory into the repository and create fresh diagnostic worktrees. Example, from the repository root:

```bash
export QWEN_DIAG_REPO="$PWD"
export QWEN_DIAG_PLAN="$PWD/tasks/qwen3_tts_e4_investigation_20260912"
export QWEN_DIAG_BASE=645b472cdb2d7b93a1825a5bfa6a603b62b03936

git worktree add --detach tmp/e5-control "$QWEN_DIAG_BASE"
git worktree add --detach tmp/e5-early "$QWEN_DIAG_BASE"
git -C tmp/e5-early apply --check "$QWEN_DIAG_PLAN/early_ids.patch"
git -C tmp/e5-early apply "$QWEN_DIAG_PLAN/early_ids.patch"
```

Use the same installed dependencies and frozen local checkpoint/dataset as E4. Run with the selected worktree first on `PYTHONPATH` and save the actual imported `sglang_omni.__file__`, SGLang/Torch versions, HEAD, and dirty diff. An editable installation pointing to another checkout can invalidate the entire comparison. Use fresh worktree names if these already exist; do not reset existing user worktrees.

**Preserve the workload and establish an unprofiled difference**

Use the E4 server command with the normal single `pipeline` process, no priority edits, no switch-interval override, no process split, and the same reference-based streaming workload. Enable strict-port mode. Save the full launch command and resolved configuration. The archive lacks a complete executable launch script, so reconstructing missing launch flags from benchmark defaults is unsafe; use the retained remote E4 command and verify the logged server max-running 16 / context 8,192.

Run control → early IDs → control, one server at a time, two full-corpus passes per boot. This small return-to-control comparison detects drift that a single historical control cannot. Keep pass1/pass2 separate. Run on the same available H100 and a consistent CPU allocation; record CPU contention and GPU processes/clocks throughout, not only before launch. Stop each server's children and wait for port/GPU release before starting the next.

Retain default sampling for the first reproduction and record resolved per-request seeds if adding diagnostics. Do not silently enable deterministic inference: that changes concurrency and vocoder graph behavior. Exact seeded numerical checks are a separate comparison.

If the difference disappears, report that the earlier regression is not reproduced under these controlled conditions. Do not manufacture a mitigation. If it persists, keep the same workload/cache preconditioning for the short trace below. Avoid replacing the 1,088-row workload with a repeatedly warmed 192-row workload: the reference cache has 256 entries and would exercise different behavior.

**Capture the CPU/GPU critical path**

Use **Nsight Systems on the server process tree**, with CUDA activity, OS runtime waits, Python sampling, and GIL tracing where supported. Launch before stage child creation so the colocated pipeline process is covered. Record `nsys --version` and the exact collection command. NVIDIA documents Python sampling, GIL tracing, and interactive launch/start/stop collection in its [user guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html).

The collection pattern is:

```text
nsys launch --session-new=tts-early \
  --trace=cuda,nvtx,osrt,python-gil --python-sampling=true \
  <the unchanged E4 server command, with its complete arguments>

# After readiness and the identical preconditioning pass:
nsys start --session=tts-early -o <absolute-output-prefix>

# Run the corresponding streaming window, then stop collection:
nsys stop --session=tts-early
```

Use a distinct session/output name for control. Check the installed version's help for these options. Keep the capture short, roughly 15–25 seconds of the relevant pass, and retain the raw `.nsys-rep`; compilation and startup graph capture should be outside it. If that window fails to reproduce the delay, capture the affected part of the original pass instead of treating a different workload as a disproof. Do not use Nsight-instrumented QPS as final acceptance evidence.

Keep raw request events for joining stage UUIDs. The archive already demonstrates that those events alone omit the critical initial-worker boundary. For this trace, add the same small temporary diagnostic ranges/metadata on both arms, and archive the patch. Collect host timestamps/integers and existing CUDA event identities; do not introduce tensor `.cpu()`, `.item()`, `.tolist()`, or new synchronization into the measured path.

| Location | Required observation | What it separates |
|---|---|---|
| `Qwen3TTSModelRunner._collect_codes` and `post_process_outputs` | Scheduler native thread ID; stream ID; batch/step ID; semantic-ID copy, predictor submission, snapshot and existing readiness-event identity | Earlier CPU publication versus actual producer completion |
| Vocoder `ingest` / `_schedule_initial` | Request ID, enqueue time, first-code event identity, reference-frame count | Time before the initial worker obtains the request |
| `_run_initial_batch` / `_build_incremental_plan` | Worker native thread ID and stream; before/after state-lock acquisition; plan begin/end; latest versus selected chunk/frame counts; readiness-event identity used | GIL/CPU/lock delay and unnecessary dependencies on later frames or other requests |
| `_launch_incremental_group` / `_launch_async` | Cohort request IDs, width, batch size, initial/followup, graph/eager; launch range | Cohort queueing and submitted GPU work |
| `_Qwen3TTSDecodeHandle._wait_and_release` / `_commit_initial` | Output event identity, wait begin/end, commit time | GPU finish/D2H wait versus CPU wakeup/commit delay |
| Scheduler/outbox stream enqueue and runtime first-audio send | Enqueue/dequeue time and request ID | Completed data waiting for host routing |
| Prepared request construction / prefix match | Digest of existing CPU `input_ids_list`, realized reference-encoder batch shape/cache outcome, prefix-hit and extend lengths | Changed input keys/work versus cache/admission history |

CUDA stream/event identities let the GPU trace show whether the initial decode is awaiting producer work or queued behind unrelated work. An occasional existing event `.query()` can record ready/not-ready without a host wait, but does not measure the event's completion timestamp; do not substitute it for the GPU timeline. Capture first-frame requirements separately from the latest-frame event used by the current planner.

Record all five important consumers/producers: talker scheduler, preprocessing executor, reference-code batcher, initial vocoder worker, and both followup workers. A profile of the talker thread alone or a GPU utilization percentage cannot resolve the question. If GIL tracing is unavailable, explicitly leave GIL attribution open.

**Choose the fix from the trace**

| Observation in candidate relative to control | Supported next action |
|---|---|
| First-frame codes are ready, but initial worker has not started and is waiting for GIL/CPU/state lock | Address that measured host ownership/critical-section delay; a GPU priority change will not remove it. |
| Initial work waits on later frames or on another request's readiness before its own decode | Reduce that specific dependency while preserving lifetime/readiness for all selected inputs. Do not remove necessary waits. |
| Decoder is submitted and dependencies are satisfied, but its GPU start is delayed by other streams | Investigate GPU service scheduling or amount of work on that critical path, then remeasure first-audio and steady generation together. |
| Decoder's own eager bootstrap execution dominates | Optimize the measured reference-prefixed bootstrap shape/path, validating decoder state and numerical behavior. Cold graph misses alone do not prove this case. |
| Audio D2H is complete but first send is delayed | Address the measured output-drain/wakeup delay. |
| Prefix keys or uncached work differ materially | Resolve why the prepared inputs/cache history differ before attributing those milliseconds to synchronization. |

For #2126, subsequently create `tmp/e5-combined` from the same base and apply both saved patches in order. Add sampling-staging slot reuse/copy spans and runtime terminal-result wait begin/end to the same trace. Compare it against early IDs, not against E4b/E4c. Terminal outbox waiting is an actual place to inspect, but E4 did not execute it and cannot blame it.

Return raw Nsight reports, raw stage events, complete server logs, benchmark per-request data, the diagnostic patch, launch/import provenance, and continuous telemetry. Preserve audio/codec evidence for exceptional outputs separately. Final acceptance should rerun the original streaming workload without diagnostics after the measured cause is addressed; no additional model/router qualification matrix is needed just to identify this delay.
