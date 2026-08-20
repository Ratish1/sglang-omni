# Prefill launch-first: implementation log

Date: 2026-08-17. Companion to `tasks/prefill_launch_first_plan_20260816.md`
(the plan) and `tasks/prefill_admission_followups_20260815.md` (facts and the
gap probe). This log records what was verified and built while executing the
plan; it does not restate the design.

Branch: `prefill-launch-first`, worktree
`.claude/worktrees/prefill-launch-first`, cut from `upstream/main` at
`dd41c4e8` (2026-08-17). Pushed to `origin` (Ratish1/sglang-omni) at
`71afe09e`: commit `346c5851` (runner half) and `71afe09e` (loop half), each
green on its own. Notes live in the repo's canonical `tasks/` folder
(`/Users/ratish/sglang-omni/tasks/`), moved there from worktree copies on
2026-08-17.

## 1. Validation tasks resolved

V2 (A1, prefill CUDA graph replay behind an in-flight step). Resolved, holds.
Read at `/Users/ratish/sglang` (0.5.16): `PrefillCudaGraphRunner.load_batch`
(`model_executor/runner/prefill_cuda_graph_runner.py:921`) fills the static
buffers through `CudaGraphBufferRegistry.fill_from`
(`model_executor/cuda_graph_buffer_registry.py:379`), a grouped
device-to-device `foreach_copy_` from the ForwardBatch's device tensors; the
omni `input_embeds` sidecar is D2D copied at replay (`:1135`); the only
host-side static buffer (`_full_cg_seq_lens_cpu`, `:284`) exists for the
Full backend, and omni selects Breakable; breakable attention metadata uses
the eager `init_forward_metadata` path unless the backend opts into captured
metadata (DSV4 only, `:318-328`). Upstream's default mode enqueues consecutive
extend forwards behind an unfinished forward at the same depth
(`SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP=False`). No prefill-graph gating
is needed.

V7 (A2, overrun-row KV accounting on the lookahead resolve path). Resolved,
holds, with a side finding. 0.5.16 bumps `req.kv_committed_len` in
`prepare_for_decode` (`schedule_batch.py:2819`) and `release_kv_cache`
(`mem_cache/common.py:131`) hands `kv_len_to_handle=kv_committed_len` to
`cache_finished_req`, so a request finished at step N after step N+1 was
prepared already covers its N+1 slot (tree-inserted, or freed as a duplicate
or unaligned tail). Verified with the real `TokenToKVPoolAllocator`,
`ReqToTokenPool`, `RadixCache` and `Req` (script `v7_overrun_check.py` in the
session scratchpad): after release, 58 free + 6 tree-owned = 64 of 64.
`_resolve_and_process` is therefore right not to compensate, and a request
that finishes at its launched prefill and is present in the next decode is
the same class. Retract goes through the same `release_kv_cache`
(`release_req`, `schedule_batch.py:1735`).

Side finding (pre-existing, not touched here): the fast path's compensating
free `_free_overrun_step_slots` (`omni_scheduler.py`, from #966) frees the
stale batch's slot after `release_kv_cache` already disposed of it. With the
same script: after the compensating free, free + tree-owned = 65 of 64 when
the sequence was new to the tree (the tree still maps the slot), and the slot
appears twice on the free list when the sequence was already cached. That is
a double free under RadixCache + page_size=1, exactly the configuration the
helper's gate admits (all ASR models on CUDA). Reachable when the loop takes
the fast path with a pending step and the drain finishes a request present in
the stale batch: on min-batch-size-2 models (Whisper, Fun-ASR, ArkASR, Higgs,
Qwen3-Omni thinker) when two requests finish on consecutive steps at the
tail. Fix candidate: delete the helper and its call in `_drop_stale_overrun`;
proof is the allocator balance above turned into a unit test. Separate
change; noted for the user.

## 2. What was built (S1 and S2)

`sglang_omni/model_runner/base.py`:
- `_PendingStep.is_prefill` (default False).
- `execute_launch` accepts extend batches when `prefill_lookahead_eligible`
  holds (raises `RuntimeError` otherwise, replacing the decode-only assert),
  runs `_prepare_and_forward(is_prefill=True, is_lookahead=True)`, then
  `post_prefill_launch`, then the unchanged publish, event and copy.
- `execute_resolve` dispatches `post_prefill_resolve` when
  `pending.is_prefill`.
- `prefill_lookahead_eligible(batch)`: `_prefill_hooks_are_default()` and not
  `batch.is_prefill_only` and `lookahead_eligible(batch)`.
  `_prefill_hooks_are_default` compares the six prefill hooks
  (`before_prefill`, `cleanup_prefill`, `custom_prefill_forward`,
  `post_prefill`, `sample_before_post_prefill`,
  `requested_capture_hidden_mode_prefill`) against `ModelRunner`'s own.
- `post_prefill_launch` / `post_prefill_resolve` defaults share the plain-LM
  sample-plus-pinned-snapshot helpers with `post_decode_launch` /
  `post_decode_resolve` (`_snapshot_next_token_ids`,
  `_read_next_token_ids_snapshot`).

`sglang_omni/scheduling/omni_scheduler.py`:
- `_prefill_lookahead_eligible(batch)`: extend mode, `chunked_req is None`,
  `mix_running_indices is None`, not `is_prefill_only`, no row with
  `inflight_middle_chunks > 0`, then the runner's
  `prefill_lookahead_eligible`.
- `use_lookahead` in `_event_loop_async_decode` is the decode condition or
  `_prefill_lookahead_eligible(batch)`; the lookahead branch is otherwise
  unchanged; the else branch now serves ineligible prefills.
- `_run_batch_resolve` emits `scheduler_prefill_end` after `execute_resolve`
  (O(1) for decode-only resolves by the emitter's existing size check).
- Docstrings of the loop, `_run_batch_launch`, `_run_batch_resolve`,
  `_resolve_pending_async`, `_async_pending_batch` say "step" not "decode
  step".

Eligible set by construction (grep of every runner subclass): only the base
`ModelRunner` made by `AsrEngineBuilder.make_model_runner`, i.e. Qwen3-ASR,
Fun-ASR, MOSS-TD, ArkASR, Whisper. Every TTS, codec, thinker and Ming runner
overrides at least one prefill hook and keeps the exact synchronous path.

Tests (`tests/unit_test/pipeline/test_async_decode.py`, 53 pass on the
CPU host, 3 CUDA-guarded skipped): differential extend launch+resolve versus
`execute` on a plain runner (same req_ids, tokens, relay publication,
generation_steps, `is_prefill` set); consecutive extend launches resolve in
order; a runner overriding any one prefill hook is ineligible and refused;
sampling-history and prefill-only gates; loop: eligible prefill launches
behind the pending decode without draining and the pending decode resolves
after; each ineligible class (chunked req, middle chunk in flight, mixed,
prefill-only, runner ineligible) drains then runs sync; prefill then decode
resolve in launch order; a row finished at the prefill resolve is dropped from
the next decode resolve with the token rows trimmed in lockstep;
`scheduler_prefill_end` is emitted exactly once, at resolve, and not by a
following decode. `_FakeBatch` now carries `forward_mode`,
`is_prefill_only`, `mix_running_indices` and per-row
`inflight_middle_chunks` (real ScheduleBatch shape); the existing prefill
drain-order test is pinned as the runner-ineligible variant.

Pre-existing failures in this CPU environment, unchanged by the branch:
`test_weight_share_load_path.py` (3), `test_cli_prefill_coalesce.py`
(Qwen3-Omni thinker case), and suites needing `msgpack` / `sgl_kernel`.

## 3. Facts learned that the plan should absorb (edits held pending confirmation)

- Plan F4 says Higgs runs the sync loop; its pipeline config sets
  `enable_async_decode: True` (`higgs_tts/config.py:84`), so Higgs runs the
  async loop and pays the sync-prefill cost today; it stays on the sync
  prefill path because its runner overrides `before_prefill` and
  `post_prefill`. Splitting those hooks is a later opt-in.
- Whisper is no longer a non-goal: upstream #1497 moved its encoder into a
  pre-LM service and #1553 turned async decode on by default (min batch 2,
  `max_running_requests` 32, 8 build workers). Its LM prefill is now the plain
  path, so it is in the eligible set and belongs in the S3 matrix. The
  admission branch's Whisper gate result (c4-c16 loss, explained by the
  encoder inside prefill) is stale.
- Upstream added `DeferredAdmission` and `_pending_request_admissions` (built
  requests waiting on an encoder future). The admission branch's rule must
  count them as builds in flight when it merges upstream/main.
- Qwen3-ASR already had breakable prefill CUDA graphs when F2 was measured,
  so the bubble numbers already include graph replay.

## 4. S3 protocol (H100)

Arms, same GPU, fresh servers, three repeats, shipped defaults:
- A: `upstream/main` at `dd41c4e8` (timer, current loop). Control.
- B: `prefill-launch-first` (timer, fixed loop). B minus A isolates the loop
  fix.
- C, D: after merging upstream/main and this branch into
  `work-conserving-admission`: `--prefill-admission eager` and `batched` on
  the fixed loop. C or D minus B isolates admission policy.

Matrix: Qwen3-ASR c1/8/16/32/64; Whisper c1/2/4/8/16; MOSS-TD c1/8/16;
Fun-ASR c16/32; ArkASR c16; one Qwen3-ASR MPS-DP cell; a c64 burst with
client disconnects (zero scheduler exceptions).

Gates: B at or above A within spread at every cell; corpus WER equal
(batch-shape flips recorded, as F8 of the followups note); zero failures.
Then C/D against B decide S4 (delete the admission rule and switch if eager
wins everywhere).

Probe (B versus A, Qwen3-ASR c32): apply the script in section 5 to the
checkout, run the workload unprofiled, read `[gap-probe]` lines and the last
`[gap-probe-cum]`, then `git checkout -- sglang_omni/scheduling/omni_scheduler.py`.
On A the sync prefill appears as `span:sync_extend` with `gap:decode->sync_extend`
and `gap:sync_extend->decode` (the earlier measurement: ~2.8-3.0 ms of idle
around ~7 ms of wall for ~2.4 ms of GPU). On B a launched prefill appears as
`span:extend` (GPU time between its stream markers, expected near 2.4 ms) with
`gap:decode->extend` and `gap:extend->decode` expected near the steady
`gap:decode->decode` (0.02-0.05 ms). Probe cost was 1.0% QPS.

## 5. Gap probe for the launch-first loop (experiment only)

Save as `gap_probe_launch_first.py` and run
`python gap_probe_launch_first.py sglang_omni/scheduling/omni_scheduler.py`
from the checkout root. It edits the file in place by exact anchors and fails
loudly if an anchor is missing. Revert with git.

```python
import pathlib, sys

TARGET = pathlib.Path(sys.argv[1])

HELPERS = '''
    # ---- gap probe (experiment only) --------------------------------------
    def _gp_init(self) -> None:
        if getattr(self, "_gp_ready", False):
            return
        self._gp_ready = True
        self._gp_dev = torch.device(self.device)
        self._gp_prev_end = None
        self._gp_prev_kind = "start"
        self._gp_records = deque()
        self._gp_window = {}
        self._gp_cum = {}
        self._gp_window_t0 = time.perf_counter()
        self._gp_started = self._gp_window_t0

    def _gp_stream(self):
        return torch.cuda.current_stream(self._gp_dev)

    def _gp_mark_start(self, kind: str = ""):
        torch.cuda.nvtx.range_push(f"omni-launch:{kind}")
        ev = torch.cuda.Event(enable_timing=True)
        ev.record(self._gp_stream())
        return ev

    def _gp_mark_end(self, kind: str, start_ev) -> None:
        end_ev = torch.cuda.Event(enable_timing=True)
        end_ev.record(self._gp_stream())
        torch.cuda.nvtx.range_pop()
        self._gp_records.append(
            (self._gp_prev_kind, kind, self._gp_prev_end, start_ev, end_ev)
        )
        self._gp_prev_end = end_ev
        self._gp_prev_kind = kind

    def _gp_note_idle(self) -> None:
        self._gp_prev_kind = "idle"

    @staticmethod
    def _gp_add(stats: dict, key: str, value: float) -> None:
        stats.setdefault(key, []).append(value)

    def _gp_flush(self) -> None:
        records = self._gp_records
        while records and records[0][4].query():
            prev_kind, kind, prev_end, start_ev, end_ev = records.popleft()
            span = start_ev.elapsed_time(end_ev)
            for stats in (self._gp_window, self._gp_cum):
                self._gp_add(stats, f"span:{kind}", span)
                if prev_end is not None:
                    self._gp_add(stats, f"gap:{prev_kind}->{kind}", prev_end.elapsed_time(start_ev))
        now = time.perf_counter()
        if now - self._gp_window_t0 >= 5.0:
            self._gp_log("gap-probe", self._gp_window, now - self._gp_window_t0)
            self._gp_log("gap-probe-cum", self._gp_cum, now - self._gp_started)
            self._gp_window = {}
            self._gp_window_t0 = now

    @staticmethod
    def _gp_log(tag: str, stats: dict, seconds: float) -> None:
        import statistics

        parts = [f"[{tag}] window={seconds:.1f}s"]
        idle_total = 0.0
        busy_total = 0.0
        for key in sorted(stats):
            vals = stats[key]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            p90 = vals_sorted[int(0.9 * (len(vals_sorted) - 1))]
            parts.append(
                f"{key} n={len(vals)} med={statistics.median(vals):.3f} "
                f"mean={statistics.fmean(vals):.3f} p90={p90:.3f} sum={sum(vals):.1f}"
            )
            if key.startswith("gap:") and not key.startswith("gap:idle") and not key.startswith("gap:start"):
                idle_total += sum(vals)
            if key.startswith("span:"):
                busy_total += sum(vals)
        parts.append(f"idle_ms={idle_total:.1f} span_ms={busy_total:.1f}")
        logger.info(" | ".join(parts))

'''

EDITS = [
    (
        "    def _event_loop_async_decode(self) -> None:\n",
        HELPERS + "    def _event_loop_async_decode(self) -> None:\n",
    ),
    (
        "        while self._running:\n            self._process_admin_requests()\n            recv_reqs = self.recv_requests()\n            recv_reqs.extend(self._take_deferred_request_payloads())\n            self.process_input_requests(recv_reqs)\n            if self._engine_paused:\n                self._process_admin_requests()\n                self._resolve_pending_async()\n",
        "        self._gp_init()\n        while self._running:\n            self._gp_flush()\n            self._process_admin_requests()\n            recv_reqs = self.recv_requests()\n            recv_reqs.extend(self._take_deferred_request_payloads())\n            self.process_input_requests(recv_reqs)\n            if self._engine_paused:\n                self._process_admin_requests()\n                self._resolve_pending_async()\n",
    ),
    (
        "                try:\n                    sched_output, pending_step = self._run_batch_launch(batch)\n                except Exception as exc:\n                    self._handle_batch_failure(batch, exc)\n                else:\n                    prev_pending = self._async_pending\n",
        "                try:\n                    gp_kind = \"decode\" if self._batch_is_decode(batch) else \"extend\"\n                    gp_start = self._gp_mark_start(gp_kind)\n                    sched_output, pending_step = self._run_batch_launch(batch)\n                    self._gp_mark_end(gp_kind, gp_start)\n                except Exception as exc:\n                    self._handle_batch_failure(batch, exc)\n                else:\n                    prev_pending = self._async_pending\n",
    ),
    (
        "                if batch:\n                    result = self.run_batch(batch)\n                    if result is not _FAILED_BATCH_RESULT:\n                        self.process_batch_result(batch, result)\n                else:\n                    self._sched_idled = True\n                    self.self_check_during_idle()\n                    self._sleep_during_idle()\n\n            self.last_batch = batch\n            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():\n                self.self_check_during_busy()\n\n    def _drain_inbox_for_request",
        "                if batch:\n                    gp_kind = \"sync_decode\" if self._batch_is_decode(batch) else \"sync_extend\"\n                    gp_start = self._gp_mark_start(gp_kind)\n                    result = self.run_batch(batch)\n                    self._gp_mark_end(gp_kind, gp_start)\n                    if result is not _FAILED_BATCH_RESULT:\n                        self.process_batch_result(batch, result)\n                else:\n                    self._gp_note_idle()\n                    self._sched_idled = True\n                    self.self_check_during_idle()\n                    self._sleep_during_idle()\n\n            self.last_batch = batch\n            if envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.get():\n                self.self_check_during_busy()\n\n    def _drain_inbox_for_request",
    ),
]

s = TARGET.read_text()
for old, new in EDITS:
    assert s.count(old) == 1, f"anchor not unique/found: {old[:60]!r}"
    s = s.replace(old, new)
TARGET.write_text(s)
print("probe applied to", TARGET)
```

The same script applies to `upstream/main` (arm A): the anchors are the
same lines; there the sync prefill shows up as `sync_extend`.

## 6. Agreed PR stack and test plan (2026-08-17)

Decisions taken with the user this session:

- PR 1, `prefill-launch-first` (this branch): one PR, two commits (runner
  half, loop half). Mechanics only; the timer stays. Gate: arm A
  (`upstream/main`) vs arm B (this branch), full matrix (section 4), probe,
  nsys structural check (section 7).
- PR 2, overrun double free (section 1 side finding): its own small branch
  from `upstream/main`. Delete `_free_overrun_step_slots` and its call in
  `_drop_stale_overrun`; replace its unit tests with the real-allocator
  balance test (release_kv_cache covers the stale slot; no compensating free).
  Runtime proof: omni's `self_check_during_idle` overrides upstream's, so
  0.5.16's KV double-free and use-after-free invariant checks
  (`scheduler_components/invariant_checker.py`) never run in omni; wire them
  under upstream's strict-mem-check env flag in this PR and run a Whisper or
  Fun-ASR c2 tail stress with the flag on. No performance arm; nsys not
  applicable. Independent of PR 1; may go first or in parallel.
- PR 3, admission: stacked on PR 1, opened only after the eager vs batched
  experiment on the fixed loop. It removes prefill coalescing and ships one
  policy with no switch: if eager wins everywhere, delete the timer and its
  five knobs and return admission to upstream's PrefillAdder (no rule needed);
  if batched wins somewhere, the rule becomes the single policy and the flag
  still goes. Gate: arm B vs top of stack. The full matrix on the top of the
  stack is the release gate for all three; no separate combined branch.
- Scheduler refactor along upstream's component split: later, on its own,
  after the measurements. `PrefillManager` is dead wiring to remove then.

## 7. nsys structural check (per arm; not a throughput gate)

Purpose: the CUDA-event probe says how large the gaps are; nsys shows why. On
arm B the extend's kernels must sit directly behind the preceding decode's on
the single stream, and no host synchronization API may sit between a decode
launch and the following extend launch. Throughput under nsys is perturbed
by a few percent, so QPS and WER gates stay on unprofiled runs.

1. Apply the probe script (section 5). It labels each host launch with an
   NVTX range `omni-launch:decode|extend|sync_decode|sync_extend`; sglang's
   own step spans emit NVTX only with `--enable-layerwise-nvtx-marker`.
2. `nsys launch --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none -- <server command>`;
   warm up; at steady state (Qwen3-ASR c32, then Whisper c8) run
   `nsys start -o <arm>_<model>_c<N>`, wait about 20 s, `nsys stop`.
3. `nsys stats --report cuda_api_sum,cuda_gpu_kern_sum -f csv <rep>`: compare
   counts of `cudaStreamSynchronize`, `cudaEventSynchronize`,
   `cudaDeviceSynchronize` and synchronous `cudaMemcpy` between arms. Arm A
   shows a sync per prefill (the `.tolist()` inside the sync extend); arm B
   should show `cudaEventQuery` / `cudaEventSynchronize` at resolve only and
   `cudaMemcpyAsync` for the pinned snapshot.
4. GUI: filter the NVTX row for `omni-launch:extend`; kernels directly behind
   the preceding decode's, no gap. On arm A `omni-launch:sync_extend` shows
   the idle before and after.
5. Repeat on the top of the stack for the release gate.

Known candidate for a sync inside arm B's extend launch: `return_logprob`
batches (kept eligible by decision; the sync costs overlap, not correctness).
If seen, record it in the PR; it is not a gate failure.

## 8. Next steps checklist

Testing policy from 2026-08-17 on: one measured repeat per cell (plus one
discarded warmup pass), not three; reverse-order controls only where a cell
decides a gate. Decided by the user to move faster; the trade is that a
single-repeat delta inside a few percent is not evidence either way.

Probe scripts as files (same content as sections 5 and 12):
`tasks/probe_gap_launch_first.py` (CUDA-event gaps plus NVTX ranges) and
`tasks/probe_sync_launch_first.py` (synchronizing-op stacks). Both take the
scheduler file path as the only argument and edit it in place; revert with
`git checkout -- sglang_omni/scheduling/omni_scheduler.py`.

1. Done 2026-08-17: `prefill-launch-first` pushed at `71afe09e` (two
   commits).
2. Done 2026-08-17 (uncommitted, awaiting go): branch
   `fix-overrun-double-free`, worktree `.claude/worktrees/fix-overrun-double-free`
   from `upstream/main` `dd41c4e8`; see section 10.
3. H100: arm A vs arm B (matrix, probe, nsys); PR 2 tail stress with the
   strict flag.
4. Merge `upstream/main` and PR 1 into `work-conserving-admission`; count
   `_pending_request_admissions` as builds in flight; run eager vs batched on
   the fixed loop; shape PR 3 from the result.
5. Plan doc corrections (section 3) once confirmed.
6. Now: runner applies `tasks/probe_sync_launch_first.py` on arm B (Qwen3-ASR
   c32, ArkASR c16) and once on arm A (Qwen3-ASR c32); returns every
   `[sync-probe]` and `[sync-probe-cum]` line; runs the two nsys queries of
   section 12 on the existing Qwen c32 exports; returns `gap:decode->decode`
   for A and B from the existing gap-probe logs. Then the seam-fix decision
   (section 13).

## 9. H100 test steps for PR 1 (hand to the runner)

Arms:
- A: `sgl-project/sglang-omni` main at `dd41c4e8`.
- B: `Ratish1/sglang-omni` branch `prefill-launch-first` at `71afe09e`.
Both on the same H100, sglang 0.5.16, shipped defaults (no admission or
decode flags), fresh server per run, three repeats per cell, alternate arms
(A, B, A, B, ...) so drift lands on both.

Matrix (same corpus and benchmark client as the earlier gates):
- Qwen3-ASR: c1, c8, c16, c32, c64.
- Whisper: c1, c2, c4, c8, c16.
- MOSS-TD: c1, c8, c16.
- Fun-ASR: c16, c32.
- ArkASR: c16.
- Qwen3-ASR under the same MPS-DP setup used for the Higgs DP3xMPS run, one
  cell at c32.
- Qwen3-ASR c64 burst with client disconnects mid-stream (the existing abort
  tooling); pass condition is zero scheduler exceptions and no stuck
  requests, throughput not gated.

Per cell record: req/s, p50/p95/p99 request latency, TTFT if the client
reports it, corpus WER, failed requests, and any scheduler warning or
exception in the server log.

Gates: B at or above A within the three-repeat spread at every cell; corpus
WER equal (single proper-name flips from batch shape are recorded, not
failed); zero failures; zero exceptions in the burst.

Probe (unprofiled runs, both arms): Qwen3-ASR c32 and Whisper c8. Apply the
section 5 script to the checkout, run one full pass, keep every
`[gap-probe]` line and the last `[gap-probe-cum]` line from the server log,
then `git checkout -- sglang_omni/scheduling/omni_scheduler.py`. Expected on
B: `gap:decode->extend` and `gap:extend->decode` medians near
`gap:decode->decode`, `span:extend` near 2.4 ms on Qwen3-ASR; on A the same
prefills appear as `sync_extend` with the earlier idle around them.

nsys (section 7): Qwen3-ASR c32 on both arms, one 20 s capture each at steady
state; return the `cuda_api_sum` csv and a screenshot of the timeline around
two consecutive `omni-launch:extend` ranges.

Report format: one table per model, rows = concurrency, columns = A mean
(min..max), B mean (min..max), delta %, WER A, WER B, failures; then the
probe lines and the nsys api counts.

## 10. PR 2: overrun double free (branch `fix-overrun-double-free`)

Base `upstream/main` at `dd41c4e8`; worktree
`.claude/worktrees/fix-overrun-double-free`. Diff saved in the session
scratchpad as `pr2_double_free.diff`. Proposed as two commits:

1. `drop the compensating free of stale overrun step slots`
   (`omni_scheduler.py`): `_free_overrun_step_slots` and its two calls in
   `_drop_stale_overrun` are gone (with the dead `drop_tokens` list); the
   docstring states why no free is needed (prepare advanced
   `kv_committed_len`, `release_kv_cache` covers it). Tests: the five helper
   tests and their allocator recorder are removed; the drop-stale reslice
   tests keep their shape assertions; new
   `test_release_kv_cache_accounts_for_the_stale_step_slot_without_a_compensating_free`
   uses the real allocator, `ReqToTokenPool`, `RadixCache` and `Req` and pins
   the pinned-sglang contract the fix rests on: after a finish that follows a
   stale prepare, free + tree-owned == total, the slot is tree-owned when the
   sequence is new and freed exactly once when it was already cached.
2. `audit kv accounting at idle when SGLANG_CHECK_KV_PAGE_INVARIANTS is set`
   (`omni_scheduler.py`): builds upstream's `SchedulerInvariantChecker` in
   `_init_upstream_scheduler_components` and, from `self_check_during_idle`,
   runs `_check_all_pools`, `_check_req_pool` and `_check_kv_page_invariants`
   only when the flag is set and nothing owns transient KV (running batch
   empty, waiting queue empty). Reporting follows upstream's strict-idle
   policy (raise by default). Off by default: omni had never run upstream's
   idle memory checks, so a hidden pre-existing imbalance would otherwise
   crash a server at first idle. Test:
   `test_idle_kv_audit_is_opt_in_and_flags_a_double_free` (real pools,
   observer and checker: balanced pool passes, flag off skips, a double free
   raises).

Whole unit tree: failure and error set identical to the `upstream/main`
baseline (3240 passed: five helper tests removed, two added).

H100 proof for PR 2 (runner): Whisper and Fun-ASR (min batch size 2) at c2
with similar-length outputs, server started with
`SGLANG_CHECK_KV_PAGE_INVARIANTS=1`, one full pass, then let the server go
idle. On `upstream/main` the audit is not wired, so run the stress on the PR 2
branch with the fix reverted (`git revert` of commit 1 on a scratch branch)
to see the detection fire, then on PR 2 as is to see it stay silent. Zero
audit errors on PR 2 is the gate. If the audit reports an imbalance that has
nothing to do with the stale slot, that is a separate pre-existing finding
to record, not a PR 2 failure.

## 11. Arm A vs arm B results (2026-08-17, runner report) and verdict

Environment: A `dd41c4e8`, B `71afe09e`, H100 80GB, driver 580.126.20,
torch 2.11.0+cu130, sglang 0.5.16, SeedTTS EN 1,088 samples per repeat,
fresh server per cell, one warmup pass, three measured repeats, cells run
whole (not interleaved); order-sensitive cells got a reverse-order control.

Correctness and safety: zero failures anywhere; WER equal or single
proper-name flips; Whisper transcripts byte-identical; c64 disconnect burst
64/64 clean with canary 200 on both arms; Qwen3-ASR DP2 under MPS B +5.05%
aggregate, zero failures.

Throughput (B vs A req/s): Qwen3-ASR c1 +0.7, c8 -8.9 (reverse control -1.6,
overlapping), c16 +7.6, c32 +1.5, c64 -3.2 (overlapping); Whisper c1 -0.05,
c2 -1.2 (ranges adjacent, non-overlapping by 0.002), c4 -0.4, c8 +0.2,
c16 +0.9; Fun-ASR c16 -6.3 (reverse control -0.6, overlapping), c32 +0.6;
ArkASR c16 -2.5 and -5.1 in reverse order (the one consistent loss);
MOSS-TD c1 -0.9, c8 +5.1, c16 +26.8 (all overlapping, strong cold/warm
regime).

Structure: gap probe Qwen c32 `gap:decode->extend` 0.803 (A sync_extend) to
0.060 ms (B), `gap:extend->decode` 1.993 to 0.554 ms, cumulative idle 746 to
544 ms; but `span:extend` 10.737 (A) to 9.554 ms (B), not the GPU time the
plan projected. nsys Qwen c32: cudaStreamSynchronize 1,176 to 897,
cudaEventSynchronize 1,241 to 950, median previous-decode to prefill GPU gap
2.111 to 1.093 ms, host gap 3.879 to 2.819 ms; a stream synchronization
inside every measured prefill launch range on both arms (A 179/179, B
191/191); a sync API in the host interval before 42 of 191 B prefills.

Verdict: gate as written fails (ArkASR c16; Whisper c2 by a hair). Reading:
the outer bubble (drain plus exposed resolve) is gone, as designed, but the
prefill launch itself still blocks the host on a synchronizing CUDA op inside
the forward path, so a prefill still costs most of what it used to; the
upside is therefore small and the fixed per-launch cost (copy, event, pinned
snapshot, resolve dispatch) shows where nothing overlaps: min-batch-size-2
models at low concurrency, where a launched prefill is drained by the next
fast-path decode. Two facts to establish before any change: which line
synchronizes inside the extend launch (section 12), and what ArkASR does
differently (read `sglang_omni/models/arkasr/` end to end). Then fix,
re-measure A vs B, and add the "launch only when something overlaps"
condition if the mechanism confirms.

## 12. Sync probe for the launch stall (experiment only)

Purpose: name the Python line of every synchronizing CUDA op inside the
extend launch (and, for comparison, inside decode launches, the schedule
call, and the sync run path). Uses `torch.cuda.set_sync_debug_mode(1)`
around those calls after warmup and logs each distinct stack once
(`[sync-probe] phase=... new site`), plus counts every 2,000 iterations
(`[sync-probe-cum]`). Costs nothing outside the sampled windows. Apply with
`python make_sync_probe.py sglang_omni/scheduling/omni_scheduler.py`,
run Qwen3-ASR c32 on arm B for one warmup plus one measured pass, collect
every `[sync-probe]` and `[sync-probe-cum]` line, revert with git. Run once
on arm A too so the same sites can be matched.

Note on reading: `torch.cuda.set_sync_debug_mode` flags `.item()`, `.cpu()`,
`.tolist()`, `nonzero`, non-pinned `.to(device)` (PyTorch copies pageable
host memory with a stream synchronize) and explicit synchronizes; the
stack's innermost sglang or sglang_omni frame is the site to report.

```python
"""Write the sync-probe patcher: wraps the async loop's launch and schedule
calls in torch.cuda sync debug mode and logs deduplicated Python stacks of
every synchronizing CUDA op. Experiment only. Usage:
  python make_sync_probe.py sglang_omni/scheduling/omni_scheduler.py
Revert with git.
"""
import pathlib, sys

TARGET = pathlib.Path(sys.argv[1])

HELPERS = '''
    # ---- sync probe (experiment only) -------------------------------------
    def _sp_init(self) -> None:
        if getattr(self, "_sp_ready", False):
            return
        import warnings

        self._sp_ready = True
        self._sp_seen: dict = {}
        self._sp_counts: dict = {}
        self._sp_active = None
        self._sp_orig_showwarning = warnings.showwarning
        self._sp_extend_launches = 0
        self._sp_decode_launches = 0

        def _showwarning(message, category, filename, lineno, file=None, line=None):
            import traceback

            if self._sp_active is None or "synchroniz" not in str(message):
                return self._sp_orig_showwarning(message, category, filename, lineno, file, line)
            stack = "".join(traceback.format_stack(limit=25)[:-1])
            key = (self._sp_active, hash(stack))
            self._sp_counts[key] = self._sp_counts.get(key, 0) + 1
            if key not in self._sp_seen:
                self._sp_seen[key] = stack
                logger.warning("[sync-probe] phase=%s new site (%s):\\n%s", self._sp_active, message, stack)

        warnings.showwarning = _showwarning

    def _sp_enter(self, phase: str) -> bool:
        import warnings

        if self._sp_active is not None:
            return False
        warnings.simplefilter("always")
        self._sp_active = phase
        torch.cuda.set_sync_debug_mode(1)
        return True

    def _sp_exit(self, entered: bool) -> None:
        if not entered:
            return
        torch.cuda.set_sync_debug_mode(0)
        self._sp_active = None

    def _sp_report(self) -> None:
        for (phase, _), count in sorted(self._sp_counts.items(), key=lambda kv: -kv[1]):
            logger.warning("[sync-probe-cum] phase=%s count=%d", phase, count)

'''

EDITS = [
    (
        "    def _event_loop_async_decode(self) -> None:\n",
        HELPERS + "    def _event_loop_async_decode(self) -> None:\n",
    ),
    (
        "        while self._running:\n            self._process_admin_requests()\n            recv_reqs = self.recv_requests()\n            recv_reqs.extend(self._take_deferred_request_payloads())\n            self.process_input_requests(recv_reqs)\n            if self._engine_paused:\n                self._process_admin_requests()\n                self._resolve_pending_async()\n",
        "        self._sp_init()\n        sp_iter = 0\n        while self._running:\n            sp_iter += 1\n            if sp_iter % 2000 == 0:\n                self._sp_report()\n            self._process_admin_requests()\n            recv_reqs = self.recv_requests()\n            recv_reqs.extend(self._take_deferred_request_payloads())\n            self.process_input_requests(recv_reqs)\n            if self._engine_paused:\n                self._process_admin_requests()\n                self._resolve_pending_async()\n",
    ),
    (
        "                self._resolve_pending_async()\n\n            batch = self.get_next_batch_to_run()\n            self.cur_batch = batch\n",
        "                self._resolve_pending_async()\n\n            sp_entered = self._sp_enter(\"schedule\") if sp_iter > 400 else False\n            try:\n                batch = self.get_next_batch_to_run()\n            finally:\n                self._sp_exit(sp_entered)\n            self.cur_batch = batch\n",
    ),
    (
        "                try:\n                    sched_output, pending_step = self._run_batch_launch(batch)\n                except Exception as exc:\n                    self._handle_batch_failure(batch, exc)\n                else:\n                    prev_pending = self._async_pending\n",
        "                try:\n                    if self._batch_is_decode(batch):\n                        self._sp_decode_launches += 1\n                        sp_phase = \"launch:decode\" if 400 < self._sp_decode_launches < 460 else None\n                    else:\n                        self._sp_extend_launches += 1\n                        sp_phase = \"launch:extend\" if 200 < self._sp_extend_launches < 260 else None\n                    sp_entered = self._sp_enter(sp_phase) if sp_phase else False\n                    try:\n                        sched_output, pending_step = self._run_batch_launch(batch)\n                    finally:\n                        self._sp_exit(sp_entered)\n                except Exception as exc:\n                    self._handle_batch_failure(batch, exc)\n                else:\n                    prev_pending = self._async_pending\n",
    ),
    (
        "                if batch:\n                    result = self.run_batch(batch)\n                    if result is not _FAILED_BATCH_RESULT:\n                        self.process_batch_result(batch, result)\n",
        "                if batch:\n                    sp_phase = \"sync_run:\" + (\"decode\" if self._batch_is_decode(batch) else \"extend\")\n                    sp_entered = self._sp_enter(sp_phase) if sp_iter > 400 else False\n                    try:\n                        result = self.run_batch(batch)\n                    finally:\n                        self._sp_exit(sp_entered)\n                    if result is not _FAILED_BATCH_RESULT:\n                        self.process_batch_result(batch, result)\n",
    ),
]

s = TARGET.read_text()
for old, new in EDITS:
    assert s.count(old) == 1, f"anchor not unique/found: {old[:60]!r}"
    s = s.replace(old, new)
TARGET.write_text(s)
print("sync probe applied to", TARGET)
```

nsys per-episode query on the existing SQLite export (adapt table names to
the export's schema; kernels are attributed to the launch range through the
runtime API call's correlation id, since the prefill's kernels execute after
the range on the stream):

```sql
SELECT n.start AS range_start,
       (n.end - n.start) / 1e6 AS host_wall_ms,
       SUM(k.end - k.start) / 1e6 AS gpu_busy_ms,
       (MAX(k.end) - MIN(k.start)) / 1e6 AS gpu_span_ms
FROM NVTX_EVENTS n
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
  ON r.globalTid = n.globalTid AND r.start >= n.start AND r.start <= n.end
JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId
WHERE n.text = 'omni-launch:extend'
GROUP BY n.start
ORDER BY n.start;

SELECT n.start AS range_start, s.value AS api, (r.end - r.start) / 1e6 AS blocked_ms
FROM NVTX_EVENTS n
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
  ON r.globalTid = n.globalTid AND r.start >= n.start AND r.start <= n.end
JOIN StringIds s ON s.id = r.nameId
WHERE n.text = 'omni-launch:extend' AND s.value LIKE '%Synchronize%'
ORDER BY n.start;
```

Wanted from these: per extend launch on B, host wall vs GPU busy of its own
kernels, and how long the synchronizing API inside blocked. If host wall is
several times GPU busy, the stall inside the launch is the remaining cost.

## 13. Reading the pinned code for the launch stall (prediction, pending the probe)

Every ASR model's LM forward routes prefill through sglang's
`general_mm_embed_routine` (`qwen3_asr/sglang_model.py:229`,
`fun_asr/sglang_model.py:531`, `arkasr/sglang_model.py:184`,
`moss_transcribe_diarize/sglang_model.py:310`). With pre-LM embeddings the
precomputed path is taken (`mm_utils.py:_get_precomputed_embedding`), then
`get_embedding_and_mask` always calls `_adjust_embedding_length`
(`mm_utils.py:866`), whose `mask.sum().item()` is a device-to-host sync on
the forward stream. Under launch-first that sync waits for the in-flight
decode step plus the prefill's own prefix kernels before the rest of the
prefill can even be enqueued, so the launch blocks the host for most of the
step it was meant to overlap. Whisper's own forward has the same class in
omni code: `int(forward_batch.encoder_lens[index].item())` per audio item
(`whisper_asr/sglang_model.py`, `_batch_precomputed_encoder_states`), where
`forward_batch.encoder_lens_cpu` (a host list) is available.

Prediction to be confirmed by section 12's probe: the `[sync-probe]` site
inside `launch:extend` on B is `mm_utils._adjust_embedding_length` for
Qwen3-ASR/Fun-ASR/ArkASR/MOSS-TD and `_batch_precomputed_encoder_states` for
Whisper. Candidate seam fix (omni-owned, no sglang patch): the four omni ASR
forwards merge precomputed embeddings themselves with host-known offsets
(text embedding lookup, then device slices from `extend_prefix_lens_cpu`,
`extend_seq_lens_cpu` and the items' offsets, length-checked on the host),
bypassing `_adjust_embedding_length`; Whisper reads `encoder_lens_cpu`.
This is a change to what the prefill computes for every ASR model, so it
carries the full WER gate again; decide with the user after the probe.

Second reading of the same report: on B, cumulative probe idle at Qwen c32
was 544 ms against A's 746 ms, but the extend-adjacent gaps account for
only about 120 ms of B's total, and nsys counted 950 `cudaEventSynchronize`
calls (resolve waits that missed the query) in 20 s. The remainder is
decode-to-decode idle: at c32 with about 100-token prompts and short outputs
the loop is close to host-bound (per-step build, resolve, output processing
for 32 rows), which caps what any prefill-side change can return at that
point and is consistent with the small deltas. This is an inference from the
numbers, not a measurement; the probe's `gap:decode->decode` distribution
per arm settles it.

## 14. Sync-probe and nsys breakdown results (2026-08-17) and the revised mechanism

Runner report (one warmup, one pass each, diagnostic throughput only):
- Sync sites inside B's `launch:extend` (Qwen3-ASR c32 and ArkASR c16):
  `sglang/srt/managers/mm_utils.py:1008` (`torch.as_tensor` for the
  placeholder tensor) and `:873` (`_adjust_embedding_length`,
  `mask.sum().item()`). Arm A's `sync_run:extend` shows the same two plus
  `torch.isin` (`:864`) and omni's `output_processor.py:38` `ids.tolist()`
  (result processing, inside A's wider region). Prediction of section 13
  confirmed as to the sites.
- nsys per extend launch, Qwen c32: host wall median 7.285 ms (B) / 7.779 ms
  (A); own-kernel busy 0.745 / 0.750 ms; own-kernel span 6.931 / 7.320 ms;
  the `cudaStreamSynchronize` calls inside the ranges block ~7 microseconds
  each (max 0.69 ms on B). The syncs are not the multi-millisecond blocker.
- Decode-to-decode gaps: median 0.019 (A) vs 0.020 ms (B), p90 0.934 vs
  0.790 ms; no steady-decode penalty from the loop change.

Revised mechanism (this supersedes the cost model in the plan's section 3):
a multimodal prefill costs ~7 to 8 ms of host time per launch with well
under 1 ms of GPU work; the GPU executes each kernel as it is enqueued and
starves for the rest of the window, so the sync calls find nothing queued
and return immediately. The "bubble" measured earlier by CUDA-event spans
was this host time: an event span around a launch measures max(GPU work,
host enqueue time), and with a starved GPU it measures the host. The
"~2.4 ms GPU per prefill" in the followups note was an inference from those
spans and is wrong. Prefill launch-first can overlap only the previous
step's host resolve (~1 ms) with a launch, which matches the measured host
gap 3.9 to 2.8 ms and GPU gap 2.1 to 1.1 ms and the small throughput deltas.
Per pass at Qwen c32: about 190 launches times ~7.5 ms is ~1.4 s of a ~3.1 s
pass with the GPU nearly idle, which is the ~40% "bubble share" the earlier
probe reported, now with the right composition. This also explains
coalescing: grouping prefills amortizes per-launch host cost, so fewer
launches means less starvation; it is the first mechanically complete
account of why the timer and the batched rule ever helped, and why they
helped Qwen3-ASR most (highest prefill rate).

Consequences: PR 1 is correct and safe but targets ~1 ms of ~8 ms; the lever
is the host cost per prefill launch (schedule-side build plus launch-side
build, forward entry, sampling prep). ArkASR's consistent small loss fits a
host-bound path that gains ~1 ms only when a step is truly in flight and
otherwise pays the added per-launch bookkeeping; not proven.

Next measurement: `tasks/probe_host_launch_first.py` on B (Qwen3-ASR c32,
ArkASR c16), one warmup plus one pass; also once on A (Qwen c32). It wall
clocks the schedule step and the launch sub-steps by batch kind
(`schedule:extend`, `launch:extend`, `runner.build:extend`,
`runner.prepare_forward:extend`, `runner.forward_call:extend`,
`runner.sample:extend`, `runner.publish:extend`) and cProfiles the extend
schedule and launch for a sampled window (`[host-profile]`, top 40 by
cumulative and by total time). If `[host-profile]` does not appear, the run
had fewer than 81 sampled events. Wanted: the split of the ~7 ms into
sglang's ForwardBatch build (mrope positions, extend tensors), the
multimodal embedding routine, attention metadata and prefill graph load,
sampler prep, and omni's own steps; and whether each is per launch or per
request. That decides between cutting host cost at omni seams and
amortizing by batching (the admission question, now with the mechanism
understood), or both.

## 15. Host-probe results (2026-08-17, runner report), critique, and what the code says is inside the forward

Runner report (one warmup, one 1,088-request pass per case; probe and
cProfile active, so throughput is diagnostic only). Medians / p90 in ms:

| key | A Qwen c32 | B Qwen c32 | B Ark c16 |
|---|---:|---:|---:|
| schedule:extend | 1.581 / 3.933 | 1.640 / 3.999 | 3.658 / 4.692 |
| sync_run:extend (A) or launch:extend (B) | 9.758 / 16.485 | 8.370 / 13.269 | 10.859 / 15.381 |
| runner.build:extend | 0.280 / 0.775 | 0.233 / 0.591 | 0.108 / 0.175 |
| runner.prepare_forward:extend | 8.260 / 15.223 | 7.713 / 12.237 | 10.501 / 14.850 |
| runner.forward_call:extend | 8.253 / 15.213 | 7.707 / 12.230 | 10.495 / 14.839 |
| runner.sample:extend | 0.084 | 0.076 | 0.062 |
| runner.publish:extend | 0.031 | 0.028 | 0.025 |

Throughput A/B Qwen c32 344.8 vs 348.2 req/s, Ark c16 96.6; WER unchanged;
0 failures. The cProfile window was mis-sampled on Qwen (the counter also
advanced on schedule calls, my bug) and the launch-only supplemental profile
inflated launches ~4x, so it names call paths only.

What the report established: the ~7 to 10 ms of host time per prefill is
almost entirely inside `tp_worker.forward_batch_generation`, that is inside
sglang's `model_runner.forward` (+ `sample`). ForwardBatch build, omni
sampling and publish are each well under 0.3 ms. The report's decision
("evaluate the multimodal embedding seam and batching") is not supported by
its own numbers: it never decomposed the forward, and the embedding path is
a dozen small ops by code reading (below).

Corrections to the report's wording: launch-first did not "remove 1.4 ms";
A's `sync_run` includes the wait for the GPU and the result finalization
that B does at resolve time. Work moved, not saved. And ArkASR's extra 2 ms
in `schedule:extend` is unexplained by the report; the new probe splits it.

What is inside `forward_batch_generation` (read from sglang 0.5.16, not
inferred), per model:

- Qwen3-ASR: builder sets `cuda_graph_backend_prefill=breakable` (locked, so
  sglang's multimodal disable rule at `server_args.py:3656` is skipped) and
  `enable_prefill_input_embeds`. `_forward_raw` (`model_runner.py:1430`)
  routes an eligible extend to `PrefillCudaGraphRunner.execute`
  (`prefill_cuda_graph_runner.py:1220`): `load_batch` (buffer fills, then
  `attn_backend.init_forward_metadata(forward_batch)` eagerly at every
  replay for FA3, `:655`), then the outer `model.forward` runs eagerly:
  omni's `Qwen3ASR.forward` -> `general_mm_embed_routine` -> `embed_mm_inputs`
  (precomputed embeddings from the pre-LM encoder: `torch.concat`, chunk
  slice, `torch.isin` mask, `mask.sum().item()`, `input_embedding`,
  `masked_scatter_`; about a dozen small ops) -> `language_model.forward`
  whose `layer_model.forward` is swapped for `replay_layer_forward`
  (`:1119`) -> `BreakableCUDAGraph.replay` (`breakable_cuda_graph.py:266`):
  for each captured segment `seg.replay()` then the eager break fn, and the
  break fns are the attention layers (`radix_attention.py:401`,
  `unified_attention_with_output` -> `get_attn_backend().forward` = FA3
  `forward_extend`: KV write + `flash_attn_with_kvcache` + output copy +
  padded-tail zero). So one Qwen3-ASR prefill launch is N_layers graph
  launches plus N_layers eager attention calls plus the eager head
  (`logits_processor`) plus `sample`. That is per launch, independent of how
  many requests are in the batch, and it is Python-bound: the kernels behind
  it are tiny (0.75 ms GPU measured), which is exactly the starvation shape
  nsys showed.
- ArkASR: builder sets no prefill backend; sglang's default is breakable but
  the "multimodal model not on the allowlist" rule
  (`server_args.py:3803`, allowlist `model_config.py:1785` = Qwen3.5 only)
  sets it to DISABLED, and omni's engine factory would refuse breakable
  anyway (`supports_breakable_prefill_cuda_graph` is False for ARK-ASR). So
  ArkASR's prefill is the fully eager Qwen2 LM: every linear, norm, rope,
  attention and MLP op is launched from Python each prefill, several hundred
  launches per prefill. That is why its forward host time is ~10.5 ms and
  why the profile showed `general_mm_embed_routine` owning the LM time (it
  includes the language-model call).
- Both: the audio encoder is not in this path (pre-LM encoder service; items
  arrive with `precomputed_embeddings`), so `fwd.embed` cannot be large.

Prediction for the next probe (falsifiable): Qwen3-ASR `fwd.seg + fwd.brk`
is 60 to 80% of `fwd.mr_forward`, with `fwd.brk_n` = number of layers;
`fwd.embed` < 0.5 ms; `fwd.attn_meta` and `fwd.load_batch` each < 1 ms;
`fwd.sample` < 0.5 ms. ArkASR `fwd.lm` is > 80% of `fwd.mr_forward` with
`fwd.attn` a few ms and `fwd.attn_n` = number of layers. Decode:
`launch:decode` well under the extend numbers (graph replay). If this holds,
the "multimodal embedding seam" option is dead, and the levers are (a) fewer
prefill launches per second (batching, the admission question), (b) for
ArkASR the breakable prefill graph contract that Qwen3-ASR already adopted
(hundreds of eager launches -> ~2 x layers), (c) per-layer eager attention
cost lives in sglang and is not ours to patch. Launch-first stays a small,
correct mechanic.

Runner steps (one repeat, no cProfile now):
1. `git status` clean on the arm; `python tasks/probe_host_launch_first.py sglang_omni/scheduling/omni_scheduler.py`
   (edits in place; prints `host probe applied to ...`).
2. Start the server; the log prints `[host-probe] installed: pcg=... attn_backend=... lm=...`
   once (Qwen: pcg=PrefillCudaGraphRunner; Ark: pcg=None or EagerRunner).
3. B: Qwen3-ASR c32, ArkASR c16; A: Qwen3-ASR c32. One warmup pass, then one measured pass.
4. Return, per case, the LAST `[host-probe-cum]` line in full (all keys, including
   the decode and resolve keys) and the throughput/WER line.
5. Revert: `git checkout -- sglang_omni/scheduling/omni_scheduler.py`.

## 16. Host-probe v2 results (2026-08-18, runner report): prediction confirmed

One warmup + one pass, no cProfile. Medians (ms):

| key | A Qwen c32 | B Qwen c32 | B Ark c16 |
|---|---:|---:|---:|
| schedule:extend / sched.prepare_extend | 1.602 / 0.731 | 1.688 / 0.814 | 3.565 / 3.201 |
| schedule:decode / sched.prepare_decode | 0.308 / 0.134 | 0.264 / 0.131 | 0.180 / 0.087 |
| extend execution (A sync_run, B launch) | 9.262 | 8.172 | 10.689 |
| fwd.mr_forward:extend | 7.918 | 7.643 | 10.336 |
| fwd.embed | 0.737 | 0.775 | 0.318 |
| fwd.lm | 5.924 | 5.702 | 9.548 |
| fwd.logits | 0.224 | 0.224 | 0.140 |
| fwd.pcg / load_batch | 7.839 / 0.471 | 7.580 / 0.436 | eager |
| fwd.seg / fwd.brk (n=28) | 0.359 / 5.154 | 0.336 / 4.950 | eager |
| fwd.attn (n) | 3.919 (28) | 3.766 (28) | 2.814 (36) |
| fwd.attn_meta | 0.154 | 0.151 | 0.130 |
| launch:decode / fwd.mr_forward:decode | 0.838 / 0.415 | 0.806 / 0.401 | 0.561 / 0.298 |
| resolve:decode / process_result:decode | 0.776 / 0.055 | 0.601 / 0.055 | 2.393 / 0.044 |
| process_result:extend | 1.452 | 1.538 | 0.344 |

Throughput B/Qwen +3.1% vs A (instrumented, one repeat); WER unchanged; 0 failures.
Installed banners: Qwen `pcg=PrefillCudaGraphRunner attn=FlashAttentionBackend
lm=Qwen3ForCausalLM`; Ark `pcg=EagerRunner ... lm=Qwen2ForCausalLM`.

Prediction from section 15 holds:
- Qwen3-ASR: the 28 eager attention breaks are 4.95 of 7.58 ms of the graph
  execute (65%); segment replays are 0.34 ms; embed 0.78 ms (10%); metadata
  0.15; load_batch 0.44. Sampling is not inside the forward here (omni's
  `_sample_next_token_ids`, ~0.08 ms per the v1 probe). Prefill host cost per
  launch is a fixed ~11 ms (schedule 1.7 + launch 8.2 + result 1.5), per
  launch not per request.
- ArkASR: eager LM 9.5 of 10.3 ms, 36 layers, attention 2.8 ms; the other
  ~6.7 ms is eager launches of linears/norms/MLP that BCG would fold into
  segments (~0.4 ms by the Qwen number). Expected effect of adopting BCG for
  ArkASR: extend launch ~10.7 -> ~4.5 ms. Two omni-side anomalies to read:
  `sched.prepare_extend` 3.2 ms (vs 0.8) and `resolve:decode` 2.4 ms (vs 0.6).
- Decode is cheap on the host (0.8 launch + 0.6 resolve on Qwen), so the
  loop is prefill-launch bound at these rates.

What this closes: the "multimodal embedding seam" idea (worth <0.8 ms);
launch-first as a primary lever (bounded by decode GPU time, ~1 ms). What it
opens, in value order: (1) BCG adoption for ArkASR (and check Fun-ASR), the
#1458 recipe; (2) admission that amortizes the fixed ~11 to 15 ms per prefill
launch, driven by measured per-launch cost and arrivals, not constants;
(3) ArkASR `prepare_for_extend` and decode-resolve costs; (4) PR 1 parked
until (2) shows whether it earns its place. Unexplained: ArkASR c16 -2.5 to
-5% under B in the earlier A/B; a probe run of A ArkASR c16 (same probe)
would give per-key medians and step/prefill counts to compare.

Measurement rule adopted (from the mis-step in the plan): before choosing a
lever, capture the whole loop once per model config with nsys (NVTX
schedule/launch/resolve by kind; GPU busy fraction per phase) and the host
probe (Python attribution). CUDA-event spans alone are not evidence of GPU
cost (see memory `gpu-timing-attribution`).

## 17. PR 2 pushed (2026-08-18) and its H100 proof

Branch `fix-overrun-double-free` on origin, base `dd41c4e8`:
`0305af0e drop the compensating free of stale overrun step slots`,
`fa2c1641 audit kv accounting at idle when SGLANG_CHECK_KV_PAGE_INVARIANTS is set`.
Unit: `tests/unit_test/pipeline/test_async_decode.py` green at both commits
(31 pass at 0305af0e, 32 pass at fa2c1641, 3 skipped each); whole tree
verified earlier at the final state (3240 pass, baseline failure set).

Why the audit belongs in omni: omni's `self_check_during_idle` overrides
upstream's and never ran the pool/page checks, so nothing could see the
#966 double free. The audit is upstream's checker (page-duplicate and
use-after-free checks, pool totals, req pool) called at true idle (no
running, no waiting), opt-in via `SGLANG_CHECK_KV_PAGE_INVARIANTS=1`, off by
default exactly as upstream, zero cost when off. It is also the proof
instrument for this PR.

Proof recipe (one pass each; Qwen3-ASR c32 SeedTTS EN, any workload where
requests finish while a next step is in flight; RadixCache on, page_size 1,
async decode on, all defaults):
- env for both runs: `SGLANG_CHECK_KV_PAGE_INVARIANTS=1
  SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` (non-strict: the audit logs a
  warning instead of raising, so the run completes and the log is the signal).
- Run 1 (fix): checkout `fa2c1641`. Expect a clean pass and NO log line
  matching `KV double free|KV page use-after-free|memory leak detected` after
  the pass ends (the audit runs when the scheduler goes idle, i.e. after the
  last request; leave the server up ~10 s after the pass).
- Run 2 (bug reproduced, audit proves it): `git checkout -b audit-only
  fa2c1641 && git revert --no-edit 0305af0e` (restores the compensating free,
  keeps the audit). Same env, same workload. Expect the pass to complete and
  the log to contain `KV double free: sub-pool 0 has N duplicate pages`
  (and/or `memory leak detected`) once idle.
- Report: both throughput/WER lines (sanity only, no gate), and the grep of
  the three patterns per run. Optional third run: Whisper c2 tail stress with
  the same env on `fa2c1641`, expect clean.

## 18. PR 2 proof run 1 (2026-08-18): reverted arm silent, and why

Runner: Qwen3-ASR c32, both arms 1088/1088, WER 0.0120 / 0.0123, audit
matches 0 in both (fix `fa2c1641`, reverted `475dfc88`). Inconclusive.

Cause, from the code (my error in the recipe, not in the fix): the restored
free is only reachable from `_drop_stale_overrun`, which only runs on the
sync fast path after a drain (`omni_scheduler.py:2580`), i.e. for a batch
built while a lookahead step was pending. Qwen3-ASR sets
`async_decode_min_batch_size = 1` (`models/qwen3_asr/stages.py:22`), so
every decode batch takes the lookahead path and the sync path only ever
sees prefill batches, whose rows were never in the pending step. The drop
never drops, the free never runs, the audit has nothing to see. The
original recipe (section 10: Whisper/Fun-ASR c2 tail stress, min bs 2) had
this right; I loosened it to "any workload with finishes in flight" without
re-deriving reachability. The audit itself is fine: it checks free-pool
duplicates and `available + evictable + protected == total` with `!=`, so
either shape of the double free is visible at idle; the unit test pins that.

Trigger conditions, precisely: a model with `async_decode_min_batch_size`
2 (Whisper, Fun-ASR, ArkASR default 2), running batch dropping from bs 2 to
bs 1 while the bs-2 step is still pending, and the surviving row finishing
(or being retracted) on that pending step. At c2 the 2->1 transition happens
on every completion, so per pass this fires on the order of tens of times.

Corrected recipe:
- Model: Whisper (min bs 2), c2, SeedTTS EN one pass (short outputs =
  more completions per second). Env as before:
  `SGLANG_CHECK_KV_PAGE_INVARIANTS=1 SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`.
- Run 1 (fix `fa2c1641`): expect no audit match at idle.
- Run 2 (audit-only branch = `fa2c1641` + `git revert --no-edit 0305af0e`),
  then `python tasks/probe_overrun_free_count.py sglang_omni/scheduling/omni_scheduler.py`
  so every execution of the restored free logs `[overrun-free] call=K slots=[...]`.
  Expect K >= 1 during the pass and, at idle, `KV double free: ... duplicate
  pages` and/or `memory leak detected`. If K = 0 the workload did not reach
  the path and the run says nothing about the fix; report K either way.
- Report both greps, K, and throughput/WER lines as sanity.

Process rule (added to the checklist): before any hardware proof run, write
the trigger conditions from the code and instrument the path so "silent"
and "unreached" are distinguishable.

## 19. PR 2 proof run 2 (Whisper c2): K=0, and the full reachability of the removed code

Runner: Whisper c2, both arms clean, `[overrun-free]` K=0. Cause: sglang
disables the radix cache for Whisper (`server_args.py:4994-5000`, "Radix
cache is disabled for Whisper"), and `_free_overrun_step_slots` returns at
its own gate `page_size != 1 or disable_radix_cache` before the probe line.
Whisper was the wrong model again; my recipe error, not the runner's.

Is the removed code dead or a duplicate? Neither exactly. It is a
compensating free of a slot upstream already releases:
- `prepare_for_decode` allocates the step slot and bumps
  `req.kv_committed_len += 1` (`schedule_batch.py:2819`) before the batch runs.
- When the request finishes, `release_kv_cache` (`mem_cache/common.py:131`)
  hands `kv_committed_len` slots to `cache_finished_req` (tree insert or
  free) and frees any tail up to `kv_allocated_len`. The stale slot is inside
  that range, so it is accounted exactly once by upstream.
- Upstream's decode result processor says so in its own words
  (`batch_result_processor.py:699`): a finished or retracted request still
  present in the next batch "should only happen when overlap scheduling is
  enabled. And all the over-allocated tokens will be freed in
  `release_kv_cache`." Upstream's overlap loop has the identical overrun and
  deliberately adds no compensating free.
- Omni's helper then frees `out_cache_loc[row]` a second time (only under
  RadixCache + page_size 1, its own gate): the page lands twice on the free
  list, two later requests can be handed the same KV slot, silent
  corruption plus an accounting mismatch that omni's idle check never ran.
  #966 was probably right against the sglang of its day (a finish path that
  released by token count rather than `kv_committed_len`); at 0.5.16 it is a
  double free.

Reachability at 0.5.16 (from the loop, `omni_scheduler.py:2580`): the free
runs only when (a) a batch is built while a lookahead step is pending, (b)
that batch takes the sync path (decode below `async_decode_min_batch_size`,
or a lookahead-ineligible batch), (c) the drain finishes a row of that
batch, (d) RadixCache on and page_size 1. So: dead for Qwen3-ASR (min bs 1,
no sync decode) and Whisper (radix off); live for ArkASR and Fun-ASR (min
bs 2, radix on) at the bs 2->1 transition when the survivor finishes on the
pending step. Under PR 1 nothing changes here (a launched prefill's rows are
not in the next batch until resolved).

Final proof recipe, if a hardware repro is still wanted: ArkASR c2 (min bs
2, radix on), same env, same two arms, `tasks/probe_overrun_free_count.py`
on the reverted arm; expect K on the order of tens per pass and the audit
flagging at idle. The code evidence above (upstream's stated contract, the
real-pool release test) is already sufficient for the PR on its own.

## 20. PR 2 proof run 3 (ArkASR c2, 2026-08-18): PASS

Fix `fa2c1641`: 1088/1088, WER 0.0112, audit matches 0. Reverted+counter
`bd0aafd8`: 1088/1088, WER 0.0112, `[overrun-free]` K=16 (16 distinct
slots), 73 `memory leak detected` warnings, first one at the same timestamp
as call=1. Accounting: after call=1, available+evictable = total+1; at idle
after the pass, 1731273+55010 = total+16. Each restored free adds exactly one
phantom page: the legit release cached the slot into the radix tree
(evictable) and the compensating free put the same slot on the free list.
That is the mechanism of section 19, measured. PR 2 is proven: path
reachable, free corrupts accounting, audit detects it, fix leaves the
workload clean. Recipe (ArkASR c2, both env vars, two arms, counter) is the
regression check for this class.

## 21. PR 2 opened (2026-08-18)

https://github.com/sgl-project/sglang-omni/pull/1607 from
`Ratish1:fix-overrun-double-free` (now `9e8159cd` after folding a docstring backtick cleanup into commit 1; title "[Scheduler] ...") against upstream main. Stack
status: PR 2 open; PR 1 (`prefill-launch-first`, pushed, not opened) parked
pending the admission work; ArkASR BCG adoption parked as a per-model item,
not scheduler work; next scheduler step is PR 3 admission driven by measured
per-launch cost, with the whole-picture capture (nsys + host probe) as gate.

## 22. PR 2 review findings (2026-08-18) and the census fix

Review on #1607: P0, the audit called upstream's
`_check_kv_page_invariants`, which takes owners from the last batch and
returns before its free-list duplicate census when no request owns anything
(`invariant_checker.py:342`), so at true idle it never ran; the test passed
only because the scalar pool check tripped first (double free = total+1).
A balanced corruption (one leaked page + one double free) passes both.
P4s: the test relied on the caller's strict-idle env; tests assigned
`context._server_args` directly instead of the sanctioned override.

Why we missed it: I read the function's docstring, not its body. Rule: read
whole functions and files we depend on before citing them.

Fix (folded into commit 2, now `6e228dd0`): new
`sglang_omni/scheduling/kv_page_census.py`, an idle census stating the idle
invariant directly: every page 1..num_pages is owned exactly once by the
allocator free list (free_pages + release_pages) or by the prefix cache
(all_values_flatten, per page for page_size > 1). Reports leaked,
double_owned and out_of_range with samples through upstream's
`raise_error_or_warn` under the strict-idle env. Skipped (returns None) for
allocators without a page free list, session slots holding KV, trees
without a flatten accessor; scheduler skips hybrid SWA/SSM and hisparse.
Idle guard now also requires `chunked_req is None` and `_async_pending is
None`. Tests use `get_context().override_server_args(page_size=...)`, pin
`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE`, and add the balanced
leak+double-free regression and a page_size 2 census case (paged allocator).
Kept opt-in (upstream runs its scalar idle checks always): omni's model
paths have not been soaked under strict idle checks; flip after a soak.
Whole-file independent review requested (fable agent) before push.

Whole-file review (fable agent, 82 tool uses) verdict: both commits correct
as accounting fixes; approve with follow-ups. Adopted: hisparse skip before
the scalar checks (matches upstream on_idle); audit once per idle stretch
(gated on forward_ct, tested); page_size > 1 census limit documented (a page
held by two tree nodes is not visible). Verified by the agent with probes:
page 0 reserved, release_pages counted, free_group closed before idle, tree
keys page-aligned so no partial page in the tree, empty-tree guard,
override_server_args restores fully, off-path cost is one env read.

Follow-up found (pre-existing, not PR 2): fast-path stale row. The stale
batch's prepare_for_decode commits a step slot; the drain finishes the row
and release_kv_cache inserts that never-forwarded slot into the radix tree
keyed by prompt+outputs+EOS (`radix_cache.py:452-470`); `_drop_stale_overrun`
then removes the row so the slot is never written. Before PR 2 it was also
on the free list; after PR 2 single-owned but unwritten. Reachable on the
same min-bs-2 transition; for ASR the key includes per-audio hash pad ids
so a hit needs the same audio and transcript plus EOS and more tokens.
Fix options: forward the stale row and skip its result (upstream overlap's
approach; needs an omni-side skip since enable_overlap is False), or drain
before building when the next batch would take the sync path
(talker_scheduler.py:121-141 already rolls back a stale prep). Tracked as a
scheduler follow-up; the PR text names it.

Branch now `44fa2821` + `e6d8cd7b`, pushed; PR text updated.

## 23. PR 2 reduced to the fix only (2026-08-18)

User decision: drop the opt-in KV audit (commit 2 and the page census)
from the PR; diagnostics, not the fix, and it had cost two review rounds.
Branch reset to `44fa2821` (single commit), force-pushed; #1607 retitled
"[Scheduler] Drop the compensating free of stale overrun step slots", body
rewritten: the H100 proof is described as using an experiment-only idle
accounting check (upstream's `_check_all_pools` at idle) on both arms, which
is exactly what ran. Kept the real-pool release contract test (31 pass, 3
skipped). The follow-up (unwritten stale slot inserted into the tree on the
fast path) stays named in the PR and in section 22. If we ever want the
audit again, the census module and its tests are in this note's history
(commit `e6d8cd7b`, now unreferenced on origin).

## 24. PR 2 second review (tests) addressed (2026-08-18)

Findings: (1) `tests/unit_test/pipeline/test_scheduler.py` fast-path drop
test still asserted the compensating free (freed == [101]); it never ran
locally because that module does not collect in this venv (msgpack, then a
torch.profiler `profile | None` TypeError, both pre-existing). (2) The new
test proved upstream's premise, not the scheduler's behavior. (3) Direct
`context._server_args` assignment.

Fix (single commit `ea480621`, force-pushed, PR text updated):
- New `test_drop_stale_overrun_leaves_drained_rows_single_owned`: real
  pools and real Reqs; drain a finished row (release, insert) and a
  retracted row (release, no insert); run the real `_drop_stale_overrun` on
  the stale batch holding their committed step slots; assert accounting
  equals total, finished slot in tree exactly once, retracted slot on the
  free list exactly once, survivor's slot neither; all-dropped returns None
  with accounting unchanged; clean batch returned as is. Verified to fail on
  the parent scheduler (59 + 4 != 64 - 3: two phantom pages) and pass here.
- `test_scheduler.py` drop test: allocator scaffolding and freed asserts
  removed; keeps drop semantics; verified standalone against both parent
  (fails: no page_size attribute needed on parent... it errors) and branch.
- `override_server_args(page_size=1)` context manager for server args.
Lesson: a module that fails to collect in the local venv still counts;
grep the whole tests tree for the symbol being deleted before pushing.

## 25. Inventory: what is left, what is valid, what must be read (2026-08-18)

Correction first (found by reading, today): upstream `get_next_batch_to_run`
merges the last extend batch into the running batch at the next scheduling
turn, before its result is processed (`scheduler.py:2687-2762`,
`filter_batch(chunked_req_to_exclude=...)` then `merge_batch`). Under PR 1
the launched prefill's rows are therefore in the very next decode batch
(tokens via the FutureMap relay), and a row that finishes at prefill resolve
is the overrun handled by `_resolve_and_process` (`omni_scheduler.py:2359`,
`skip_rids`). PR 1's own tests pin this. My later explanation to the user
("the request joins decode one step later under B") was wrong, and the
ArkASR -2.5% hypothesis built on it is void. Unexplained again.

Left to do, with validity:
1. PR 2 (#1607): open, tests fixed; awaiting review. Valid, done on our side.
2. PR 1 (`prefill-launch-first`, parked): mechanic correct, ~1 ms lever;
   fate decided by PR 3's measurement. Valid to keep parked; not to merge
   alone.
3. PR 3 admission: valid and the main lever (fixed ~11-15 ms host per prefill
   launch, amortized by batching). Design not started; needs the MUST reads
   below first.
4. Follow-up from review: fast-path stale row's unwritten step slot inserted
   into the tree (`_drop_stale_overrun` after the drain). Valid, small,
   scheduler-owned; fix options: forward the row and skip its result (as
   upstream overlap), or roll back the stale prep before draining
   (`talker_scheduler.py:121-141` precedent, to read).
5. Plan doc corrections (section 3): F4 Higgs runs async, Whisper in scope,
   DeferredAdmission counted. Valid; edit after confirmation and after the
   reads below confirm each line.
6. Async decode min batch size knob: set aside by the user for now.
7. Per-model items (BCG for ArkASR/Fun-ASR, ArkASR prepare_for_extend 3.2 ms
   and decode resolve 2.4 ms): parked; not scheduler work.
8. Probes and notes: current; keep.

MUST read in full before designing PR 3 or editing the plan (each with the
claim it settles):
a. sglang `Scheduler.event_loop_overlap` and `get_next_batch_to_run` end to
   end (0.5.16 `scheduler.py`): the reference overlap behavior we compare
   against; merge of last extend batch; where retraction happens relative to
   launch/resolve.
b. sglang `get_new_batch_prefill` and `PrefillAdder` (`schedule_policy.py`):
   the admission budget rules (`max_prefill_tokens`, `chunked_prefill_size`,
   `max_running_requests`, new-token ratio, `batch_is_full`), i.e. what
   admission already does before omni's coalescing adds a wait.
c. omni `prefill_coalesce.py`, `get_new_batch_prefill` override,
   `_take_deferred_request_payloads`, `DeferredAdmission`,
   `_pending_request_builds/_admissions`, request builder threads: the exact
   admission path today, so PR 3 removes and replaces the right thing.
d. omni runner `execute_launch/execute_resolve`, `_snapshot_next_token_ids`,
   `_read_next_token_ids_snapshot`, `SGLangExecutionBridge` FutureMap
   relay: how the next batch gets tokens from an unresolved step; WAR safety
   of the pinned ping-pong buffer.
e. omni loop `_event_loop_async_decode` end to end once more with the
   merge fact above, and `_drop_stale_overrun` / `_resolve_and_process`.
f. `talker_scheduler.py:100-150`: the existing rollback of a stale prep.
g. sglang `cumulate_penalty_output_tokens` (`schedule_batch.py:2765`) and
   omni `lookahead_eligible`: the sampling-history policy difference; needed
   only if the eligibility gate is revisited.
h. For plan corrections: `higgs_tts/config.py` async flag and its runner
   hooks; Whisper builder after #1497/#1553; `DeferredAdmission` uses.

Rule: each read produces a short fact list with file:line into this note
before any design text is written; unread means unknown.

## 26. Fact list from reading omni's admission path (c, d, e), 2026-08-18, at upstream/main 2cac60e8

Worktree `.claude/worktrees/admission` (branch `admission`) at `2cac60e8`.
Since our base `dd41c4e8`, upstream added a bounded queue with rejection
(#1449): `max_queued_requests`, `_queued_admission_count` (waiting +
pending builds + pending admissions + build backlog + deferred payloads),
`_waiting_queue_is_full`, `_reject_queue_full` (`omni_scheduler.py:960-980`,
`982-1057`, `1196-1210`).

Arrival to admission (what the scheduler sees as "a request is here"):
- Loop turn: `recv_requests` + `_take_deferred_request_payloads` ->
  `process_input_requests` (`:863-936`): drains build and admission results,
  stages payloads into the build executor up to `request_build_max_pending`
  in flight (rest to a bounded backlog), rejects when the queue is full.
- Builds run in a thread pool (`_run_request_builder`, `:938`); results are
  drained on later loop turns (`_drain_request_build_results`, `:1059`).
  A build may return `DeferredAdmission` (a value plus a Future, e.g. the
  pre-LM encoder); it is held in `_pending_request_admissions` until ready
  (`_admit_or_defer_built_request`, `:1092-1142`; `_drain_request_admission_results`, `:1144`).
- `_enqueue_built_request` (`:1163-1225`): limits, KV capacity check,
  stream state, then `waiting_queue.append(req)` with
  `req._coalesce_enqueue_t = perf_counter()`. This append is the arrival
  admission acts on; everything before it is build/encoder latency.

Admission today (`get_new_batch_prefill` override, `:1334-1368`):
- Falls through to upstream when `prefill_coalesce_requests <= 1`, or a
  chunked request is in flight, or (`when_idle` False and decode idle), or
  (`requires_pending_builds` and no build/admission/backlog work pending and
  not (`after_builds_during_decode` and decode busy)), or the waiting queue
  is empty or already >= `prefill_coalesce_requests`, or the oldest waiting
  request has waited >= `prefill_coalesce_wait_s`. Otherwise returns an
  empty plan (hold). Five constants per model.
- Defaults: Qwen3-ASR 16 requests / 40 ms / when_idle / requires_pending /
  after_builds (`qwen3_asr/stages.py:26-30`); Whisper 2 / 6 ms / when_idle /
  requires_pending (`whisper_asr/stages.py:23-27`); MOSS-TD 4 / 12 ms / all
  three (`moss_transcribe_diarize/stages.py:88-92`); ArkASR, Fun-ASR, Higgs
  0 (off).
- Upstream `get_new_batch_prefill` then applies its own budget rules (agent
  fact list, section 27) to whatever is in the waiting queue.

Loops (`start`, `:1717`): async decode loop when `enable_async_decode`
(Qwen3-ASR, ArkASR, Fun-ASR, MOSS-TD, Whisper, Higgs, Qwen3-Omni thinker,
Zonos2 per flag), the overlap loop is refused (`:2344`), else the normal
sync loop (dots, minimax music, moss_tts_local default False, higgs stages
default False but pipeline config True).
- Async loop (`:2549-2636`): recv/build -> optional pre-drain for mixed
  chunk (Qwen3-Omni thinker only, `qwen3_omni/stages.py:975`) ->
  `get_next_batch_to_run` -> lookahead iff decode and bs >= min bs and
  runner eligible; else drain pending, `_drop_stale_overrun`, run sync.
  Prefill is always sync on this base (PR 1 changes that for eligible
  runners).
- Tokens between steps: `SGLangExecutionBridge.publish_next_tokens` stashes
  next tokens into upstream's FutureMap by `req_pool_indices` at every launch
  (sync and lookahead) and sets `batch.input_ids = None`; the next forward
  resolves inputs at entry via `resolve_forward_inputs`
  (`sglang_execution.py:73-124`). So a batch may include rows whose producing
  step is unresolved on the host; upstream merges the last extend batch into
  the running batch at the next turn (section 25).
- Lookahead step: `execute_launch` (`base.py:283`) builds, forwards, samples
  on GPU, `post_decode_launch` snapshot (pinned ping-pong host staging or
  device snapshot), publishes tokens, records an event, keeps a
  `schedule_batch.copy()`; `execute_resolve` (`:346`) waits the event,
  skips rows finished/retracted in a prior step, runs the collect and
  finalize. WAR safety of the host staging: two pinned buffers, resolve(N)
  reads one while launch(N+1) writes the other (`:204-231`).

Consequence for PR 3: admission has two layers, omni's hold (five constants
per model) and upstream's budget; the "arrival" is the post-build
`waiting_queue.append`; the loop is prefill-launch-bound at ~11-15 ms host
per launch (section 16); a hold's only purpose is to make one launch carry
more requests. The measured cost model gives the hold rule its inputs
without constants: hold while (rate of arrivals into the waiting queue) x
(remaining wait) is expected to add a request before the launch would have
finished anyway; details after the upstream fact list lands.

Plan-correction facts (h), verified at 2cac60e8:
- Higgs: pipeline config `enable_async_decode: True` (`higgs_tts/config.py:84`,
  stage default False at `stages.py:468`); runner overrides `before_prefill`
  and `post_prefill` (`higgs_tts/model_runner.py:84,97`) so its prefill is
  the sync path. Plan F4 ("Higgs runs the sync loop") is wrong as written.
- Whisper: `AsrEngineBuilder` runners are the base `ModelRunner`
  (`engine_factory.py:373`), async decode on by default (min bs 2),
  coalescing 2 / 6 ms; radix cache disabled by sglang. In scope for any
  eligible-prefill mechanic; not a non-goal.
- `DeferredAdmission` is produced only by ArkASR's request builder
  (`arkasr/request_builders.py:100,128`) for its encoder future; Qwen3-ASR's
  pre-LM encoder blocks inside the build thread instead
  (`qwen3_asr/encoder_service.py:179`). An admission rule that counts
  "builds in flight" must count `_pending_request_builds`,
  `_pending_request_admissions` and the build backlog, as
  `_queued_admission_count` already does (`omni_scheduler.py:960`).

## 27. Fact list from reading sglang 0.5.16 (a, b), agent report 2026-08-18 (S=scheduler.py, SP=schedule_policy.py, SB=schedule_batch.py, OU=overlap_utils.py, BRP=batch_result_processor.py, NT=new_token_ratio_tracker.py)

- Overlap loop: recv -> get_next_batch_to_run -> run_batch(N) -> queue
  (batch.copy(), result) -> process result N-1 -> last_batch = N (S:1554-1618).
  No batch-size threshold; prefill uses the same path (S:1620-1661, 3319-3376).
  Token relay: `future_map.stash(req_pool_indices, next_token_ids)` after
  forward, `input_ids=None`, resolved at next forward entry (S:3374-3410,
  OU:84-106, 245-262, 501-517).
- get_next_batch_to_run: chunked_req stash; last extend batch filtered and
  merged into running BEFORE its result is processed (S:2745-2767);
  `get_new_batch_prefill` first (prefill priority), else `update_running_batch`
  (S:2783-2811). Retract only when the next decode does not fit
  (`check_decode_mem`, S:3155, SB:2594-2597); retracted go to the END of
  the waiting queue (S:3216, 2391-2398), `is_retracted` cleared at
  prepare_for_extend (SB:2284).
- get_new_batch_prefill: None if batch_is_full or queue empty (S:2870);
  running cap via `get_num_allocatable_reqs` (S:2891-2897); FCFS default
  (server_args.py:737-751, SP:184-206); PrefillAdder budgets:
  `rem_input_tokens = max_prefill_tokens` (16384 default), `rem_chunk_tokens =
  chunked_prefill_size`, `rem_total_tokens = KV available + evictable -
  sum_running(min(max_new - out, 4096) x new_token_ratio)` (SP:441-470,
  483-490, 556-588); each admitted req charges extend + min(max_new, 4096)
  + page (SP:708-737); first waiting req always accepted; chunking when input
  exceeds rem_chunk_tokens (SP:1058-1179). One request is enough; the only
  accumulation mechanisms are opt-in and off (PrefillDelayer for DP
  attention, MinFreeSlotsDelayer) (S:1047-1067, 2875-2884).
- Mixed batches only with `enable_mixed_chunk` (default False) (S:1004,
  3091-3107). Prefill result: first token appended, finished -> release
  (BRP:220-241); under overlap a req finished at N-1 still in N is skipped
  (BRP:697-703).
- Arrival timing: `wait_queue_entry_time` used only for FCFS tie-break with
  priority, queued-limit eviction, and the off-by-default waiting timeout
  (S:2397, 2440-2519). Admission is work-conserving; no time-based batching.
- new_token_ratio: init min(0.7 x conservativeness, 1), decays per
  non-retract decode step to 0.14 x init over 600 steps, jumps after retract
  (NT:22-55, S:3218, 3190); reset at idle (S:3675).

Load-bearing for PR 3: upstream admits any single fitting request
immediately (prefill-first, work-conserving); omni's hold is the only
accumulation; the KV/token budgets and the queue bound stay upstream's.

## 28. PR 3 rule correction, estimator attempt parked, full sglang read launched (2026-08-18)

- Plan section 2a added to tasks/admission_plan_20260818.md: the one-step
  marginal rule of section 2 ("hold iff lambda c H0 > k") over-holds by a
  square root; the stationary latency-sum model gives K* = sqrt(2 lambda c
  H0), the per-turn rule "hold iff k (k + 1) < 2 lambda c H0" with time cap
  T* = (K* - 1) / lambda. Checked by simulation (Poisson arrivals, count
  threshold): best K by brute force = 13 / 5 / 1 / 35 for (lambda, c, H0) =
  (348, 24, 10 ms) / (200, 5, 10 ms) / (10, 1, 10 ms) / (400, 150, 10 ms);
  formula gives 12.9 / 4.5 / 0.4 / 34.6. The count+time rule with a stale
  lambda (5x over-estimate) stays within 18% of the objective and halves the
  maximum wait versus count-only.
- Facts read today: Req.stream is never set by an omni request builder
  (default False); _first_emit_done holds rids that emitted a first stream
  message; models without a stream builder never add to it; idle turn sleeps
  0.1 ms with builds pending else 1 ms (_sleep_during_idle); omni's
  _enqueue_built_request never sets time_stats.wait_queue_entry_time so
  upstream's waiting timeout is inert for fresh omni requests; mixed batches
  carry decoding rows behind extend rows (mix_with_running, SB:2534-2565).
- Attempt parked: a production estimator module (bucketed median regression
  for H0, arrival-rate window, build-duration median) wired into both loops
  with tests. User judged it over-built and not mechanical; the diff is saved
  at the session scratchpad as admission_estimators_parked.patch (plus the two
  files) and the worktree is restored. Lesson recorded: the decision inputs
  should be measured at the seam that produces them (runner host-launch span,
  loop admission count), not fitted statistically after the fact; and no
  design before the upstream split is read end to end.
- Local dev aid kept uncommitted in the admission worktree:
  sglang_omni/profiler/torch_profiler.py gets "from __future__ import
  annotations" because sglang's macOS metal profiler patch replaces
  torch.profiler.profile with a function and the class annotation
  "profile | None" then fails at import, which is why
  tests/unit_test/pipeline/test_scheduler.py never collected locally. Also
  installed msgpack into /Users/ratish/sglang/.venv. With both,
  test_scheduler.py runs (76 passed) and test_async_decode.py runs.
- Launched five fable read agents over the full upstream scheduler split
  (scheduler.py; schedule_policy.py + delayers + intake components;
  schedule_batch.py + mem_cache/common.py + overlap_utils.py; result
  processing components; observability and misc components), each writing
  tasks/sglang_read_*_20260818.md with file:line facts and an omni usage
  table. Design of the admission decision resumes only after those are read.

## 29. PR 3 implemented on upstream's delayer hook (2026-08-19)

- Reads landed: seven full-file reports (tasks/sglang_read_*_20260818.md) and
  the synthesis tasks/scheduler_component_map_20260818.md. Design in plan
  section 2b: the hold is an object on self.min_free_slots_delayer, the hook
  upstream consults in _get_new_batch_prefill_raw (S:2873-2883) after the
  batch_is_full and empty-queue checks, before calc_priority, skipped while a
  chunked prefill is in flight. Omni's get_new_batch_prefill override, the
  five prefill_coalesce_* parameters, prefill_coalesce.py, the CLI flags and
  helper, and every model default are removed (21 production files).
- Rule (sglang_omni/scheduling/prefill_admission.py, one class): launchable
  = min(waiting, num_allocatable_reqs); hold iff running_bs > 0 and
  launchable < running_bs * sqrt(2 H0 / S) and hold_age < sqrt(2 H0 S).
  H0 = smallest prefill turn wall observed (lifetime min); S = twice the mean
  age of running rows since forward_entry_time. No windows, no per-model
  constant. hold_since resets only when launchable drops to zero. Off at
  tp_size > 1 (hook left None). User direction applied: no separate estimator
  windows (an earlier draft had 32/64 sample windows and a residence history;
  replaced by state read at decision time).
- Turn wall observation: both loops time schedule + run + process of a
  prefill turn, excluding the async drain (_observe_prefill_turn).
- Enqueue now calls req.time_stats.set_wait_queue_entry_time() like upstream
  _add_request_to_queue (S:2399); _coalesce_enqueue_t is gone; the MiniMax
  pair stamps the uncond row with the cond row's entry time.
- Tests: tests/unit_test/scheduling/test_prefill_admission.py (threshold,
  cap, slot cap, hold_since reset, min cost, residence from ages, and two
  tests driving the real upstream _get_new_batch_prefill_raw: the delayer is
  consulted before calc_priority and skipped while chunked_req is set);
  loop tests in test_async_decode.py and test_scheduler.py with a fake clock
  (prefill wall recorded, drain excluded, decode turns ignored); enqueue
  stamps wait_queue_entry_time; hook installed at tp 1 and empty at tp 2;
  MiniMax pairing test. Old gate/validation/CLI tests deleted; model pipeline
  tests stripped of the removed keys. Cookbook passages updated.
- Verification: failure sets of the affected suites equal the base's on this
  Mac (pre-existing: sgl_kernel/msgpack collection errors, higgs collection,
  fun_asr and minimax rvq, test_compile/test_ipc/runtime_adapter); full-tree
  comparison run in the background against a base worktree.
- Decisions recorded from the user: all running rows count in running_bs;
  the idle-with-pending-builds hold is dropped and the Qwen c8 gate decides
  whether the one-line idle branch (hold at idle iff pending arrivals m > 0
  and hold_age < H0 m / (k + m)) is needed; CLI flags removed outright.
- Not in this PR: the fast-path stale-slot follow-up from the PR 2 review
  (its own PR after #1607), the on_idle housekeeping gap, the chunked_req
  abort/retract-pause gaps, and the macOS torch_profiler annotation fix
  (kept as an uncommitted local aid; separate one-line PR).
