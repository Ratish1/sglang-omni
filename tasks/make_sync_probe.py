"""Write the sync-probe patcher: wraps the async loop's launch and schedule
calls in torch.cuda sync debug mode and logs deduplicated Python stacks of
every synchronizing CUDA op. Experiment only. Usage:
  python make_sync_probe.py sglang_omni/scheduling/omni_scheduler.py
Revert with git.
"""

import pathlib
import sys

TARGET = pathlib.Path(sys.argv[1])

HELPERS = """
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

"""

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
        '                self._resolve_pending_async()\n\n            sp_entered = self._sp_enter("schedule") if sp_iter > 400 else False\n            try:\n                batch = self.get_next_batch_to_run()\n            finally:\n                self._sp_exit(sp_entered)\n            self.cur_batch = batch\n',
    ),
    (
        "                try:\n                    sched_output, pending_step = self._run_batch_launch(batch)\n                except Exception as exc:\n                    self._handle_batch_failure(batch, exc)\n                else:\n                    prev_pending = self._async_pending\n",
        '                try:\n                    if self._batch_is_decode(batch):\n                        self._sp_decode_launches += 1\n                        sp_phase = "launch:decode" if 400 < self._sp_decode_launches < 460 else None\n                    else:\n                        self._sp_extend_launches += 1\n                        sp_phase = "launch:extend" if 200 < self._sp_extend_launches < 260 else None\n                    sp_entered = self._sp_enter(sp_phase) if sp_phase else False\n                    try:\n                        sched_output, pending_step = self._run_batch_launch(batch)\n                    finally:\n                        self._sp_exit(sp_entered)\n                except Exception as exc:\n                    self._handle_batch_failure(batch, exc)\n                else:\n                    prev_pending = self._async_pending\n',
    ),
    (
        "                if batch:\n                    result = self.run_batch(batch)\n                    if result is not _FAILED_BATCH_RESULT:\n                        self.process_batch_result(batch, result)\n",
        '                if batch:\n                    sp_phase = "sync_run:" + ("decode" if self._batch_is_decode(batch) else "extend")\n                    sp_entered = self._sp_enter(sp_phase) if sp_iter > 400 else False\n                    try:\n                        result = self.run_batch(batch)\n                    finally:\n                        self._sp_exit(sp_entered)\n                    if result is not _FAILED_BATCH_RESULT:\n                        self.process_batch_result(batch, result)\n',
    ),
]

s = TARGET.read_text()
for old, new in EDITS:
    assert s.count(old) == 1, f"anchor not unique/found: {old[:60]!r}"
    s = s.replace(old, new)
TARGET.write_text(s)
print("sync probe applied to", TARGET)
