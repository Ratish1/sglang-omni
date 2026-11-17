"""Measurement only: the coalescer counts decode as busy on the iteration right after a decode step, as it read before admission filtered first."""

import importlib.abc
import importlib.util
import sys

SCHEDULER = "sglang_omni.scheduling.omni_scheduler"


def patch_scheduler(module) -> None:
    cls = module.OmniScheduler

    def prefill_coalesce_waiting_requests(self, running_batch):
        if self.prefill_coalesce_requests <= 1 or self.chunked_req is not None:
            return None
        decode_is_idle = running_batch is None or running_batch.is_empty()
        last_batch = self.last_batch
        if decode_is_idle and last_batch is not None and last_batch.forward_mode.is_decode():
            decode_is_idle = False
        if not self.prefill_coalesce_when_idle and decode_is_idle:
            return None
        if self.prefill_coalesce_requires_pending_builds:
            with self.request_admission_lock:
                build_work_pending = bool(
                    self.pending_request_builds or self.pending_request_admissions or self.backlogged_request_build_payloads
                )
            if not build_work_pending and not (self.prefill_coalesce_after_builds_during_decode and not decode_is_idle):
                return None
        waiting = self.waiting_queue
        if not waiting or len(waiting) >= self.prefill_coalesce_requests:
            return None
        return waiting

    cls.prefill_coalesce_waiting_requests = prefill_coalesce_waiting_requests
    print("coalesce_patch applied", file=sys.stderr, flush=True)


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name != SCHEDULER or name in self.seen:
            return None
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(loaded):
            exec_module(loaded)
            patch_scheduler(loaded)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, PatchOnImport())
