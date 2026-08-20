"""Append loop counters to omni_scheduler.py. Experiment only; works on any
arm because it wraps methods that exist on main and on prefill-launch-first
alike. Records, per scheduler process:
  batch histogram: (kind, batch size) per get_next_batch_to_run result
  drained_pending: kind of the in-flight step each time it is force-drained
Prints a summary line every 2000 iterations and at interpreter exit.
Usage:
  python make_loop_counters.py sglang_omni/scheduling/omni_scheduler.py
Revert with git.
"""

import pathlib
import sys

TARGET = pathlib.Path(sys.argv[1])

BLOCK = """

# ---- loop counters (experiment only) --------------------------------------
def _lc_install():
    import atexit
    from collections import Counter

    counters = {"batch": Counter(), "drained_pending": Counter(), "iters": 0}

    def _dump():
        batch = sorted(counters["batch"].items())
        drained = sorted(counters["drained_pending"].items())
        print(
            "[loop-counters] iters=%d batch=%s drained_pending=%s"
            % (counters["iters"], batch, drained),
            flush=True,
        )

    orig_next = OmniScheduler.get_next_batch_to_run

    def get_next_batch_to_run(self):
        batch = orig_next(self)
        counters["iters"] += 1
        if batch is None:
            counters["batch"][("empty", 0)] += 1
        else:
            mode = batch.forward_mode
            kind = "extend" if (mode is not None and mode.is_extend()) else "decode"
            counters["batch"][(kind, len(batch.reqs))] += 1
        if counters["iters"] % 2000 == 0:
            _dump()
        return batch

    orig_drain = OmniScheduler._resolve_pending_async

    def _resolve_pending_async(self):
        pending = self._async_pending
        if pending is not None:
            mode = pending[0].forward_mode
            kind = "extend" if (mode is not None and mode.is_extend()) else "decode"
            counters["drained_pending"][kind] += 1
        return orig_drain(self)

    OmniScheduler.get_next_batch_to_run = get_next_batch_to_run
    OmniScheduler._resolve_pending_async = _resolve_pending_async
    atexit.register(_dump)


_lc_install()
"""

source = TARGET.read_text()
assert "_lc_install" not in source, "loop counters already applied"
TARGET.write_text(source + BLOCK)
print("loop counters applied to", TARGET)
