"""Time the shared freeze_gc(context) helper against a bare gc.freeze() on a heap of extra tracked objects.

usage: python3 freeze_gc_cost.py <extra objects> <helper|bare>
"""

import gc
import json
import sys
import time

from sglang.srt.utils.common import freeze_gc

extra_objects = int(sys.argv[1])
mode = sys.argv[2]
heap = [{"index": index} for index in range(extra_objects)]
gc.collect()
started = time.perf_counter()
if mode == "helper":
    freeze_gc("probe")
else:
    gc.freeze()
elapsed_ms = (time.perf_counter() - started) * 1000
print(
    json.dumps(
        {
            "mode": mode,
            "extra_objects": extra_objects,
            "frozen": gc.get_freeze_count(),
            "elapsed_ms": round(elapsed_ms, 2),
        }
    ),
    flush=True,
)
