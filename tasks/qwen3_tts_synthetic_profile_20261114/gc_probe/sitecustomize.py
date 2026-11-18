"""Garbage collection pauses of the first Python process started with this directory on PYTHONPATH.

With GC_PROBE_OUT set, the process takes the variable out of its environment (so the processes it
spawns are not probed) and appends to GC_PROBE_OUT.<pid>.gc.tsv one row per collection that is a
full one or lasts at least GC_PROBE_MIN_MS: wall time at its start, generation, milliseconds,
objects collected, the collector's counts and the frozen object count. Once the file
GC_PROBE_OUT.census exists (looked for at most once a second), the next collection also writes the
tracked objects by type to GC_PROBE_OUT.<pid>.census.txt, once.
"""

import collections
import gc
import os
import time

GC_PROBE_MIN_MS = 2.0
CENSUS_TYPES = 60
CENSUS_CHECK_S = 1.0


def write_census(prefix: str) -> None:
    objects = gc.get_objects()
    by_type = collections.Counter(f"{type(item).__module__}.{type(item).__qualname__}" for item in objects)
    with open(f"{prefix}.{os.getpid()}.census.txt", "w") as handle:
        handle.write(f"{time.time():.3f} tracked {len(objects)} frozen {gc.get_freeze_count()}\n")
        for name, count in by_type.most_common(CENSUS_TYPES):
            handle.write(f"{count}\t{name}\n")


def install(prefix: str) -> None:
    handle = open(f"{prefix}.{os.getpid()}.gc.tsv", "a", buffering=1)
    started = {}
    census = {"is_written": False}

    def on_collection(phase: str, info: dict) -> None:
        if phase == "start":
            started["wall"] = time.time()
            started["clock"] = time.perf_counter()
        else:
            elapsed_ms = (time.perf_counter() - started["clock"]) * 1000.0
            if info["generation"] == 2 or elapsed_ms >= GC_PROBE_MIN_MS:
                counts = gc.get_count()
                handle.write(
                    f"{started['wall']:.3f}\t{info['generation']}\t{elapsed_ms:.2f}\t{info['collected']}\t"
                    f"{counts[0]},{counts[1]},{counts[2]}\t{gc.get_freeze_count()}\n"
                )
            else:
                pass
            now_s = time.monotonic()
            if not census["is_written"] and now_s - census.get("checked_s", 0.0) >= CENSUS_CHECK_S:
                census["checked_s"] = now_s
                if os.path.exists(f"{prefix}.census"):
                    census["is_written"] = True
                    write_census(prefix)
                else:
                    pass
            else:
                pass

    gc.callbacks.append(on_collection)


if os.environ.get("GC_PROBE_OUT"):
    install(os.environ.pop("GC_PROBE_OUT"))
else:
    pass
