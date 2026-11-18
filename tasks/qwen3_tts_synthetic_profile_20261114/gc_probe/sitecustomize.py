"""Garbage collection pauses of the first Python process started with this directory on PYTHONPATH.

With GC_PROBE_OUT set, the process takes the variable out of its environment (so the processes it
spawns are not probed) and appends to GC_PROBE_OUT.<pid>.gc.tsv one row per collection that is a
full one or lasts at least GC_PROBE_MIN_MS: wall time at its start, generation, milliseconds,
objects collected, the collector's counts and the frozen object count. A thread looks every
CENSUS_POLL_S for GC_PROBE_OUT.census.<n> (n = 1, 2, ...) and, when it appears, writes the tracked
objects by type to GC_PROBE_OUT.<pid>.census.<n>.txt.
"""

import collections
import gc
import os
import threading
import time

GC_PROBE_MIN_MS = 2.0
CENSUS_POLL_S = 0.5


def write_census(prefix: str, index: int) -> None:
    started_s = time.perf_counter()
    objects = gc.get_objects()
    by_type = collections.Counter(f"{type(item).__module__}.{type(item).__qualname__}" for item in objects)
    path = f"{prefix}.{os.getpid()}.census.{index}.txt"
    with open(f"{path}.part", "w") as handle:
        handle.write(
            f"{time.time():.3f} tracked {len(objects)} frozen {gc.get_freeze_count()} "
            f"took_ms {(time.perf_counter() - started_s) * 1000:.0f}\n"
        )
        for name, count in by_type.most_common():
            handle.write(f"{count}\t{name}\n")
    os.replace(f"{path}.part", path)


def watch_census_triggers(prefix: str) -> None:
    index = 1
    while True:
        if os.path.exists(f"{prefix}.census.{index}"):
            write_census(prefix, index)
            index += 1
        else:
            time.sleep(CENSUS_POLL_S)


def install(prefix: str) -> None:
    handle = open(f"{prefix}.{os.getpid()}.gc.tsv", "a", buffering=1)
    started = {}

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

    gc.callbacks.append(on_collection)
    threading.Thread(target=watch_census_triggers, args=(prefix,), name="gc-probe-census", daemon=True).start()


if os.environ.get("GC_PROBE_OUT"):
    install(os.environ.pop("GC_PROBE_OUT"))
else:
    pass
