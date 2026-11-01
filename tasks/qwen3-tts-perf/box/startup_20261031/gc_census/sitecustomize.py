# Startup E probe (11-01): with GC_CENSUS_DIR set, every Python process that starts with this directory on its path
# appends one line per generation 1 or 2 garbage collection to GC_CENSUS_DIR/gc_<pid>.tsv: wall time at its end,
# generation, seconds, objects collected, and the running count and seconds of generation 0 collections so far
# (those run thousands of times, so they are summed, not written). The command line is written once. Lines are written
# as they happen, since a stage process leaves through os._exit and never runs atexit hooks.
import gc
import os
import sys
import time

DIRECTORY = os.environ.get("GC_CENSUS_DIR")

if DIRECTORY:
    started = {}
    young = [0, 0.0]
    handle = open(os.path.join(DIRECTORY, f"gc_{os.getpid()}.tsv"), "a", buffering=1)
    handle.write(f"# {' '.join(sys.argv)[:300]}\n")

    def record(phase, info):
        generation = info.get("generation")
        if phase == "start":
            started[generation] = time.perf_counter()
        else:
            began = started.pop(generation, None)
            if began is None:
                pass
            elif generation == 0:
                young[0] += 1
                young[1] += time.perf_counter() - began
            else:
                handle.write(
                    f"{time.time():.6f}\t{generation}\t{time.perf_counter() - began:.6f}\t"
                    f"{info.get('collected', 0)}\t{young[0]}\t{young[1]:.6f}\n"
                )

    gc.callbacks.append(record)
else:
    pass
