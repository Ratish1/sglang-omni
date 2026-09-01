"""Stream a torch profiler chrome trace into compact per-category arrays and pickle them.

Keeps X events only. Categories kept: python_function, kernel, gpu_memcpy, gpu_memset,
cuda_runtime, cuda_driver, cpu_op, user_annotation, overhead.
"""

import json
import pickle
import sys
import time
from array import array

src, dst = sys.argv[1], sys.argv[2]
names = {}


def nid(n):
    i = names.get(n)
    if i is None:
        i = len(names)
        names[n] = i
    return i


cols = {
    "python_function": dict(
        id=array("q"),
        parent=array("q"),
        name=array("q"),
        tid=array("q"),
        ts=array("d"),
        dur=array("d"),
    ),
    "kernel": dict(
        name=array("q"),
        stream=array("q"),
        ts=array("d"),
        dur=array("d"),
        corr=array("q"),
        graph=array("q"),
    ),
    "gpu_memcpy": dict(
        name=array("q"),
        stream=array("q"),
        ts=array("d"),
        dur=array("d"),
        corr=array("q"),
        bytes=array("q"),
    ),
    "gpu_memset": dict(
        name=array("q"),
        stream=array("q"),
        ts=array("d"),
        dur=array("d"),
        corr=array("q"),
    ),
    "cuda_runtime": dict(
        name=array("q"), tid=array("q"), ts=array("d"), dur=array("d"), corr=array("q")
    ),
    "cuda_driver": dict(
        name=array("q"), tid=array("q"), ts=array("d"), dur=array("d"), corr=array("q")
    ),
    "cpu_op": dict(
        name=array("q"), tid=array("q"), ts=array("d"), dur=array("d"), extid=array("q")
    ),
    "user_annotation": dict(
        name=array("q"), tid=array("q"), ts=array("d"), dur=array("d")
    ),
    "overhead": dict(name=array("q"), tid=array("q"), ts=array("d"), dur=array("d")),
}
other = {}
dec = json.JSONDecoder()
t0 = time.time()
n = 0
with open(src, "r", encoding="utf-8") as f:
    buf = f.read(1 << 25)
    start = buf.index('"traceEvents"')
    pos = buf.index("[", start) + 1
    buf = buf[pos:]
    pos = 0
    done = False
    while not done:
        while True:
            # skip separators
            while pos < len(buf) and buf[pos] in " \n\r\t,":
                pos += 1
            if pos >= len(buf):
                break
            if buf[pos] == "]":
                done = True
                break
            try:
                ev, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                break
            pos = end
            n += 1
            if ev.get("ph") != "X":
                other[ev.get("ph")] = other.get(ev.get("ph"), 0) + 1
                continue
            cat = ev.get("cat")
            c = cols.get(cat)
            if c is None:
                other[cat] = other.get(cat, 0) + 1
                continue
            a = ev.get("args") or {}
            c["name"].append(nid(ev["name"]))
            c["ts"].append(ev["ts"])
            c["dur"].append(ev.get("dur", 0.0))
            if cat == "python_function":
                c["id"].append(a.get("Python id", -1))
                p = a.get("Python parent id")
                c["parent"].append(-1 if p is None else p)
                c["tid"].append(ev["tid"])
            elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
                c["stream"].append(a.get("stream", -1))
                c["corr"].append(a.get("correlation", -1))
                if cat == "kernel":
                    c["graph"].append(a.get("graph id", 0))
                if cat == "gpu_memcpy":
                    c["bytes"].append(a.get("bytes", 0))
            elif cat in ("cuda_runtime", "cuda_driver"):
                c["tid"].append(ev["tid"])
                c["corr"].append(a.get("correlation", -1))
            elif cat == "cpu_op":
                c["tid"].append(ev["tid"])
                c["extid"].append(a.get("External id", -1))
            else:
                c["tid"].append(ev["tid"])
            if n % 500000 == 0:
                print(f"{n} events {time.time()-t0:.0f}s", flush=True)
        if done:
            break
        more = f.read(1 << 25)
        if not more:
            break
        buf = buf[pos:] + more
        pos = 0
print("events", n, "other", other, f"{time.time()-t0:.0f}s", flush=True)
out = {
    "names": names,
    "cols": {k: {kk: vv for kk, vv in v.items()} for k, v in cols.items()},
}
with open(dst, "wb") as f:
    pickle.dump(out, f, protocol=5)
print("wrote", dst, f"{time.time()-t0:.0f}s")
