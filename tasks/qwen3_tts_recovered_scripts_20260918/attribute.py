"""Kernel and host-block attribution for an sglang-omni torch trace.

Pass 1  collect GPU kernels (name, dur) and the host-blocking runtime calls
        (tid, ts, dur, name) we want mapped back to code.
Pass 2  stream python_function events; for each blocking call keep the INNERMOST
        enclosing python frame on the same tid.  Timing comes from every event;
        only the join is reconstructed, exactly once per call.
"""
import gzip, sys, bisect, collections, statistics as st

path, label = sys.argv[1], sys.argv[2]
BLOCKING = ("cudaEventSynchronize", "cudaStreamSynchronize", "cudaDeviceSynchronize",
            "cudaMemcpyAsync", "cudaHostAlloc", "cudaMalloc")

def events(fh):
    """Yield (cat, name, tid, ts, dur) for pretty-printed chrome trace blocks."""
    cat = name = tid = ts = None
    for line in fh:
        s = line.strip()
        if s.startswith('"cat":'):   cat = s.split('"')[3]; name = tid = ts = None
        elif s.startswith('"name":') and cat: name = s.split('"')[3]
        elif s.startswith('"tid":') and cat:
            try: tid = int(s.split(':')[1].strip().rstrip(','))
            except ValueError: tid = None
        elif s.startswith('"ts":') and cat:
            try: ts = float(s.split(':')[1].strip().rstrip(','))
            except ValueError: ts = None
        elif s.startswith('"dur":') and cat and name is not None:
            try: dur = float(s.split(':')[1].strip().rstrip(','))
            except ValueError: cat = None; continue
            yield cat, name, tid, ts, dur
            cat = name = tid = ts = None

# ---------- pass 1 ----------
kernels = collections.defaultdict(list)
blocks  = []                      # (tid, ts, dur, name)
with gzip.open(path, 'rt', errors='replace') as fh:
    for cat, name, tid, ts, dur in events(fh):
        if cat == "kernel":
            kernels[name].append(dur)
        elif cat == "cuda_runtime" and name in BLOCKING and tid is not None and ts is not None:
            blocks.append([tid, ts, dur, name, None, None])  # innermost any, innermost source

def q(v, p):
    v = sorted(v); return v[min(len(v)-1, int(round(p*(len(v)-1))))]

print(f"================ {label}: GPU KERNEL TIME (cut at 1.0% share) ================")
tot = sum(sum(v) for v in kernels.values())
rows = sorted(((sum(v), n, len(v)) for n, v in kernels.items()), reverse=True)
print(f"{'kernel':58s}{'launches':>10s}{'GPU ms':>10s}{'share':>8s}{'mean us':>10s}")
shown = 0.0
for s, n, c in rows:
    share = 100*s/tot if tot else 0
    if share < 1.0: continue
    shown += share
    print(f"{n[:57]:58s}{c:>10d}{s/1000:>10.1f}{share:>7.1f}%{s/c:>10.1f}")
print(f"  total GPU kernel time {tot/1000:.1f} ms across {len(rows)} distinct kernels; "
      f"rows shown cover {shown:.1f}%")

# ---------- pass 2 : innermost python frame per blocking call ----------
by_tid = collections.defaultdict(list)
for i, b in enumerate(blocks):
    by_tid[b[0]].append(i)
for t in by_tid:
    by_tid[t].sort(key=lambda i: blocks[i][1])
starts = {t: [blocks[i][1] for i in idx] for t, idx in by_tid.items()}

with gzip.open(path, 'rt', errors='replace') as fh:
    for cat, name, tid, ts, dur in events(fh):
        if cat != "python_function" or tid not in by_tid or ts is None:
            continue
        end = ts + dur
        idx, ss = by_tid[tid], starts[tid]
        lo = bisect.bisect_left(ss, ts)
        for k in range(lo, len(ss)):
            if ss[k] > end: break
            j = idx[k]
            b = blocks[j]
            if b[1] >= ts and b[1] + b[2] <= end + 1e-6:
                if b[4] is None or ts > b[4][0]:
                    b[4] = (ts, name)
                if (".py(" in name and "torch/" not in name
                        and (b[5] is None or ts > b[5][0])):
                    b[5] = (ts, name)

print(f"\n================ {label}: HOST BLOCKING CALLS BY CODE SCOPE ================")
agg = collections.defaultdict(list)
for tid, ts, dur, name, scope, src in blocks:
    call = scope[1] if scope else "?"
    call = call.replace("<built-in method ", "").split(" of ")[0]
    agg[(name + " via " + call, src[1] if src else "<unattributed>")].append(dur)
rows = sorted(((sum(v), n, sc, len(v), v) for (n, sc), v in agg.items()), reverse=True)
print(f"{'runtime call via python call':44s}{'n':>7s}{'total ms':>10s}{'mean us':>9s}{'p90 us':>9s}  code scope")
for s, n, sc, c, v in rows[:18]:
    if s/1000 < 1.0: continue
    print(f"{n[:43]:44s}{c:>7d}{s/1000:>10.1f}{st.mean(v):>9.1f}{q(v,.9):>9.1f}  {sc[:96]}")
