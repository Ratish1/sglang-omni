import gzip, re, sys, json, statistics as st
path, label = sys.argv[1], sys.argv[2]
TARGETS = ("prepare_decode_buffers", "apply_sglang_qwen3_tts_result")
re_name = re.compile(r'"name":\s*"(.*)"\s*,?\s*$')
re_tid  = re.compile(r'"tid":\s*(\d+)')
re_ts   = re.compile(r'"ts":\s*([0-9.]+)')
re_dur  = re.compile(r'"dur":\s*([0-9.]+)')
cur_name = None; cur_tid = None; cur_ts = None
rows = []      # every python_function event: (tid, ts, dur, name)
hits = {t: [] for t in TARGETS}
cat_python = False
with gzip.open(path, 'rt', errors='replace') as f:
    for line in f:
        s = line.strip()
        if s.startswith('"cat":'):
            cat_python = 'python_function' in s or 'cpu_op' in s
        m = re_name.search(s)
        if m and s.startswith('"name":'):
            cur_name = m.group(1); continue
        m = re_tid.search(s)
        if m and s.startswith('"tid":'):
            cur_tid = int(m.group(1)); continue
        m = re_ts.search(s)
        if m and s.startswith('"ts":'):
            cur_ts = float(m.group(1)); continue
        m = re_dur.search(s)
        if m and s.startswith('"dur":'):
            dur = float(m.group(1))
            if cur_name is not None and cur_tid is not None and cur_ts is not None:
                rows.append((cur_tid, cur_ts, dur, cur_name))
                for t in TARGETS:
                    if t in cur_name:
                        hits[t].append((cur_tid, cur_ts, dur)); break
            cur_name = None; cur_ts = None
def q(v, p):
    v = sorted(v); return v[min(len(v)-1, int(round(p*(len(v)-1))))]
print(f"=== {label}: frame total duration (us) ===")
for t in TARGETS:
    v = [d for _, _, d in hits[t]]
    if not v:
        print(f"  {t}: NO EVENTS"); continue
    print(f"  {t:32s} n={len(v):5d} mean={st.mean(v):8.1f} p50={q(v,.5):8.1f} p90={q(v,.9):8.1f} p99={q(v,.99):8.1f}")
# self time = dur - sum(direct children dur)
by_tid = {}
for tid, ts, dur, nm in rows:
    by_tid.setdefault(tid, []).append((ts, dur, nm))
for k in by_tid: by_tid[k].sort()
print(f"=== {label}: frame SELF time (us) ===")
for t in TARGETS:
    selfs = []
    for tid, ts, dur in hits[t]:
        ev = by_tid.get(tid, []); end = ts + dur
        child_total = 0.0; cursor = ts
        import bisect
        i = bisect.bisect_left(ev, (ts, -1, ""))
        while i < len(ev):
            cts, cdur, cnm = ev[i]
            if cts >= end: break
            if cts >= cursor and cts + cdur <= end + 1e-6 and not (cts == ts and cdur == dur):
                child_total += cdur; cursor = cts + cdur
            i += 1
        selfs.append(max(0.0, dur - child_total))
    if selfs:
        print(f"  {t:32s} n={len(selfs):5d} mean={st.mean(selfs):8.1f} p50={q(selfs,.5):8.1f} p90={q(selfs,.9):8.1f} p99={q(selfs,.99):8.1f}")
