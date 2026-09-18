import gzip, re, sys, statistics as st, collections
opener = (lambda p: gzip.open(p, "rt", errors="replace")) if sys.argv[1].endswith(".gz") else (lambda p: open(p, "rt", errors="replace"))
path, label = sys.argv[1], sys.argv[2]
# aggregate CUDA runtime + memcpy events by name: count, total us, mean, p90
cur = {}
agg = collections.defaultdict(list)
re_f = re.compile(r'"(cat|name|dur)":\s*"?([^",]+)"?')
cat = name = None
with opener(path) as f:
    for line in f:
        s = line.strip()
        if s.startswith('"cat":'):
            cat = s.split('"')[3]; name = None
        elif s.startswith('"name":') and cat:
            name = s.split('"')[3]
        elif s.startswith('"dur":') and cat and name:
            if cat in ("cuda_runtime", "gpu_memcpy", "gpu_memset", "cuda_driver"):
                agg[(cat, name)].append(float(s.split(':')[1].strip().rstrip(',')))
            cat = name = None
def q(v, p):
    v = sorted(v); return v[min(len(v)-1, int(round(p*(len(v)-1))))]
rows = []
for (c, n), v in agg.items():
    rows.append((sum(v), c, n, len(v), st.mean(v), q(v, .5), q(v, .9), max(v)))
rows.sort(reverse=True)
total = sum(r[0] for r in rows)
print(f"=== {label}: CUDA runtime / memcpy events, by total time ===")
print(f"{'name':42s}{'cat':13s}{'n':>7s}{'total ms':>10s}{'share':>7s}{'mean us':>9s}{'p50':>8s}{'p90':>9s}{'max us':>10s}")
for tot, c, n, cnt, mean, p50, p90, mx in rows:
    share = 100*tot/total if total else 0
    if share < 0.5 and 'HostAlloc' not in n and 'Malloc' not in n and 'Synchronize' not in n:
        continue
    print(f"{n[:41]:42s}{c[:12]:13s}{cnt:>7d}{tot/1000:>10.1f}{share:>6.1f}%{mean:>9.1f}{p50:>8.1f}{p90:>9.1f}{mx:>10.1f}")
print(f"  [total across all runtime/memcpy events: {total/1000:.1f} ms]")
