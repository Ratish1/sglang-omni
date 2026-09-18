import gzip, json, re, sys, statistics as st
path=sys.argv[1]; label=sys.argv[2]
TARGETS=("prepare_decode_buffers","apply_sglang_qwen3_tts_result")
pat=re.compile(r'"name":\s*"([^"]*(?:%s)[^"]*)"' % "|".join(TARGETS))
hits={t:[] for t in TARGETS}
frames=[]   # (tid, ts, dur, target)
with gzip.open(path,'rt',errors='replace') as f:
    for line in f:
        if '"dur"' not in line: continue
        m=pat.search(line)
        if not m: continue
        try:
            obj=json.loads(line.rstrip().rstrip(','))
        except Exception:
            continue
        if obj.get("ph")!="X": continue
        nm=obj.get("name","")
        for t in TARGETS:
            if t in nm:
                hits[t].append(obj["dur"])
                frames.append((obj.get("tid"),obj["ts"],obj["dur"],t))
                break
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(round(p*(len(v)-1))))]
print(f"=== {label} : total wall duration of each frame (us) ===")
for t in TARGETS:
    v=hits[t]
    if not v: print(f"  {t}: NO EVENTS"); continue
    print(f"  {t:36s} n={len(v):5d}  mean={st.mean(v):9.1f}  p50={q(v,.5):9.1f}  p90={q(v,.9):9.1f}  p99={q(v,.99):9.1f}  max={max(v):9.1f}")
json.dump(frames, open(f"/tmp/frames_{label}.json","w"))
print(f"  (frame windows saved for self-time pass: {len(frames)})")
