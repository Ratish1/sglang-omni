import json, glob, statistics as st, os, sys
BASE="/workspace/sglang-omni/.tmp/nbc-4090"
LABELS=["A1","B1","A2","B2"]
def q(v,p):
    v=sorted(v); return v[min(len(v)-1,int(round(p*(len(v)-1))))]
def summary(label,p):
    f=f"{BASE}/{label}/{p}/speed_results.json"
    if not os.path.exists(f): return None
    return json.load(open(f))["summary"]

print("="*100)
print("STREAMING c12, measured pass (s2)   [full seed-tts corpus, 1088 samples]")
print(f"{'boot':6s}{'n':>6s}{'fail':>6s}{'qps':>9s}{'rtf':>8s}{'TTFC mean':>11s}{'p95':>9s}{'p99':>9s}{'lat mean':>10s}{'inter':>8s}{'c50':>7s}{'under':>8s}")
for l in LABELS:
    s=summary(l,"s2")
    if not s: print(f"{l:6s} MISSING"); continue
    print(f"{l:6s}{s['total_requests']:>6d}{s['failed_requests']:>6d}{s['throughput_qps']:>9.3f}{s['rtf_mean']:>8.3f}"
          f"{s['audio_ttfp_mean_s']*1000:>10.1f}ms{s['audio_ttfp_p95_s']*1000:>8.1f}{s['audio_ttfp_p99_s']*1000:>9.1f}"
          f"{s['latency_mean_s']:>9.3f}s{s['inter_chunk_mean_s']*1000:>7.1f}{s['c50']:>7.1f}{s['max_playback_underrun_mean_s']*1000:>7.0f}")
print()
print("NON-STREAMING c12 (n1)")
print(f"{'boot':6s}{'n':>6s}{'fail':>6s}{'qps':>9s}{'rtf':>8s}{'lat mean':>10s}{'lat p95':>10s}")
for l in LABELS:
    s=summary(l,"n1")
    if not s: print(f"{l:6s} MISSING"); continue
    print(f"{l:6s}{s['total_requests']:>6d}{s['failed_requests']:>6d}{s['throughput_qps']:>9.3f}{s['rtf_mean']:>8.3f}{s['latency_mean_s']:>9.3f}s{s['latency_p95_s']:>9.3f}s")

# ---- drift bound and paired deltas ----
print()
def val(l,p,k): 
    s=summary(l,p); return None if not s else s[k]
for p,keys in (("s2",["throughput_qps","audio_ttfp_mean_s","audio_ttfp_p95_s","latency_mean_s"]),
               ("n1",["throughput_qps","latency_mean_s"])):
    print(f"--- {p}: within-arm drift vs between-arm effect ---")
    for k in keys:
        a1,a2,b1,b2=[val(l,p,k) for l in LABELS[0::2]+LABELS[1::2]][0],val("A2",p,k),val("B1",p,k),val("B2",p,k)
        if None in (a1,a2,b1,b2): continue
        scale=1000 if ("ttfp" in k or "latency" in k) else 1
        u="ms" if scale==1000 else ""
        drift=abs(a1-a2)*scale; driftB=abs(b1-b2)*scale
        eff=((b1+b2)/2-(a1+a2)/2)*scale
        print(f"  {k:22s} A:{a1*scale:8.3f}/{a2*scale:8.3f}  B:{b1*scale:8.3f}/{b2*scale:8.3f}  "
              f"driftA={drift:6.3f} driftB={driftB:6.3f}  B-A={eff:+7.3f}{u}")
    print()

# ---- per-segment decomposition from events ----
SEGS=[("preprocess","stage_dispatch","stage_complete","preprocessing"),
      ("intake","stage_complete","scheduler_request_build_start",None),
      ("build","scheduler_request_build_start","scheduler_request_build_end",None),
      ("to_queue","scheduler_request_build_end","scheduler_queue_enter",None),
      ("queue","scheduler_queue_enter","scheduler_prefill_start",None),
      ("prefill","scheduler_prefill_start","scheduler_prefill_end",None)]
def load(label):
    ev={}
    for p in glob.glob(f"{BASE}/{label}/events/*.jsonl"):
        for line in open(p):
            e=json.loads(line); r=e["request_id"]; n=e["event_name"]; stg=e["stage"]
            d=ev.setdefault(r,{})
            key=n if n not in ("stage_dispatch","stage_complete","stage_input_received","stage_stream_chunk_sent","stage_stream_chunk_received") else f"{stg}.{n}"
            d.setdefault(key,e["timestamp_ns"]/1e9)
    return ev
print("="*100)
print("TTFC PER-SEGMENT DECOMPOSITION (ms, steady state = all but first 12 admissions)")
rows={}
for l in LABELS:
    ev=load(l)
    if not ev: print(f"{l}: no events"); continue
    r=sorted(ev.values(), key=lambda d:d.get("request_admission",0))[12:]
    def seg(s,e):
        v=[(d[e]-d[s])*1000 for d in r if s in d and e in d]
        return (st.mean(v),q(v,.5)) if v else (float('nan'),)*2
    rows[l]={
      "preprocess": seg("preprocessing.stage_dispatch","preprocessing.stage_complete"),
      "intake":     seg("preprocessing.stage_complete","scheduler_request_build_start"),
      "to_queue":   seg("scheduler_request_build_end","scheduler_queue_enter"),
      "queue":      seg("scheduler_queue_enter","scheduler_prefill_start"),
      "prefill":    seg("scheduler_prefill_start","scheduler_prefill_end"),
      "VOCODER":    seg("tts_engine.stage_stream_chunk_sent","coordinator.stage_stream_chunk_received"),
      "TOTAL":      seg("request_admission","coordinator.stage_stream_chunk_received"),
    }
if rows:
    names=list(next(iter(rows.values())).keys())
    print(f"{'segment':13s}" + "".join(f"{l+' mean':>12s}" for l in LABELS) + "   |" + "".join(f"{l+' p50':>11s}" for l in LABELS))
    for nm in names:
        line=f"{nm:13s}"+"".join(f"{rows[l][nm][0]:>12.2f}" for l in LABELS if l in rows)
        line+="   |"+"".join(f"{rows[l][nm][1]:>11.2f}" for l in LABELS if l in rows)
        print(line)
    print()
    print("  B-A (mean of both B boots minus mean of both A boots), and the within-arm drift for scale:")
    for nm in names:
        if not all(l in rows for l in LABELS): break
        a=(rows['A1'][nm][0]+rows['A2'][nm][0])/2; b=(rows['B1'][nm][0]+rows['B2'][nm][0])/2
        dA=abs(rows['A1'][nm][0]-rows['A2'][nm][0]); dB=abs(rows['B1'][nm][0]-rows['B2'][nm][0])
        flag=" <-- exceeds drift" if abs(b-a) > max(dA,dB) else ""
        print(f"    {nm:13s} B-A={b-a:+8.2f}ms   driftA={dA:6.2f} driftB={dB:6.2f}{flag}")
