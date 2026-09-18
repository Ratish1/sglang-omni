import json, glob, statistics as st
def steady(base,l,lo=0.15,hi=0.70):
    adm={}; term={}
    for p in glob.glob(f"{base}/{l}/events/events_coordinator_*.jsonl"):
        for line in open(p):
            e=json.loads(line); r=e["request_id"]; n=e["event_name"]; t=e["timestamp_ns"]/1e9
            if n=="request_admission": adm[r]=t
            elif n=="terminal_response": term[r]=t
    t0=min(adm.values()); wall=max(term.values())-t0
    a,b=t0+lo*wall, t0+hi*wall
    done=[r for r,t in term.items() if a<=t<b]
    return wall, len(done)/(b-a)
for base,labels,tag in (("/workspace/sglang-omni/.tmp/ab16b",("A1","A2","B1","B2"),"A vs B c16"),):
    print(f"=== {tag}: steady-state throughput (completions in the 15-70% window) ===")
    v={}
    for l in labels:
        w,q=steady(base,l); v[l]=q
        print(f"  {l}: wall={w:5.1f}s   steady qps={q:6.3f}")
    A=(v['A1']+v['A2'])/2; B=(v['B1']+v['B2'])/2
    dA=abs(v['A1']-v['A2']); dB=abs(v['B1']-v['B2'])
    print(f"\n  A={A:.3f}  B={B:.3f}  B-A={B-A:+.3f}  driftA={dA:.3f} driftB={dB:.3f}"
          f"  {'RESOLVED' if abs(B-A)>max(dA,dB) else 'inside drift'}")
