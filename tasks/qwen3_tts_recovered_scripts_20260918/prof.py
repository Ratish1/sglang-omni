import json, glob, bisect
def profile(base,l,nbins=10,nsamp=2000):
    adm={}; term={}
    for p in glob.glob(f"{base}/{l}/events/events_coordinator_*.jsonl"):
        for line in open(p):
            e=json.loads(line); r=e["request_id"]; n=e["event_name"]; t=e["timestamp_ns"]/1e9
            if n=="request_admission": adm[r]=t
            elif n=="terminal_response": term[r]=t
    t0=min(adm.values()); wall=max(term.values())-t0
    ev=sorted([(v-t0,1) for v in adm.values()]+[(term[r]-t0,-1) for r in term])
    ts=[e[0] for e in ev]; run=[]; c=0
    for _,d in ev: c+=d; run.append(c)
    bins=[[0.0,0] for _ in range(nbins)]
    for k in range(nsamp):
        t=(k+0.5)*wall/nsamp
        i=bisect.bisect_right(ts,t)-1
        v=run[i] if i>=0 else 0
        b=min(nbins-1,int(t/wall*nbins))
        bins[b][0]+=v; bins[b][1]+=1
    return wall,[s/max(n,1) for s,n in bins]
base="/workspace/sglang-omni/.tmp/ab16b"
print("mean in-flight per decile of the pass")
print(f"{'boot':5s}{'wall':>7s}  " + "".join(f"{i*10:>6d}%" for i in range(10)))
for l in ("A1","A2","B1","B2"):
    w,d=profile(base,l)
    print(f"{l:5s}{w:>7.1f}  " + "".join(f"{x:>7.1f}" for x in d))
