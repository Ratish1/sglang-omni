import json, glob, statistics as st, collections
BASE="/workspace/sglang-omni/.tmp/cpu16"
LAB=["A1","A2","C1","C2"]
def load(l):
    ev={}
    for p in glob.glob(f"{BASE}/{l}/events/*.jsonl"):
        for line in open(p):
            e=json.loads(line); r=e["request_id"]; n=e["event_name"]; s=e["stage"]
            k=n if n not in ("stage_dispatch","stage_complete","stage_input_received",
                             "stage_stream_chunk_sent","stage_stream_chunk_received") else f"{s}.{n}"
            ev.setdefault(r,{}).setdefault(k,e["timestamp_ns"]/1e9)
    return ev
SEG=[("preprocess","preprocessing.stage_dispatch","preprocessing.stage_complete"),
     ("intake","preprocessing.stage_complete","scheduler_request_build_start"),
     ("to_queue","scheduler_request_build_end","scheduler_queue_enter"),
     ("queue","scheduler_queue_enter","scheduler_prefill_start"),
     ("HANDOFF(to_queue+queue)","scheduler_request_build_end","scheduler_prefill_start"),
     ("prefill","scheduler_prefill_start","scheduler_prefill_end"),
     ("VOCODER","tts_engine.stage_stream_chunk_sent","coordinator.stage_stream_chunk_received"),
     ("TOTAL","request_admission","coordinator.stage_stream_chunk_received")]
rows={}
for l in LAB:
    ev=load(l)
    if not ev: print(f"{l}: no events"); continue
    rs=sorted(ev.values(), key=lambda d:d.get("request_admission",0))[16:]
    rows[l]={n:(st.mean(v) if (v:=[(d[e]-d[s])*1000 for d in rs if s in d and e in d]) else float('nan'))
             for n,s,e in SEG}
print(f"{'segment':26s}"+"".join(f"{l:>10s}" for l in LAB)+"      C-A    driftA   driftC")
for n,_,_ in SEG:
    if not all(l in rows for l in LAB): break
    a=(rows['A1'][n]+rows['A2'][n])/2; c=(rows['C1'][n]+rows['C2'][n])/2
    dA=abs(rows['A1'][n]-rows['A2'][n]); dC=abs(rows['C1'][n]-rows['C2'][n])
    flag=" <-- beyond drift" if abs(c-a)>max(dA,dC) else ""
    print(f"{n:26s}"+"".join(f"{rows[l][n]:10.2f}" for l in LAB)+f"{c-a:+9.2f}{dA:9.2f}{dC:9.2f}{flag}")
