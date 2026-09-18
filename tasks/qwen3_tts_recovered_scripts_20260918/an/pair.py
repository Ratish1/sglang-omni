import json, statistics as st
A='/workspace/sglang-omni/.tmp/out/g1-fullbase-20260917T123922Z/seedtts/measured/speed_results.json'
B='/workspace/sglang-omni/.tmp/out/g1-fullragged-20260917T124948Z/seedtts/measured/speed_results.json'
def load(p):
    r=json.load(open(p))['per_request']
    return {x['id']:x for x in r}
a,b=load(A),load(B)
ka,kb=set(a),set(b)
print('n A',len(a),'n B',len(b),'ids only in A',len(ka-kb),'ids only in B',len(kb-ka))
ids=sorted(ka&kb)
print('paired ids',len(ids))
same=[i for i in ids if abs(a[i]['audio_duration_s']-b[i]['audio_duration_s'])<=0.001]
diff=[i for i in ids if abs(a[i]['audio_duration_s']-b[i]['audio_duration_s'])>0.001]
print('identical duration (<=1ms)',len(same),'differ',len(diff))
d=sorted(abs(a[i]['audio_duration_s']-b[i]['audio_duration_s']) for i in ids)
def q(v,p):
    import math
    if not v: return float('nan')
    k=(len(v)-1)*p; f=math.floor(k); c=math.ceil(k)
    return v[f] if f==c else v[f]+(v[c]-v[f])*(k-f)
print('|B-A| over all paired: p50 %.3f p90 %.3f p99 %.3f max %.3f mean %.3f'%(q(d,.5),q(d,.9),q(d,.99),d[-1],sum(d)/len(d)))
dd=sorted(abs(a[i]['audio_duration_s']-b[i]['audio_duration_s']) for i in diff)
if dd: print('|B-A| over differing only: p50 %.3f p90 %.3f max %.3f'%(q(dd,.5),q(dd,.9),dd[-1]))
sa=sum(a[i]['audio_duration_s'] for i in ids); sb=sum(b[i]['audio_duration_s'] for i in ids)
print('total audio s: A %.2f  B %.2f  B-A %.2f  (%.2f%%)'%(sa,sb,sb-sa,100*(sb-sa)/sa))
for th in (20,60):
    print('above %ds: A %d  B %d'%(th,sum(1 for i in ids if a[i]['audio_duration_s']>th),sum(1 for i in ids if b[i]['audio_duration_s']>th)))
print('max duration: A %.2f  B %.2f'%(max(a[i]['audio_duration_s'] for i in ids),max(b[i]['audio_duration_s'] for i in ids)))
print()
print('ten largest |B-A|:')
print('%-52s %8s %8s %8s'%('id','A dur','B dur','B-A'))
for i in sorted(ids,key=lambda i:-abs(a[i]['audio_duration_s']-b[i]['audio_duration_s']))[:10]:
    print('%-52s %8.2f %8.2f %8.2f'%(i,a[i]['audio_duration_s'],b[i]['audio_duration_s'],b[i]['audio_duration_s']-a[i]['audio_duration_s']))
print()
print('paired subset (durations equal within 1ms), n=%d'%len(same))
print('%-6s %10s %10s %10s %10s %10s'%('arm','lat mean','lat p95','rtf mean','rtf p95','audio sum'))
for nm,m in (('A',a),('B',b)):
    L=sorted(m[i]['latency_s'] for i in same); R=sorted(m[i]['rtf'] for i in same)
    print('%-6s %10.4f %10.4f %10.4f %10.4f %10.1f'%(nm,sum(L)/len(L),q(L,.95),sum(R)/len(R),q(R,.95),sum(m[i]['audio_duration_s'] for i in same)))
La=[a[i]['latency_s'] for i in same]; Lb=[b[i]['latency_s'] for i in same]
Ra=[a[i]['rtf'] for i in same]; Rb=[b[i]['rtf'] for i in same]
print('ratio B/A: lat mean %.4f  lat p95 %.4f  rtf mean %.4f  rtf p95 %.4f'%(
 (sum(Lb)/len(Lb))/(sum(La)/len(La)), q(sorted(Lb),.95)/q(sorted(La),.95),
 (sum(Rb)/len(Rb))/(sum(Ra)/len(Ra)), q(sorted(Rb),.95)/q(sorted(Ra),.95)))
print()
print('full 1088 for reference')
for nm,m in (('A',a),('B',b)):
    L=sorted(m[i]['latency_s'] for i in ids); R=sorted(m[i]['rtf'] for i in ids)
    print('%-6s lat mean %.4f p95 %.4f  rtf mean %.4f p95 %.4f'%(nm,sum(L)/len(L),q(L,.95),sum(R)/len(R),q(R,.95)))
print()
print('ttfp/inter-chunk fields present:', sorted(set(a[ids[0]])))
