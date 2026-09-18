import json
A='/workspace/sglang-omni/.tmp/out/g1-fullbase-20260917T123922Z'; B='/workspace/sglang-omni/.tmp/out/g1-fullragged-20260917T124948Z'
def load(d):
    r=json.load(open(d+'/seedtts/measured/speed_results.json'))['per_request']
    return {x['id']:x for x in r}
a,b=load(A),load(B)
for nm,m in (('A main',a),('B ragged',b)):
    L=sorted(m.values(), key=lambda x:-x['audio_duration_s'])[:8]
    print(nm, 'count ==81.92:', sum(1 for x in m.values() if abs(x['audio_duration_s']-81.92)<0.001))
    for x in L: print('   %-52s %7.2f  lat %8.3f rtf %7.3f chunks %d'%(x['id'],x['audio_duration_s'],x['latency_s'],x['rtf'],x['audio_chunk_count']))
