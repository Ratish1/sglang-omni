#!/usr/bin/env bash
# 10-23, cards 0 and 1, after queue l5q: the prefill's numerics with and without L5. t_l4b (eager prefill) against
# t_l5b (the prefill replayed from full CUDA graphs) at c1f seed 1234, both orders, both servers under prefill_probe,
# which records every prompt's first token logits and embeddings (the outputs are not scored); then
# prefill_reference.py on card 0 puts each server's logits against an fp32 forward of the same embeddings.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_probe.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_probe.sh'
D=/data/c20261022b
Q=$D/quality
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
export PAIR_HOME=$D
export PAIR_PORTS=43000,43400
log() { echo "$* $(date +%T)" >> $Q/queue_probe.txt; }
until grep -q "queue l5q done" $Q/queue_l5q.txt 2>/dev/null; do sleep 30; done
log "queue probe start t_l4b $(git -C /workspace/t_l4b rev-parse --short HEAD) t_l5b $(git -C /workspace/t_l5b rev-parse --short HEAD)"
rm -rf $Q/prefill_probe
(cd /workspace/sglang-omni && PAIR_PROBE_PATH=$HERE/prefill_probe PREFILL_PROBE=$Q/prefill_probe PAIR_CELLS=c1f \
  PAIR_CELLS_B=c1f python3 $IN/serve_pair_c20261021.py /workspace/t_l4b /workspace/t_l5b $Q/p_probe > $Q/p_probe.log 2>&1)
log "pair p_probe rc $? patched $(grep -h 'prefill probe patched' $Q/p_probe/*/serve_gpu*.log | wc -l) records $(for d in $Q/prefill_probe/*/*; do printf '%s:%s ' ${d#$Q/prefill_probe/} $(ls $d | wc -l); done)"
(cd /workspace/t_l4b && CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 -u $HERE/prefill_reference.py $Q/prefill_probe \
  $Q/prefill_reference.json > $Q/prefill_reference.txt 2>&1)
log "reference rc $? $(python3 -c "
import json; r = json.load(open('$Q/prefill_reference.json'))
print('prompts', r['prompts'], '|', ' | '.join(f\"{s} kl mean {e['all']['kl']['mean']:.2e} p99 {e['all']['kl']['p99']:.2e} max_abs median {e['all']['max_abs']['median']:.3f} top1 {e['all']['top1']:.4f}\" for s, e in r['against_reference'].items()))
print(' | '.join(f\"{p}: kl mean {v['kl']['mean']:.2e} identical {v['identical']} second farther {v['second_farther_from_reference']} ratio {v['kl_to_reference_mean_ratio']:.2f}\" for p, v in r['pairs'].items()))
" 2>&1 | tr '\n' ' ')"
log "queue probe done"
