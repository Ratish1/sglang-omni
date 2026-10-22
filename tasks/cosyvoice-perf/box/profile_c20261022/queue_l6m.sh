#!/usr/bin/env bash
# 10-22, card 0: the chain with the prefix graph tiers sharing one set of input buffers. The prefix cache's GPU suite
# and the vocoder suite, then a warm boot whose log gives the prefix graphs' memory, the reserve and the pool, and
# the boot's recompiles and tracebacks.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_l6m.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_l6m.sh'
D=/data/c20261022b
P=$D/l6m
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
export PAIR_HOME=$D
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-items-chain
git -C /workspace/t_chain checkout -q --detach $(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
log "queue l6m start t_chain $(git -C /workspace/t_chain log --oneline -1 | cut -c1-60) sglang_omni $(git -C /workspace/t_chain rev-parse HEAD:sglang_omni | cut -c1-8)"
(cd /workspace/t_chain && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_chain:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3/test_prefix_cache_gpu.py tests/unit_test/fun_cosyvoice3/test_vocoder.py \
  -q -p no:cacheprovider -rfEs > $P/units.log 2>&1)
log "units rc $? $(grep -E 'passed|failed' $P/units.log | tail -n 1)"
bash $IN/warm_c20261021.sh t_chain 0 0-15,64-79 29400 $P/warm
log "warm $(grep 'bench rc' $P/warm/progress.txt) $(grep -E 'Completed requests|Failed requests' $P/warm/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $P/warm/serve.log) recompiles $(grep -ciE 'recompil' $P/warm/serve.log)"
grep -E "prefix graphs captured|Flow prefix cache" $P/warm/serve.log | sed 's/^/    /' | cut -c1-260 >> $P/queue.txt
log "queue l6m done"
