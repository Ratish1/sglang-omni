#!/usr/bin/env bash
# 10-23, card 0: the speech tokenizer's graphs captured in the process's math mode (chain 03f504f0a, L3's cf27ecafd).
# The tokenizer's unit file, then tokenizer_parity_c20261020.py on every seed-tts EN reference: with the class now
# following the process flag, its graph_tf32=False graphs are the served ones, against eager fp32 and ONNX Runtime,
# with each method's time per call.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_tokfp32.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_tokfp32.sh'
D=/data/c20261022b
Q=$D/quality
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
CHAIN2=03f504f0a6e1b60b5b1fae5082d53a739aa259e7
COSYVOICE=/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS
log() { echo "$* $(date +%T)" >> $Q/queue_tokfp32.txt; }
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-items-chain
[ -d /workspace/t_chain2 ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_chain2 $CHAIN2
log "queue tokfp32 start t_chain2 $(git -C /workspace/t_chain2 log --oneline -1 | cut -c1-70) $(git -C /workspace/t_chain2 rev-parse HEAD:sglang_omni | cut -c1-8) $(git -C /workspace/t_chain2 diff --quiet HEAD -- sglang_omni || echo DIRTY)"
(cd /workspace/t_chain2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_chain2:$COSYVOICE taskset -c 0-15,64-79 \
  timeout 1800 python3 -m pytest tests/unit_test/fun_cosyvoice3/test_speech_tokenizer.py -q -p no:cacheprovider -rfEs \
  > $Q/tokfp32_units.log 2>&1)
log "units rc $? $(grep -E 'passed|failed' $Q/tokfp32_units.log | tail -n 1)"
(cd /workspace/t_chain2 && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_chain2:$COSYVOICE taskset -c 0-15,64-79 \
  timeout 1800 python3 -u $IN/tokenizer_parity_c20261020.py $Q/tokenizer_parity_fp32.json > $Q/tokenizer_parity_fp32.log 2>&1)
log "parity rc $? $(python3 -c "
import json; r = json.load(open('$Q/tokenizer_parity_fp32.json'))
print(' | '.join(f\"{k}: {v['tokens_differing']} in {v['references_differing']}\" for k, v in r['comparisons'].items()))
print(' | ms per call ' + ' '.join(f\"{k} {v['mean']}\" for k, v in r['ms_per_call'].items()))
" 2>&1 | tr '\n' ' ')"
log "queue tokfp32 done"
