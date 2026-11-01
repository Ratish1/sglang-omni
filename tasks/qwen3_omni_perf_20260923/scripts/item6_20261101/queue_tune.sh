set -u
# 11-01 item 6: whether the thinker MoE's small-M GEMMs are bound by their tiling under Triton 3.8.0 (the served H100
# tables were tuned under 3.7.1 and loaded by fallback; SGLang's log says to re-tune). SGLang's own tuner
# (benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py) on the thinker's text config presented as
# Qwen3MoeForCausalLM, token counts 1 to 16, BLOCK_SIZE_M 16 only (at most two rows an expert there), both cards.
# Then moe_config_probe.py with the tuned table as an overlay (SGLANG_MOE_CONFIG_DIR, the served 3.7.1 down table
# copied beside it) against SGLang's own tables, on card 0.
Q=/data/i6_1101
S=/data/tools/tasks/qwen3_omni_perf_20260923/scripts
TUNER=/sgl-workspace/sglang/benchmark/kernels/fused_moe_triton
MODEL=$Q/thinker_text_model
OVERLAY=$Q/overlay
log() { echo "$(date -u +%T) $*" >> $Q/queue.log; }
mkdir -p $MODEL $Q/tune
python3 - "$MODEL" <<'EOF'
import json
import sys
from huggingface_hub import snapshot_download

source = json.load(open(f"{snapshot_download('Qwen/Qwen3-Omni-30B-A3B-Instruct')}/config.json"))
text = dict(source["thinker_config"]["text_config"])
text["architectures"] = ["Qwen3MoeForCausalLM"]
text["model_type"] = "qwen3_moe"
text["torch_dtype"] = "bfloat16"
json.dump(text, open(f"{sys.argv[1]}/config.json", "w"), indent=1)
EOF
python3 - "$Q/tune/search_space.json" <<'EOF'
import itertools
import json
import sys

space = [
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k, "GROUP_SIZE_M": g, "num_warps": w, "num_stages": s}
    for s, k, n, w, g in itertools.product([2, 3, 4, 5], [64, 128, 256], [32, 64, 128, 256], [4, 8], [1, 16, 32, 64])
]
json.dump(space, open(sys.argv[1], "w"))
print(len(space))
EOF
log "tune start: tools $(git -C /data/tools rev-parse --short HEAD); apps [$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | tr '\n' ';')]"
(cd $Q/tune && python3 $TUNER/tuning_fused_moe_triton.py --model $MODEL --tp-size 1 --tune \
  --batch-sizes 1 2 3 4 6 8 12 16 --search-space-file $Q/tune/search_space.json > $Q/tune/tune.out 2> $Q/tune/tune.err)
log "tune rc $?; $(grep -E 'Writing best config|Tuning took' $Q/tune/tune.out | tr '\n' ' ')"
TABLE=$(ls $Q/tune/E=128,N=768,*.json 2>/dev/null | head -1)
SERVED=/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/triton_3_7_1
mkdir -p $OVERLAY/configs/triton_3_8_0
python3 - "$TABLE" "$SERVED/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json" "$OVERLAY/configs/triton_3_8_0/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json" <<'EOF'
import json
import sys

tuned, served, out = sys.argv[1:]
table = json.load(open(served))
table.update(json.load(open(tuned)))
json.dump(table, open(out, "w"), indent=1)
EOF
cp "$SERVED/E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3_down.json" $OVERLAY/configs/triton_3_8_0/
log "overlay: $(ls $OVERLAY/configs/triton_3_8_0 | tr '\n' ' ')"
for arm in served tuned; do
  if [ $arm = tuned ]; then export SGLANG_MOE_CONFIG_DIR=$OVERLAY; else unset SGLANG_MOE_CONFIG_DIR; fi
  taskset -c 0-15,64-79 python3 $S/moe_config_probe.py --label $arm --bandwidth-tbps 3.10 > $Q/tune/probe_$arm.out 2>&1
  log "probe $arm rc $?"
  grep -E "^tokens +(1|2|3|4|6|8|12|16) " $Q/tune/probe_$arm.out | while read -r line; do log "  $arm $line"; done
done
log "tune done"
