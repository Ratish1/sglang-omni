#!/usr/bin/env bash
# 10-22, host-85-234-79-221, cards 0 and 1: tasks/cosyvoice-perf/profile_20261022/PLAN.md. The items chain
# (perf/cosyvoice3-items-chain: main with L2 to L7a) as t_chain: its touched units, the long cells' lists, a warm, the
# formal pass alone on card 0, then the mapping pass (card 0) beside the memory pass (card 1), then the nsys pass alone
# on card 0 (the pipeline probe, graphs whole, py-spy at c16), then the readers: gates, the pool per cell, BBuf's
# analyzer per formal and mapping pair, the nsys ledgers (the runner's own).
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261022b/queue_prof.pgid; exec bash /workspace/t_marks/tasks/cosyvoice-perf/box/profile_c20261022/queue_profile.sh'
D=/data/c20261022b
P=$D/prof
IN=/workspace/t_marks/tasks/cosyvoice-perf/box
HERE=$IN/profile_c20261022
B=/data/bbuf/AI-Infra-Auto-Driven-SKILLS/skills/llm-torch-profiler-analysis
export PAIR_HOME=$D
mkdir -p $P
log() { echo "$* $(date +%T)" >> $P/queue.txt; }
cpus() { [ "$1" = 0 ] && echo 0-15,64-79 || echo 16-31,80-95; }
git -C /workspace/sglang-omni fetch -q origin perf/cosyvoice3-items-chain
sha=$(git -C /workspace/sglang-omni rev-parse FETCH_HEAD)
if [ -d /workspace/t_chain ]; then
  git -C /workspace/t_chain checkout -q --detach $sha
else
  git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_chain $sha
fi
log "queue profile start t_marks $(git -C /workspace/t_marks rev-parse --short HEAD) t_chain $(git -C /workspace/t_chain log --oneline -1 | cut -c1-60) sglang_omni $(git -C /workspace/t_chain rev-parse HEAD:sglang_omni | cut -c1-8) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
grep -q sglang_omni $B/scripts/profile_common.py || { log "backend predates BBuf #96, stop"; exit 1; }

(cd /workspace/t_chain && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_chain:/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS \
  timeout 3600 python3 -m pytest tests/unit_test/fun_cosyvoice3 tests/unit_test/test_platforms.py -q -p no:cacheprovider -rfEs \
  > $P/units.log 2>&1)
units_rc=$?
log "units rc $units_rc $(grep -E 'passed|failed' $P/units.log | tail -n 1)"
[ $units_rc = 0 ] || { log "units failed, stop"; exit 1; }

(cd /workspace/t_chain && PYTHONPATH=/workspace/t_chain python3 $HERE/long_cells_meta.py $P/meta > $P/meta.log 2>&1)
log "meta rc $? $(tail -n 1 $P/meta.log)"

bash $IN/warm_c20261021.sh t_chain 0 $(cpus 0) 29400 $D/warm_t_chain
log "warm $(grep 'bench rc' $D/warm_t_chain/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_t_chain/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_t_chain/serve.log)"

profile() {
  local mode=$1 card=$2 port=$3
  (cd /workspace/t_chain && PYTHONPATH=/workspace/t_chain python3 $HERE/profile_cells.py --tree /workspace/t_chain \
    --card $card --port $port --mode $mode --out $P/$mode --meta $P/meta > $P/$mode.out 2>&1)
  log "$mode rc $? $(tr '\n' ' ' < $P/$mode.out | cut -c1-400)"
}
profile formal 0 27000
profile mapping 0 28200 &
profile memory 1 27600 &
wait

M=$P/meta
CELLS="S_c16:16:320 warm_P_c16:16:32:::$M/P_c16_warm.lst P_c16:16:96:::$M/P_c16_run.lst \
warm_D_c64:64:16:::$M/D_c64_warm.lst D_c64:64:64:::$M/D_c64_run.lst \
warm_PD_c16:16:16:::$M/PD_c16_warm.lst PD_c16:16:32:::$M/PD_c16_run.lst \
warm_Db_c16:16:16:::$M/Db_c16_warm.lst:b Db_c16:16:32:::$M/Db_c16_run.lst:b"
CELLS="$CELLS" bash $IN/cosy_profile_run_c20261021.sh t_chain 0 $(cpus 0) 28800 $P/nsys
log "nsys $(tail -n 3 $P/nsys/progress.txt | tr '\n' ' ')"

python3 $HERE/trace_gates.py $P/formal $P/mapping > $P/gates.txt 2>&1
log "gates rc $? fails $(grep -c FAIL $P/gates.txt)"
python3 $HERE/pool_summary.py $P/memory > $P/pool.txt 2>&1
log "pool rc $?"
for cell_dir in $P/formal/*/; do
  cell=$(basename $cell_dir)
  [ -d $P/mapping/$cell ] || continue
  python3 $B/scripts/analyze_llm_torch_profile.py --framework sglang --mapping-input $P/mapping/$cell \
    --formal-input $P/formal/$cell > $P/report_$cell.md 2>&1
  log "report $cell rc $?"
done
log "queue profile done"
