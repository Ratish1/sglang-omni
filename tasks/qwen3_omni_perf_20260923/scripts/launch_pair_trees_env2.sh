#!/bin/sh
# Two trees, each arm with its own extra serve args. Usage:
#   sh launch_pair_trees_env2.sh <out_root> <round> <base_tree> <arm_tree> <base_card> <arm_card> <arm_name> "<base extra args>" "<arm extra args>" <bench_arm> [conc] [MAX_SAMPLES]
W=/workspace/sglang-omni/.tmp
S=$W/wt/tools/tasks/qwen3_omni_perf_20260923/scripts
R=$W/$1/r$2
BT=$3 AT=$4 BC=$5 AC=$6 AN=$7 BEXTRA=$8 AEXTRA=$9 ARM=${10} CONC=${11:-16}
mkdir -p $R
cd $W
git -C $BT log --oneline -1 > $R/base_head.txt
git -C $AT log --oneline -1 > $R/arm_head.txt
echo "$BEXTRA" > $R/base_extra_args.txt
echo "$AEXTRA" > $R/arm_extra_args.txt
pin() { case $1 in 0) echo "0-13,112-125 0";; 1) echo "14-27,126-139 0";; 2) echo "28-41,140-153 0";; 3) echo "56-69,168-181 1";; esac; }
bp=$(pin $BC); ap=$(pin $AC)
setsid env PIN_CPUS=${bp% *} PIN_NODE=${bp#* } ${12:+MAX_SAMPLES=${12}} EXTRA_SERVE_ARGS="$BEXTRA" \
  bash $S/run_bench_boot.sh $BT $R/base_c$BC $BC 8${BC}00 h200 $CONC $ARM > $R/base_c$BC.boot.log 2>&1 < /dev/null &
echo "base_c$BC pgid $!" >> $R/launch.txt
setsid env PIN_CPUS=${ap% *} PIN_NODE=${ap#* } ${12:+MAX_SAMPLES=${12}} EXTRA_SERVE_ARGS="$AEXTRA" \
  bash $S/run_bench_boot.sh $AT $R/${AN}_c$AC $AC 8${AC}00 h200 $CONC $ARM > $R/${AN}_c$AC.boot.log 2>&1 < /dev/null &
echo "${AN}_c$AC pgid $!" >> $R/launch.txt
sleep 2; cat $R/launch.txt $R/base_head.txt $R/arm_head.txt
