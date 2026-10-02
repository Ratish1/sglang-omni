#!/bin/sh
# Two trees, each arm with its own extra serve args. Usage:
#   sh launch_pair_trees_env2.sh <out_root> <round> <base_tree> <arm_tree> <base_card> <arm_card> <arm_name> "<base extra args>" "<arm extra args>" <bench_arm> [conc] [MAX_SAMPLES]
# DTYPE picks the config and the CPU pin map: h200 (default) or bf16 / fp8 on the H100 host,
# PIN_LAYOUT=numa1 the H200 map for cards that all sit on NUMA node 1,
# whose four leased cards all sit on NUMA node 0 (CPUs 0-31,64-95).
DTYPE=${DTYPE:-h200}
# PORT_BASE moves the servers off 8000 to 8300 (and their scorers off 8100 to 8400) when another
# tenant on the host network holds those ports. A card's scorer takes its server port plus 100,
# which is the next card's server port; PORT_STEP=200 keeps a pair on adjacent cards apart.
PORT_BASE=${PORT_BASE:-8000}
PORT_STEP=${PORT_STEP:-100}
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
if [ "$DTYPE" = h200 ] && [ "${PIN_LAYOUT:-}" = numa1 ]; then
  # node-radixark-16-0001's leased cards 4 to 7: all on NUMA node 1 (CPUs 56-111,168-223)
  pin() { case $1 in 0) echo "56-69,168-181 1";; 1) echo "70-83,182-195 1";; 2) echo "84-97,196-209 1";; 3) echo "98-111,210-223 1";; esac; }
elif [ "$DTYPE" = h200 ]; then
  pin() { case $1 in 0) echo "0-13,112-125 0";; 1) echo "14-27,126-139 0";; 2) echo "28-41,140-153 0";; 3) echo "56-69,168-181 1";; esac; }
else
  pin() { case $1 in 0) echo "0-7,64-71 0";; 1) echo "8-15,72-79 0";; 2) echo "16-23,80-87 0";; 3) echo "24-31,88-95 0";; esac; }
fi
bp=$(pin $BC); ap=$(pin $AC)
setsid env PIN_CPUS=${bp% *} PIN_NODE=${bp#* } ${12:+MAX_SAMPLES=${12}} EXTRA_SERVE_ARGS="$BEXTRA" \
  bash $S/run_bench_boot.sh $BT $R/base_c$BC $BC $((PORT_BASE + BC * PORT_STEP)) $DTYPE $CONC "$ARM" > $R/base_c$BC.boot.log 2>&1 < /dev/null &
echo "base_c$BC pgid $!" >> $R/launch.txt
setsid env PIN_CPUS=${ap% *} PIN_NODE=${ap#* } ${12:+MAX_SAMPLES=${12}} EXTRA_SERVE_ARGS="$AEXTRA" \
  bash $S/run_bench_boot.sh $AT $R/${AN}_c$AC $AC $((PORT_BASE + AC * PORT_STEP)) $DTYPE $CONC "$ARM" > $R/${AN}_c$AC.boot.log 2>&1 < /dev/null &
echo "${AN}_c$AC pgid $!" >> $R/launch.txt
sleep 2; cat $R/launch.txt $R/base_head.txt $R/arm_head.txt
