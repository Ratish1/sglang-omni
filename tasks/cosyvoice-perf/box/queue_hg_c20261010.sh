#!/usr/bin/env bash
# Step 2 (HiFT from CUDA graphs, on step 1) verification, tree t_hg (main ba456b68f with
# hift_window_graphs_v1.patch). Card 0: the graph bench (eager against replay on the same shapes,
# capture time and memory), the real speech accuracy through the graphs, then seeded stream c1 under
# nsys and the AR decode steps beside vocoder steps (main's trace is prof2_c1_t_main). Cards 1 and 2:
# t_main against t_hg in both orders, seeded stream c1, stream c16 (WER and SIM), stream c64, long
# form c16.
# usage (in the container): bash queue_hg_c20261010.sh
R=/data/c20261010h
IN=$R/in
mkdir -p $R/bench
card0() {
  (
    cd /workspace/t_hg
    export PYTHONPATH=/workspace/t_hg
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 $IN/hift_graph_bench_c20261010.py \
      --out $R/bench/t_hg_graphs.json > $R/bench/t_hg_graphs.log 2>&1
    echo "graph bench rc $? $(date +%T)" >> $R/queue_hg.txt
    CUDA_VISIBLE_DEVICES=0 taskset -c 0-15,64-79 python3 $IN/hift_graph_bench_c20261010.py \
      --accuracy --out $R/bench/t_hg_graphs_acc.json > $R/bench/t_hg_graphs_acc.log 2>&1
    echo "graph accuracy rc $? $(date +%T)" >> $R/queue_hg.txt
  )
  bash $IN/cosy_profile_c1_c20261010.sh t_hg 0 0-15,64-79 34000 $R/prof2_c1_t_hg
  python3 $R/an/ar_overlap.py $R/prof2_c1_t_hg/c1.sqlite > $R/prof2_c1_t_hg/ar_overlap.txt 2>&1
  echo "profile rc $? $(date +%T)" >> $R/queue_hg.txt
}
echo "start $(date +%T) $(cat /workspace/t_hg/BRANCH_MARKER.txt)" >> $R/queue_hg.txt
card0 &
CARD0=$!
export ROOT=$R
export CELLS="c1:1:64:1234:1 c16:16:::1 c64:64:::1 L16:16:::1:/data/c20261010/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=1 CARD_1=2 CPUS_0=32-47,96-111 CPUS_1=48-63,112-127"
echo "hga $(date +%T) card1=t_main card2=t_hg" >> $R/queue_hg.txt
env $PAIR bash $IN/ab_pair.sh $R/hga t_main t_hg 33000
echo "hgb $(date +%T) card1=t_hg card2=t_main" >> $R/queue_hg.txt
env $PAIR bash $IN/ab_pair.sh $R/hgb t_hg t_main 33000
wait $CARD0
echo "queue done $(date +%T)" >> $R/queue_hg.txt
