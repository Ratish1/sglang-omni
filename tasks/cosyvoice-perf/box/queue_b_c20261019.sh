#!/usr/bin/env bash
# 10-19, cards 2 and 3: S2 (perf/cosyvoice3-vocoder-overlap, tree t_s2) against main 0c521f191 (t_m19).
# Card 3: the checkpoint's vocoder driven by both trees from one fixed set of streams (s2_vocoder_drive), every
# synchronizing call site listed per tree, the audio of every request compared bit for bit. Card 2 at the same time:
# the touched unit suites (fun_cosyvoice3, the streaming vocoder base) on both trees. Then the warm smoke of t_s2, the
# pair p_s2 (c1 seeded, c1u, c16, c64, r16, both orders), and S2's profile (the S0 cells) on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261019/queue_b.pgid; exec bash queue_b_c20261019.sh'
D=/data/c20261019
IN=$D/in
M=t_m19
B=t_s2
R=$D/s2
mkdir -p $R
log() { echo "$* $(date +%T)" >> $D/queue_b.txt; }

log "queue start"
(
  cd $R
  for tree in $M $B; do
    CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/$tree taskset -c 16-31,80-95 python3 -u \
      $IN/s2_vocoder_drive_c20261019.py --check-waits --out $R/drive_$tree.npz > $R/drive_$tree.log 2>&1
    echo "drive $tree rc $? $(tail -n 2 $R/drive_$tree.log | tr '\n' ' ')" >> $D/queue_b.txt
  done
  python3 $IN/s2_vocoder_drive_c20261019.py --compare $R/drive_$M.npz $R/drive_$B.npz > $R/drive_compare.txt 2>&1
  echo "drive compare $(head -n 1 $R/drive_compare.txt) $(date +%T)" >> $D/queue_b.txt
) &
(
  for tree in $B $M; do
    cd /workspace/$tree
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$tree taskset -c 0-15,64-79 python3 -m pytest -q -p no:cacheprovider \
      tests/unit_test/fun_cosyvoice3 tests/unit_test/scheduling/test_streaming_vocoder.py > $R/units_$tree.log 2>&1
    echo "units $tree rc $? $(tail -n 1 $R/units_$tree.log)" >> $D/queue_b.txt
  done
) &
wait

bash $IN/warm_c20261019.sh $B 2 0-15,64-79 26200 $D/warm_s2
log "warm $(grep 'bench rc' $D/warm_s2/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_s2/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_s2/serve.log)"

(cd /workspace/sglang-omni && PAIR_CELLS=c1,c1u,c16,c64,r16 python3 $IN/serve_pair_c20261019.py /workspace/$M \
  /workspace/$B $D/p_s2 > $D/p_s2.log 2>&1)
log "pair p_s2 rc $?"

mkdir -p $D/prof_s2
PYSPY=1 CELLS="c16:16:320 c64:64:640 L16:16:48:::$D/longform/meta.lst" \
  bash $IN/cosy_profile_run_c20261019.sh $B 2 0-15,64-79 27200 $D/prof_s2/s2 > $D/prof_s2/run_s2.log 2>&1
log "profile $(tail -n 1 $D/prof_s2/s2/progress.txt)"
for cell in c16 c64 L16; do
  python3 $IN/chunk_timeline_c20261018.py s2=$D/prof_s2/s2/$cell.sqlite > $D/prof_s2/chunks_$cell.txt 2>&1
done
log "queue done"
