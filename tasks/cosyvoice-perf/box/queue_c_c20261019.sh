#!/usr/bin/env bash
# 10-19, cards 2 and 3: queue b again on S2's head 609ce55be (tree t_s2b, the flow noise on the device added), against
# main 0c521f191 (t_m19, whose drive and units queue b already took). Card 3: the drive with every synchronizing call
# listed, its audio compared with main's bit for bit; card 2 at the same time: the touched suites. Then the warm smoke
# of t_s2b, the pair p_s2 (c1 seeded, c1u, c16, c64, r16, both orders) and S2's profile on card 2.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261019/queue_c.pgid; exec bash queue_c_c20261019.sh'
D=/data/c20261019
IN=$D/in
M=t_m19
B=t_s2b
R=$D/s2
log() { echo "$* $(date +%T)" >> $D/queue_c.txt; }

log "queue start"
(
  cd $R
  CUDA_VISIBLE_DEVICES=3 PYTHONPATH=/workspace/$B taskset -c 16-31,80-95 python3 -u \
    $IN/s2_vocoder_drive_c20261019.py --check-waits --out $R/drive_$B.npz > $R/drive_$B.log 2>&1
  echo "drive $B rc $? $(grep '^steps' $R/drive_$B.log)" >> $D/queue_c.txt
  python3 $IN/s2_vocoder_drive_c20261019.py --compare $R/drive_$M.npz $R/drive_$B.npz > $R/drive_compare_$B.txt 2>&1
  echo "drive compare $(head -n 1 $R/drive_compare_$B.txt) $(date +%T)" >> $D/queue_c.txt
) &
(
  cd /workspace/$B
  CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/workspace/$B taskset -c 0-15,64-79 python3 -m pytest -q -p no:cacheprovider \
    tests/unit_test/fun_cosyvoice3 tests/unit_test/scheduling/test_streaming_vocoder.py > $R/units_$B.log 2>&1
  echo "units $B rc $? $(grep -E 'passed|failed' $R/units_$B.log | tail -n 1)" >> $D/queue_c.txt
) &
wait

bash $IN/warm_c20261019.sh $B 2 0-15,64-79 26200 $D/warm_s2b
log "warm $(grep 'bench rc' $D/warm_s2b/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/warm_s2b/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/warm_s2b/serve.log)"

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
