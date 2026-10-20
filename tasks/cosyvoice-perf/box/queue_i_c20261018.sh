#!/usr/bin/env bash
# 10-18, cards 2 and 3: where the c16 inter chunk gap goes. Trees t_main4 (upstream main 76de3ff8f) and t_ic
# (prof/cosyvoice3-inter-chunk d50266ba6: #2910 5d1e2317e with main merged). Warm both (the smoke test), then the kit
# profile of both at the same time: c16 closed loop and r16 open loop (16 requests a second, one arrival schedule),
# 640 requests each, py-spy at c16; each chunk split into the AR's and the vocoder's part, the first audio timeline,
# the card's busy span. Then the open loop pair without the profiler (r12, r16, both orders) and its gap breakdown.
# usage (in the container): setsid bash -c 'echo $$ > /data/c20261018/queue_i.pgid; exec bash queue_i_c20261018.sh'
D=/data/c20261018
IN=$D/in
M=t_main4
B=t_ic
CELLS="c16:16:640 r16:64:640::16"
log() { echo "$* $(date +%T)" >> $D/queue_i.txt; }

log "queue start"
bash $IN/warm_c20261018.sh $M 2 0-15,64-79 26100 $D/warm_main4 &
bash $IN/warm_c20261018.sh $B 3 16-31,80-95 26500 $D/warm_ic &
wait
for w in warm_main4 warm_ic; do
  log "warm $w $(grep 'bench rc' $D/$w/progress.txt) $(grep -E 'Completed requests|Failed requests' $D/$w/bench.log | tr -s ' ' | tr '\n' ' ') tracebacks $(grep -c Traceback $D/$w/serve.log)"
done

mkdir -p $D/prof_ic
PEER=$D/prof_ic/ic PYSPY=1 CELLS="$CELLS" bash $IN/cosy_profile_run_c20261018.sh $M 2 0-15,64-79 27100 \
  $D/prof_ic/main > $D/prof_ic/run_main.log 2>&1 &
PEER=$D/prof_ic/main PYSPY=1 CELLS="$CELLS" bash $IN/cosy_profile_run_c20261018.sh $B 3 16-31,80-95 27500 \
  $D/prof_ic/ic > $D/prof_ic/run_ic.log 2>&1 &
wait
log "profiles $(tail -n 1 $D/prof_ic/main/progress.txt) / $(tail -n 1 $D/prof_ic/ic/progress.txt)"
for cell in c16 r16; do
  python3 $IN/chunk_timeline_c20261018.py main=$D/prof_ic/main/$cell.sqlite ic=$D/prof_ic/ic/$cell.sqlite \
    > $D/prof_ic/chunks_$cell.txt 2>&1
  python3 $IN/first_audio_timeline_c20261017.py main=$D/prof_ic/main/$cell.sqlite ic=$D/prof_ic/ic/$cell.sqlite \
    > $D/prof_ic/timeline_$cell.txt 2>&1
  python3 $IN/card_busy_span_c20261017.py main=$D/prof_ic/main/$cell.sqlite ic=$D/prof_ic/ic/$cell.sqlite \
    > $D/prof_ic/card_busy_$cell.txt 2>&1
  log "analyses $cell"
done

(cd /workspace/sglang-omni && PAIR_CELLS=r12,r16 python3 $IN/serve_pair_c20261018.py /workspace/$M /workspace/$B \
  $D/p_ol > $D/p_ol.log 2>&1)
log "pair p_ol rc $?"
python3 $IN/inter_chunk_mechanism_c20261018.py $D/p_ol 76de3ff8f d50266ba6 r12 r16 > $D/p_ol_mechanism.txt 2>&1
log "queue done"
