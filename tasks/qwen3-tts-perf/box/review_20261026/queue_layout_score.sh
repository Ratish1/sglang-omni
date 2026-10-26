#!/usr/bin/env bash
# Scoring for queue_layout_rest.sh, whose single pass ran out of memory: the ASR server (84 GB at its default static
# fraction) and three official processes left no room for the similarity model. Every served arm is transcribed first
# (shipped and the proposed configuration first), the ASR server stops, then similarity runs; the official arms still
# running finish beside it, any that failed rerun alone, and the summary is rebuilt. Card 1 only.
set -u
D=${D:?the layout results directory}
P=${P:?the probe tree}
T=${T:?the kit tree}
REFPY=${REFPY:?the reference venv python}
CASES=${CASES:?the parity cases}
WAIT_PIDS=${WAIT_PIDS:?the official processes still running}
HERE=$T/tasks/qwen3-tts-perf/box/review_20261026
CARD=1
CAP=300
L=$D/queue.log
echo "queue layout score start $(date +%T) probe $(git -C $P rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD)" >> $L
start_server() {
  local label=$1 port=$2
  shift 2
  (cd $P && SGLANG_OMNI_STRICT_PORT=1 PYTHONPATH=$P setsid bash -c "echo \$\$ > $D/serve_$label.pid; exec python3 \
    -m sglang_omni.cli serve --host 127.0.0.1 --port $port $*" > $D/serve_$label.log 2>&1 < /dev/null &)
  sleep 2
  local pgid=$(cat $D/serve_$label.pid)
  local waited=0
  until curl -sf http://127.0.0.1:$port/health > /dev/null; do
    if ! kill -0 $pgid 2> /dev/null || [ $waited -ge 1800 ]; then
      echo "$label server not healthy after $waited s; $(grep -E 'Error|error' $D/serve_$label.log | tail -n 2 | tr '\n' ' ')" >> $L
      kill -TERM -$pgid 2> /dev/null
      return 1
    else
      sleep 5
      waited=$((waited + 5))
    fi
  done
  echo "$label healthy after $waited s, pgid $pgid, card $(nvidia-smi -i $CARD --query-gpu=memory.used --format=csv,noheader) $(date +%T)" >> $L
}
stop_server() {
  local pgid=$(cat $D/serve_$1.pid)
  kill -TERM -$pgid 2> /dev/null
  for second in $(seq 120); do
    kill -0 $pgid 2> /dev/null || break
    sleep 1
  done
  kill -KILL -$pgid 2> /dev/null
  echo "$1 stopped $(date +%T)" >> $L
}
run_official() {
  local size=$1 model=$2 language=$3 layout=$4
  local flags=""
  if [ $layout = whole ]; then
    flags="--whole-text"
  else
    :
  fi
  (cd $D && $REFPY $HERE/parity_reference.py $CASES $D/official_${size}_${language}_${layout}.json --model $model \
    --card $CARD --batch 64 --max-new-tokens $CAP --language $language $flags > $D/official_${size}_${language}_${layout}.log 2>&1)
  local rc=$?
  echo "official $size $language $layout rc $rc $(grep -c 'reached' $D/official_${size}_${language}_${layout}.log) batches $(date +%T)" >> $L
  return $rc
}
DIRS=""
for size in 06 17; do
  for arm in shipped min2_nomask; do
    for language in English auto; do
      DIRS="$DIRS $D/served/${size}_${arm}_$language"
    done
  done
done
for size in 06 17; do
  for arm in nomask min2 whole whole_nomask whole_min2; do
    for language in English auto; do
      DIRS="$DIRS $D/served/${size}_${arm}_$language"
    done
  done
done
export MASTER_PORT=27850
if start_server asr 27800 --model-path Qwen/Qwen3-ASR-1.7B --asr.gpu $CARD; then
  for dir in $DIRS; do
    (cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $dir --asr-port 27800 >> $L 2>&1)
    echo "transcribed $(basename $dir) rc $? $(date +%T)" >> $L
  done
  stop_server asr
else
  :
fi
for dir in $DIRS; do
  (cd $P && PYTHONPATH=$P python3 $HERE/layout.py score $dir --sim-card $CARD >> $L 2>&1)
  echo "similarity $(basename $dir) rc $? $(date +%T)" >> $L
done
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py summary $D/official_*.json --served $DIRS --max-new-tokens $CAP \
  > $D/summary.txt 2>&1)
echo "summary scored rc $? $(date +%T)" >> $L
for pid in $WAIT_PIDS; do
  while kill -0 $pid 2> /dev/null; do
    sleep 10
  done
done
for arm in "06|Qwen/Qwen3-TTS-12Hz-0.6B-Base|auto|whole" "17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|auto|stream" \
  "17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|English|whole" "17|Qwen/Qwen3-TTS-12Hz-1.7B-Base|auto|whole"; do
  IFS='|' read -r size model language layout <<< "$arm"
  if [ ! -s $D/official_${size}_${language}_${layout}.json ]; then
    echo "official $size $language $layout again, alone on the card" >> $L
    run_official $size $model $language $layout
  else
    :
  fi
done
(cd $P && PYTHONPATH=$P python3 $HERE/layout.py summary $D/official_*.json --served $DIRS --max-new-tokens $CAP \
  > $D/summary.txt 2>&1)
echo "summary final rc $? $(date +%T)" >> $L
echo "queue_layout_score_done $(date +%T)" >> $L
