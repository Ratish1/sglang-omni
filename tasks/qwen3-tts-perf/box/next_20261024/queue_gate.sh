#!/usr/bin/env bash
# After #3102's merge (main 67476823a), the two built items left from the 10-22 profile, each cut from main: D8 (the
# speaker mel on the device, eager reference and speaker encodes past the graphs at recurring widths) and D3's first
# step (every layer's codec K and V in one arena tensor). Gate before the profile and the pairs, cards 2 and 3, one
# job at a time so no timing shares the host: units and the real-checkpoint tests on both branches; D8's speaker and
# reference encode benches (main, D8, D8, main); D8's distance to the model's fp32 preprocessing (embedding cosine and
# codes, main and D8 paired); whether a fresh FFT or conv plan waits for other streams' work, and in serving the other
# streams' longest chunk silence while long references of new widths are preprocessed; D3's arena identity
# against main (cohorts through gather, decode, scatter) and its decode bench (main, D3, D3, main). A branch whose
# units, gates or identity fail is flagged in $D/failed_<branch> and the later queues skip it.
set -u
D=/data/next1024
T=${T:?the kit tree}
HERE=$T/tasks/qwen3-tts-perf/box/chain_20261022
LONG=/data/d1merge/long
M=${M:?the main tree}
P8=${P8:?the D8 tree}
P3=${P3:?the D3 tree}
SNAP=$(ls -d /root/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/* | head -n 1)
L=$D/queue.log
mkdir -p $D/units
echo "queue gate start $(date +%T) main $(git -C $M rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d8 $(git -C $P8 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')d3 $(git -C $P3 rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')kit $(git -C $T rev-parse HEAD) loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $L
units() {
  local name=$1 tree=$2
  (cd $tree && CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=$tree timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts -q \
    -p no:cacheprovider -rfEs > $D/units/$name.log 2>&1)
  local rc=$?
  echo "units $name rc $rc $(grep -E 'passed|failed' $D/units/$name.log | tail -n 1) $(date +%T)" >> $L
  (cd $tree && CUDA_VISIBLE_DEVICES=2,3 QWEN3_TTS_TOKENIZER_PATH=$SNAP/speech_tokenizer QWEN3_TTS_BASE_1_7B_PATH=$SNAP \
    PYTHONPATH=$tree timeout 1800 python3 -m pytest tests/unit_test/qwen3_tts -m benchmark -k "test_real_" -q \
    -p no:cacheprovider -rfEs > $D/units/gates_$name.log 2>&1)
  local gates=$?
  echo "gates $name rc $gates $(grep -E 'passed|failed' $D/units/gates_$name.log | tail -n 1) $(date +%T)" >> $L
  if [ $rc -ne 0 ] || [ $gates -ne 0 ]; then
    touch $D/failed_$name
  else
    :
  fi
}
units d8 $P8
units d3 $P3
for arm in "main $M" "d8 $P8" "d8b $P8" "mainb $M"; do
  set -- $arm
  echo "== speaker bench seedtts $1" >> $D/speaker_bench.log
  (cd $2 && PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_seedtts_$1.pt --card 3 2>&1 \
    | grep -E "embed wall|graphs:|Error|Traceback" >> $D/speaker_bench.log)
  echo "== speaker bench L3 $1" >> $D/speaker_bench.log
  (cd $2 && PYTHONPATH=$2 python3 $HERE/speaker_embed_bench.py $D/speaker_L3_$1.pt --card 3 --repeats 5 \
    --meta $LONG/L3_c16_run.lst >> $D/speaker_bench.log 2>&1)
  echo "== reference batcher $1" >> $D/reference_batcher_bench.log
  (cd $2 && PYTHONPATH=$2 python3 $HERE/reference_batcher_bench.py --card 3 --warm $LONG/L3_c16_warm.lst \
    --run $LONG/L3_c16_run.lst >> $D/reference_batcher_bench.log 2>&1)
  echo "bench $1 $(date +%T)" >> $L
done
for pair in "main d8" "main mainb" "d8 d8b"; do
  set -- $pair
  python3 $HERE/speaker_embed_bench.py --compare $D/speaker_seedtts_$1.pt $D/speaker_seedtts_$2.pt \
    > $D/speaker_compare_seedtts_$1_$2.txt 2>&1
  python3 $HERE/speaker_embed_bench.py --compare $D/speaker_L3_$1.pt $D/speaker_L3_$2.pt > $D/speaker_compare_L3_$1_$2.txt 2>&1
done
(cd $M && PYTHONPATH=$M timeout 3600 python3 $HERE/preprocessing_truth_probe.py $D/truth_fp32.pt --truth --card 3 \
  --meta $LONG/L1_c16_run.lst $LONG/L3_c16_run.lst > $D/truth_fp32.log 2>&1)
echo "truth probe fp32 rc $? $(date +%T)" >> $L
for arm in "main $M" "d8 $P8"; do
  set -- $arm
  (cd $2 && PYTHONPATH=$2 timeout 3600 python3 $HERE/preprocessing_truth_probe.py $D/truth_$1.pt --card 3 \
    --meta $LONG/L1_c16_run.lst $LONG/L3_c16_run.lst > $D/truth_$1.log 2>&1)
  echo "truth probe $1 rc $? $(date +%T)" >> $L
done
(cd $M && PYTHONPATH=$M python3 $HERE/preprocessing_truth_probe.py --compare $D/truth_fp32.pt $D/truth_main.pt \
  $D/truth_d8.pt > $D/truth_compare.txt 2>&1)
echo "truth compare rc $? $(date +%T)" >> $L
python3 $HERE/plan_wait_probe.py --card 3 > $D/plan_wait.txt 2>&1
echo "plan wait rc $? $(date +%T)" >> $L
# note (ratish): a new FFT size waits for the whole device (plan_wait.txt), and D8's mel past the graphs makes one
# per new rounded width, so the served cost of each width's first clip is read here, both card orders.
FW=$D/fresh_width
(cd $M && PYTHONPATH=$M python3 $T/tasks/qwen3-tts-perf/box/next_20261024/fresh_width_meta.py $FW > $D/fresh_width_meta.log 2>&1)
echo "fresh width list rc $? $(tail -n 1 $D/fresh_width_meta.log) $(date +%T)" >> $L
export PAIR_ROOT=$D/pairs
for spec in "fw_a 2 3 2400" "fw_b 3 2 3600"; do
  set -- $spec
  python3 $T/tasks/qwen3-tts-perf/box/v3_20261020/pair.py --label $1 --tree $M --b-tree $P8 --concurrency 16 \
    --a-card $2 --b-card $3 --port $4 --meta $FW/fresh_width.lst --events --skip-score > $D/$1.out 2>&1
  echo "$1 rc $? $(date +%T)" >> $L
  python3 $T/tasks/qwen3-tts-perf/box/next_20261024/arrival_stalls.py $1 > $D/stalls_$1.txt 2>&1
done
for arm in "main $M" "d3 $P3"; do
  set -- $arm
  (cd $2 && PYTHONPATH=$2 python3 $HERE/vocoder_identity_probe.py $D/identity_$1.pt --card 3 > $D/identity_$1.log 2>&1)
  echo "identity $1 rc $? $(date +%T)" >> $L
done
python3 $HERE/vocoder_identity_probe.py --compare $D/identity_main.pt $D/identity_d3.pt > $D/identity_compare.txt 2>&1
echo "identity compare: $(tail -n 1 $D/identity_compare.txt) $(date +%T)" >> $L
if grep -q "^differs" $D/identity_compare.txt || ! grep -q "entries bit identical" $D/identity_compare.txt; then
  touch $D/failed_d3
else
  :
fi
for arm in "main $M" "d3 $P3" "d3b $P3" "mainb $M"; do
  set -- $arm
  echo "== $1" >> $D/decode_bench.log
  (cd $2 && TORCHINDUCTOR_CACHE_DIR=$D/inductor_bench_$(basename $2) PYTHONPATH=$2 python3 $HERE/vocoder_decode_bench.py --card 3 \
    2>&1 | grep -E "^(cold|warm|tail) |Error|Traceback" >> $D/decode_bench.log)
  echo "decode bench $1 $(date +%T)" >> $L
done
echo "queue_gate_done $(date +%T) failed [$(ls $D | grep '^failed_' | tr '\n' ' ')]" >> $L
