#!/usr/bin/env bash
# Block 47 of 10-31: the suites of the CPU-side startup fixes, card 1. LAUNCHER: the Qwen3-TTS suite; its new test also
# on main, where it must fail. MING: the Ming-TTS and Ming-Omni suites; its new test also on main. FREEZE: the pipeline
# suite. Waits for block 46 first.
# usage: D=... MAIN=<tree> LAUNCHER=<tree> MING=<tree> FREEZE=<tree> bash queue_block47.sh
set -u
D=${D:?the results directory}
MAIN=${MAIN:?the main tree}
LAUNCHER=${LAUNCHER:?the launcher tree}
MING=${MING:?the Ming tree}
FREEZE=${FREEZE:?the freeze tree}
W=$D/block47
L=$W/queue.log
mkdir -p $W
until grep -q "queue block46 done" $D/block46/queue.log 2> /dev/null; do sleep 20; done
for arm in MAIN LAUNCHER MING FREEZE; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ')$(git -C $tree diff --quiet HEAD -- sglang_omni || echo DIRTY)" >> $L
done
echo "queue block47 start $(date +%T)" >> $L

suite() {
  local name=$1 tree=$2
  shift 2
  (cd $tree && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$tree timeout 3600 python3 -m pytest -q -p no:cacheprovider "$@" > $W/$name.log 2>&1)
  echo "$name exit $? $(grep -E '(passed|failed|error)' $W/$name.log | tail -n 1) $(date +%T)" >> $L
}

on_main() {
  local name=$1 source=$2 file=$3 test=$4
  cp $source/$file $MAIN/$file
  suite $name $MAIN $file::$test
  git -C $MAIN checkout -q -- tests
}

suite launcher_suite $LAUNCHER tests/unit_test/qwen3_tts
on_main launcher_test_on_main $LAUNCHER tests/unit_test/qwen3_tts/test_config.py test_admission_defaults_load_without_the_engine
suite ming_suite $MING tests/unit_test/ming_tts tests/unit_test/ming_omni
on_main ming_test_on_main $MING tests/unit_test/ming_tts/test_sglang_model.py test_model_class_imports_without_the_audio_vae
suite freeze_suite $FREEZE tests/unit_test/pipeline
echo "queue block47 done $(date +%T)" >> $L
