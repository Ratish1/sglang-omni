#!/usr/bin/env bash
# Decode-step component costs (component_costs.py) on main, S1, S2a and S3, each tree twice in mirrored order on card 0
# with card 1 idle, then the S3 head's touched units (its test doubles fixed in f3f922644). Waits for queue_s3.
set -u
D=/data/async_decode_20261020
IN=${T:-/workspace/t_marks}/tasks/qwen3-tts-perf/box/async_decode_20261020
S3_HEAD=/workspace/s3_f3f922644
L=$D/queue.log
mkdir -p $D/components
until grep -q queue_s3_done $L; do sleep 30; done
echo "queue components start $(date +%T)" >> $L
for ROUND in 1 2; do
  if [ $ROUND = 1 ]; then
    ORDER="m_bcd0d8c68 s1_d1eef0380 s2a_9db6227f1 s3_ce54a94fb"
  else
    ORDER="s3_ce54a94fb s2a_9db6227f1 s1_d1eef0380 m_bcd0d8c68"
  fi
  for NAME in $ORDER; do
    (cd /workspace/$NAME && PYTHONPATH=/workspace/$NAME timeout 1800 python3 $IN/component_costs.py --card 0 \
      > $D/components/${NAME}_r$ROUND.txt 2>&1)
    echo "components $NAME r$ROUND rc $? $(date +%T)" >> $L
  done
done
(cd $S3_HEAD && PYTHONPATH=$S3_HEAD timeout 3600 python3 -m pytest tests/unit_test/qwen3_tts tests/unit_test/model_runner \
  tests/unit_test/scheduling -q -p no:cacheprovider -rfE > $D/units/$(basename $S3_HEAD).log 2>&1)
echo "units $(basename $S3_HEAD) rc $? $(date +%T)" >> $L
echo "component_costs_done $(date +%T)" >> $L
