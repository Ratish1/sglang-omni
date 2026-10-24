#!/usr/bin/env bash
# After queue_d8_c64_events.sh: D8's c16 RTF read +0.4 and +0.2 % beyond the card offset main against main showed, and
# D8 moves the speaker mel from the CPU into the speaker graph. The speaker embedding's device time per SeedTTS
# reference, main, D8, D8, main, card 3 (speaker_device_time.py copied beside this script on the box).
set -u
D=/data/next1024
L=$D/queue.log
until grep -q queue_d8_c64_events_done $L; do sleep 20; done
echo "speaker device time start $(date +%T)" >> $L
for arm in "main /workspace/main_67476823a" "d8 /workspace/d8_ccdcf38c5" "d8b /workspace/d8_ccdcf38c5" "mainb /workspace/main_67476823a"; do
  set -- $arm
  (cd $2 && PYTHONPATH=$2 python3 $D/speaker_device_time.py --card 3 > $D/speaker_time_$1.txt 2>&1)
  echo "speaker device time $1: $(grep 'speaker embed' $D/speaker_time_$1.txt) $(date +%T)" >> $L
done
echo "speaker_device_time_done $(date +%T)" >> $L
