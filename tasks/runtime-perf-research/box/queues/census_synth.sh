#!/usr/bin/env bash
C=/workspace/marks/tasks/qwen3_omni_perf_20260923/scripts
one() {
  d=$1; r=$(basename $d)
  case $r in speech*) s=ADHLI;; *) s=ADI;; esac
  python3 $C/omni_census.py $d/trace.sqlite --window $d/window.txt --sections $s --sample-rate 24000 > $d/census.txt 2>&1
}
export -f one; export C
ls -d /data/rp1114/sy_f/synth/*/ | sed 's:/$::' | taskset -c 0-55 xargs -P 4 -I{} bash -c 'one {}'
echo "census sy_f done $(date -u +%T)" >> /data/rp1114/queue.log
