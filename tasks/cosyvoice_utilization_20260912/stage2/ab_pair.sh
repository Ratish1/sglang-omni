#!/usr/bin/env bash
# A/B points, one after another; at each point the two arms boot and run at the
# same time on two cards, so both see the same host. Run it detached on the box:
#
#   TAG=s9 A_REV=<main> B_REV=<slice> POINTS="streaming:16 buffered:16 streaming:1:64" \
#     setsid nohup bash .../stage2/ab_pair.sh > .tmp/logs/s9.log 2>&1 < /dev/null &
#
#   TAG      run label; arms are named <TAG>-a-<point> and <TAG>-b-<point>
#   A_REV    revision of arm A            B_REV   revision of arm B
#   A_CARD   card of arm A           4    B_CARD  card of arm B           5
#   POINTS   mode:concurrency[:samples], samples empty for the whole split
#   SEED     as in g1_stream_identity.sh                      1234
#   SERVE    serve arguments of both arms
set -euo pipefail

REPO=${REPO:-/workspace/sglang-omni}
TAG=${TAG:?TAG is the run label}
A_REV=${A_REV:?A_REV is arm A revision}
B_REV=${B_REV:?B_REV is arm B revision}
A_CARD=${A_CARD:-4}
B_CARD=${B_CARD:-5}
POINTS=${POINTS:-"streaming:16 buffered:16"}
export SEED=${SEED-1234}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-unset}
export SERVE=${SERVE:-"--tts_engine.engine.mem_fraction_static 0.28"}
G1="$REPO/.tmp/wt/analysis/tasks/cosyvoice_utilization_20260912/stage2/g1_stream_identity.sh"

cd "$REPO"
mkdir -p .tmp/logs
for point in $POINTS; do
  IFS=: read -r mode conc samples <<< "$point"
  name="$mode-c$conc"
  echo "point $name start $(date -u +%H:%M:%S)"
  # The arms share the repo's git state while they set their worktrees up, so
  # the second one starts once the first is past that.
  MODE=$mode CONC=$conc SAMPLES=${samples:-} ARM="$TAG-a-$name" REV=$A_REV CARD=$A_CARD \
    PORT=$((8500 + A_CARD)) bash "$G1" > ".tmp/logs/$TAG-a-$name.log" 2>&1 &
  a=$!
  until grep -q '^out ' ".tmp/logs/$TAG-a-$name.log" 2>/dev/null; do
    kill -0 $a 2>/dev/null || break
    sleep 1
  done
  MODE=$mode CONC=$conc SAMPLES=${samples:-} ARM="$TAG-b-$name" REV=$B_REV CARD=$B_CARD \
    PORT=$((8500 + B_CARD)) bash "$G1" > ".tmp/logs/$TAG-b-$name.log" 2>&1 &
  b=$!
  wait $a && echo "point $name arm a ok" || echo "point $name arm a FAILED"
  wait $b && echo "point $name arm b ok" || echo "point $name arm b FAILED"
done
echo "finished=1 $(date -u +%H:%M:%S)"
