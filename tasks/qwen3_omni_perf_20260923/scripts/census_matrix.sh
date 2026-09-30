#!/usr/bin/env bash
# The census matrix on one card: cells one after another, each a run_probe_boot.sh boot of the
# same tree. A cell is config:arm:concurrency:mode[:samples], where config is default (the
# profile as shipped: talker prefill graph only on the H200, both graphs on the H100), both
# (adds the thinker prefill graph) or talkeronly (removes it), and mode is probe (nsys
# with the probe, the census), clean (no profiler, no probe: the same cell's baseline), or lines
# (probe plus op labels at graph capture, for attribution); samples overrides MAX_SAMPLES for
# the cell. Each cell lands in
# OUT/<config>_<arm>_c<concurrency>_<mode>. PROFILE (default h200) picks the serve profile of
# run_probe_boot.sh (bf16 is the H100 profile, for tool checks off the H200).
# usage: census_matrix.sh <tree> <out root> <card> <port> "<cell> <cell> ..."
set -u
TREE=$1 ROOT=$2 CARD=$3 PORT=$4 CELLS=$5
S=$(cd "$(dirname "$0")" && pwd)
BOTH="--thinker.engine.cuda_graph_backend_prefill breakable --thinker.engine.cuda_graph_max_bs_prefill 2048"
TALKER_ONLY="--thinker.engine.cuda_graph_backend_prefill disabled"
mkdir -p "$ROOT"
for cell in $CELLS; do
  IFS=: read -r config arm conc mode samples <<< "$cell"
  out="$ROOT/${config}_${arm}_c${conc}_${mode}"
  case $config in
    default) extra="" ;;
    both) extra="$BOTH" ;;
    talkeronly) extra="$TALKER_ONLY" ;;
    *) echo "unknown config $config" >> "$ROOT/matrix.log"; continue ;;
  esac
  case $mode in
    probe) nsys_args="" lines="" ;;
    clean) nsys_args="none" lines="" ;;
    lines) nsys_args="" lines="capture" ;;
    *) echo "unknown mode $mode" >> "$ROOT/matrix.log"; continue ;;
  esac
  echo "$(date +%T) start $cell" >> "$ROOT/matrix.log"
  env EXTRA_SERVE_ARGS="$extra" LINES="$lines" ${nsys_args:+NSYS_ARGS=$nsys_args} \
    ${samples:+MAX_SAMPLES=$samples} \
    bash "$S/run_probe_boot.sh" "$TREE" "$out" "$CARD" "$PORT" "${PROFILE:-h200}" "$conc" "$arm"
  rc=$?
  echo "$(date +%T) end $cell rc $rc" >> "$ROOT/matrix.log"
  # a cell that could not stop its server leaves the card to it; the rest would run on top
  if [ -f "$out/FAILED" ] && grep -q "not stopped" "$out/FAILED"; then
    echo "$(date +%T) stop: $cell left its server running" >> "$ROOT/matrix.log"
    break
  fi
done
echo "$(date +%T) matrix done" >> "$ROOT/matrix.log"
touch "$ROOT/MATRIX_DONE"
