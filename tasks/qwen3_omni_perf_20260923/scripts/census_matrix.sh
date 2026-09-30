#!/usr/bin/env bash
# The census matrix on one card: cells one after another, each a run_probe_boot.sh boot of the
# same tree on the H200 profile. A cell is config:arm:concurrency:mode, where config is default
# (talker prefill graph only) or both (the thinker prefill graph too), and mode is probe (nsys
# with the probe, the census), clean (no profiler, no probe: the same cell's baseline), or lines
# (probe plus op labels at graph capture, for attribution). Each cell lands in
# OUT/<config>_<arm>_c<concurrency>_<mode>. PROFILE (default h200) picks the serve profile of
# run_probe_boot.sh (bf16 is the H100 profile, for tool checks off the H200).
# usage: census_matrix.sh <tree> <out root> <card> <port> "<cell> <cell> ..."
set -u
TREE=$1 ROOT=$2 CARD=$3 PORT=$4 CELLS=$5
S=$(cd "$(dirname "$0")" && pwd)
BOTH="--thinker.engine.cuda_graph_backend_prefill breakable --thinker.engine.cuda_graph_max_bs_prefill 2048"
mkdir -p "$ROOT"
for cell in $CELLS; do
  IFS=: read -r config arm conc mode <<< "$cell"
  out="$ROOT/${config}_${arm}_c${conc}_${mode}"
  case $config in
    default) extra="" ;;
    both) extra="$BOTH" ;;
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
    bash "$S/run_probe_boot.sh" "$TREE" "$out" "$CARD" "$PORT" "${PROFILE:-h200}" "$conc" "$arm"
  echo "$(date +%T) end $cell rc $?" >> "$ROOT/matrix.log"
done
echo "$(date +%T) matrix done" >> "$ROOT/matrix.log"
touch "$ROOT/MATRIX_DONE"
