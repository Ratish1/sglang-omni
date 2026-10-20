#!/usr/bin/env bash
# The redraw moved into collect_tokens (our #2626 branch 021f62916, stacked on #2625, fixed_pr2626_v3.patch) on
# main 1ef7b8913 as t_f2626c: the component test (ras_rows_component: seeded identity with the old every row
# redraw, cost per step old against new), the Fun-CosyVoice3, sampling and model_runner suites, then one same time
# A/B against #2625 alone (t_f2625) so the pair isolates #2626, both card orders, with a seeded and an unseeded
# c1 cell.
# usage (in the container): bash queue_ras2_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
MAIN=1ef7b8913cd2a4a3c803f1c75fafad42537356ce
git -C /workspace/sglang-omni worktree add -q --detach /workspace/t_f2626c "$MAIN"
git -C /workspace/t_f2626c apply -3 $R/patches/m1ef7b8913_20261012/fixed_pr2626_v3.patch
git -C /workspace/t_f2626c add -A
git -C /workspace/t_f2626c reset -q
echo "t_f2626c on $MAIN with fixed_pr2626_v3 changed=$(git -C /workspace/t_f2626c status --short | wc -l)" \
  | tee /workspace/t_f2626c/BRANCH_MARKER.txt >> $R/queue_ras2.txt
mkdir -p $R/ras2
(cd /workspace/t_f2626c && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/workspace/t_f2626c python3 -u $IN/ras_rows_component_c20261012.py) \
  > $R/ras2/component.log 2>&1
echo "component rc $? $(date +%T)" >> $R/queue_ras2.txt
(cd /workspace/t_f2626c && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/t_f2626c python3 -m pytest \
  tests/unit_test/fun_cosyvoice3 tests/unit_test/sampling tests/unit_test/model_runner -q -p no:cacheprovider -rfE \
  > $R/ras2/units.log 2>&1)
echo "units: $(grep -E '[0-9]+ (passed|failed)' $R/ras2/units.log | tail -1) $(date +%T)" >> $R/queue_ras2.txt
cp -r $R/inductor_t_f2626 $R/inductor_t_f2626c
export ROOT=$R
export CELLS="c1:1:64:1234:1 c1u:1:64::1 c16:16:::1 c64:64:::1 L16:16:::1:$R/longform/meta.lst"
export SCORE_CELLS="c16"
PAIR="CARD_0=0 CARD_1=1 CPUS_0=0-15,64-79 CPUS_1=16-31,80-95"
echo "r2 a $(date +%T) card0=t_f2625 card1=t_f2626c" >> $R/queue_ras2.txt
env $PAIR bash $IN/ab_pair_sync.sh $R/ras2/a t_f2625 t_f2626c 38000
echo "r2 b $(date +%T) card0=t_f2626c card1=t_f2625" >> $R/queue_ras2.txt
env $PAIR bash $IN/ab_pair_sync.sh $R/ras2/b t_f2626c t_f2625 38000
echo "queue done $(date +%T)" >> $R/queue_ras2.txt
