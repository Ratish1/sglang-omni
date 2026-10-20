#!/usr/bin/env bash
# Item B identity rerun with v2 of the identity script (a device synchronize before reading each part's
# outputs), main t_bm then B t_b on card 1, then compared bit for bit.
# usage (in the container): bash queue_b2_c20261012.sh
set -u
R=/data/c20261012
IN=$R/in
OUT=$R/b2
mkdir -p $OUT
for pair in "t_bm main" "t_b b"; do
  set -- $pair
  (cd /workspace/$1 && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/workspace/$1 \
    TORCHINDUCTOR_CACHE_DIR=$R/inductor_$1 python3 -u $IN/flow_uploads_identity_v2_c20261012.py $OUT/$2.pt) \
    > $OUT/identity_$2.log 2>&1
  echo "identity $2 rc $?" >> $OUT/queue.txt
done
python3 $IN/flow_uploads_identity_v2_c20261012.py --compare $OUT/main.pt $OUT/b.pt >> $OUT/queue.txt 2>&1
echo done >> $OUT/queue.txt
