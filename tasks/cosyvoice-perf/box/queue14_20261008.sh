#!/usr/bin/env bash
# The PR evidence of items 1 and 2 on their final code (patches/pr_review_20261008), one run after
# the other on cards 0 and 1: a warm-up boot of both trees that also fills each tree's own Triton
# cache (the fused conv kernel's entries in convk2's cache show the served process ran it, and
# their times show when it compiled), then main against item 1 and item 1 against items 1 and 2,
# each pair then the swapped order, c1 and c16 (buffered c16 for item 2).
# usage (in the container): bash queue14_20261008.sh
cd /data/p20261008
P=/data/p20261008
R=$P/pr_review
TRITON_FINALS="TRITON_CACHE_DIR=$R/triton_finals2"
TRITON_CONVK="TRITON_CACHE_DIR=$R/triton_convk2"

triton_check() {
  {
    echo "== $1 $(date +%T)"
    find "$R/triton_convk2" -name 'group_conv_mish_kernel*' -printf '%TT %p\n' | sort
    echo "binaries $(find "$R/triton_convk2" -name 'group_conv_mish_kernel.cubin' | wc -l)"
  } >> "$R/triton_check.txt"
}

echo "warm $(date +%T)" >> $R/queue.txt
CELLS="w16:16:64::1 wb16:16:64::0" SCORE_CELLS=none \
  EXTRA_ENV_0="$TRITON_FINALS" EXTRA_ENV_1="$TRITON_CONVK" \
  bash ab_pair.sh $R/warm finals2 convk2 26000
triton_check warm
echo "p1 $(date +%T)" >> $R/queue.txt
CELLS="c1:1:64:1234:1 c16:16:::1" SCORE_CELLS=c16 EXTRA_ENV_1="$TRITON_FINALS" \
  bash ab_pair.sh $R/p1 sglang-omni finals2 27000
echo "p2 $(date +%T)" >> $R/queue.txt
CELLS="c1:1:64:1234:1 c16:16:::1" SCORE_CELLS=c16 EXTRA_ENV_0="$TRITON_FINALS" \
  bash ab_pair.sh $R/p2 finals2 sglang-omni 28000
echo "p3 $(date +%T)" >> $R/queue.txt
EXTRA_ENV_0="$TRITON_FINALS" EXTRA_ENV_1="$TRITON_CONVK" \
  bash ab_pair.sh $R/p3 finals2 convk2 29000
echo "p4 $(date +%T)" >> $R/queue.txt
EXTRA_ENV_0="$TRITON_CONVK" EXTRA_ENV_1="$TRITON_FINALS" \
  bash ab_pair.sh $R/p4 convk2 finals2 30000
triton_check p4
echo "queue done $(date +%T)" >> $R/queue.txt
