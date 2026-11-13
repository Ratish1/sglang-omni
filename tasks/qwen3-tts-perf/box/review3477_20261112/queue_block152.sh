#!/usr/bin/env bash
# Block 152 of 11-13: #3477's review fixes, two H100 on host-85-234-79-221. F is the fix head (tail graph rule, waiting
# whole requests fail once CUDA decode is disabled, the arena log as an f-string, upstream main merged), P the PR head
# before them, PRE the tail graph rule commit with F's tests. The unit suites on F, the new case on PRE (it must fail),
# the device check on F, seeded c1 byte identity of P against F in both modes, then bf16 whole requests against fp32.
# usage: D=... T=<kit tree> P=<tree> F=<tree> PRE=<tree> bash queue_block152.sh
set -u
D=${D:?the results directory}
T=${T:?the kit tree}
K=$T/tasks/qwen3-tts-perf/box
L=$D/queue.log
NEW_CASE=tests/unit_test/qwen3_tts/test_pipeline.py::test_qwen3_tts_whole_requests_waiting_for_a_slot_fail_once_cuda_decode_is_disabled
mkdir -p $D
echo $$ > $D/queue.pgid
for arm in P F PRE; do
  tree=${!arm}
  echo "tree $arm $tree $(git -C $tree rev-parse HEAD HEAD:sglang_omni | tr '\n' ' ') dirty $(git -C $tree status --porcelain | wc -l)" >> $L
done
echo "queue start $(date +%T) kit $(git -C $T rev-parse HEAD); gpus $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')" >> $L
(cd $F && PYTHONPATH=$F timeout 1500 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts tests/unit_test/pipeline/test_stage.py tests/unit_test/utils/test_audio.py > $D/suite_F.log 2>&1)
echo "suite F rc $? $(date +%T): $(grep -E 'passed|failed' $D/suite_F.log | tail -1)" >> $L
(cd $PRE && PYTHONPATH=$PRE timeout 300 python3 -m pytest -q -p no:cacheprovider $NEW_CASE > $D/new_case_PRE.log 2>&1)
echo "new case on PRE rc $? $(date +%T): $(grep -E 'passed|failed' $D/new_case_PRE.log | tail -1)" >> $L
(cd $F && PYTHONPATH=$F QWEN3_TTS_BASE_1_7B_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base timeout 1500 python3 -m pytest -q -p no:cacheprovider tests/unit_test/qwen3_tts/test_pipeline.py -k test_real_base_whole_requests_on_reserved_slots > $D/device.log 2>&1)
echo "device rc $? $(date +%T): $(grep -E 'passed|failed' $D/device.log | tail -1)" >> $L
(cd $K/next_20261031 && python3 seeded_identity.py --main $P --branch $F --root $D/identity_PF --port 25000 > $D/identity_PF.txt 2>&1)
echo "identity P F rc $? $(date +%T): $(grep -E 'byte identical' $D/identity_PF.txt | cut -c1-120 | tr '\n' '#')" >> $L
(cd $F && PYTHONPATH=$F TORCHINDUCTOR_CACHE_DIR=$D/inductor_bf16 timeout 1800 python3 $K/review3477_20261112/bf16_whole_requests.py > $D/bf16.txt 2>&1)
echo "bf16 rc $? $(date +%T): $(grep -cE '^[a-z_]+ +ref' $D/bf16.txt) rows" >> $L
echo "queue done $(date +%T)" >> $L
