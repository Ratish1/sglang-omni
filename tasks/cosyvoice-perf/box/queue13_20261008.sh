#!/usr/bin/env bash
# The tokenizer fix with its own non-blocking torch stream passed to ONNX Runtime
# (user_compute_stream) in place of the EP level unified stream (tokC), against the full fix
# (tokarena), both card orders, c16 and buffered c16.
cd /data/p20261008
P=/data/p20261008
[ -d /workspace/tokC ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/tokC a093c7a25
(cd /workspace/tokC && git log --oneline -1 | grep -q "triton kernel" || git am -q $P/patches/stack_a093c7a25/000*.patch; git checkout -q -- . && git apply $P/patches/speech_tokenizer_prewarm_port_20261008.patch)
ln -sfn inductor_convk $P/inductor_tokC
python3 - <<'PY'
from pathlib import Path
p = Path("/workspace/tokC/sglang_omni/models/fun_cosyvoice3/utils.py")
s = p.read_text()
old = '''                    {
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "use_ep_level_unified_stream": "1",
                    },'''
new = '''                    {
                        "cudnn_conv_algo_search": "HEURISTIC",
                        "has_user_compute_stream": "1",
                        "user_compute_stream": str(self.stream.cuda_stream),
                    },'''
assert old in s
s = s.replace(old, new)
old2 = "        providers = (\n            [\n                (\n                    \"CUDAExecutionProvider\","
assert old2 in s
s = s.replace(old2, "        self.stream = torch.cuda.Stream(device=device) if device.startswith(\"cuda\") else None\n" + old2, 1)
p.write_text(s)
PY
echo "tokC $(git -C /workspace/tokC diff --stat | tail -1)" >> $P/queue.txt
echo "abtokc1 $(date +%T)" >> $P/queue.txt
CELLS="c16:16:::1 b16:16:::0" SCORE_CELLS=none bash ab_pair.sh $P/abtokc1 tokarena tokC 21000
echo "abtokc2 $(date +%T)" >> $P/queue.txt
CELLS="c16:16:::1 b16:16:::0" SCORE_CELLS=none bash ab_pair.sh $P/abtokc2 tokC tokarena 22000
echo "queue13 done $(date +%T)" >> $P/queue.txt
