#!/usr/bin/env bash
# Which half of the tokenizer fix costs the c16 inter chunk mean, and which bounds the arena: the fix
# without the unified stream (tokA) against the fix without the lock (tokB), both card orders, c16 and
# buffered c16, card memory from dmon.
cd /data/p20261008
P=/data/p20261008
for v in tokA tokB; do
  [ -d /workspace/$v ] || git -C /workspace/sglang-omni worktree add -q --detach /workspace/$v a093c7a25
  (cd /workspace/$v && git checkout -q -- . && git am -q --abort 2>/dev/null; git log --oneline -1 | grep -q "triton kernel" || git am -q $P/patches/stack_a093c7a25/000*.patch; git apply $P/patches/speech_tokenizer_prewarm_port_20261008.patch)
  ln -sfn inductor_convk $P/inductor_$v
done
python3 - <<'PY'
from pathlib import Path
a = Path("/workspace/tokA/sglang_omni/models/fun_cosyvoice3/utils.py")
s = a.read_text()
assert '"use_ep_level_unified_stream": "1",\n' in s
a.write_text(s.replace('                        "use_ep_level_unified_stream": "1",\n', ''))
b = Path("/workspace/tokB/sglang_omni/models/fun_cosyvoice3/utils.py")
s = b.read_text()
old = "        with self.run_lock:\n            result = self.session.run("
assert old in s
s = s.replace(old, "        if True:\n            result = self.session.run(")
b.write_text(s)
PY
for v in tokA tokB; do echo "$v $(git -C /workspace/$v diff --stat | tail -1)"; done >> $P/queue.txt
echo "abtokv1 $(date +%T)" >> $P/queue.txt
CELLS="c16:16:::1 b16:16:::0" SCORE_CELLS=none bash ab_pair.sh $P/abtokv1 tokA tokB 25000
echo "abtokv2 $(date +%T)" >> $P/queue.txt
CELLS="c16:16:::1 b16:16:::0" SCORE_CELLS=none bash ab_pair.sh $P/abtokv2 tokB tokA 26000
echo "queue12 done $(date +%T)" >> $P/queue.txt
