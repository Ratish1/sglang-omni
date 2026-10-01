"""#2406 serving counter: one stderr line per causal step with the rows that ran
on the prefix cache, the rows that fell back to the whole-history hop and the
pool's free frames. Put this directory on PYTHONPATH and set OMNI_PREFIX_COUNT=1.
No runtime code changes."""

import importlib.abc
import importlib.util
import os
import sys
import time

TARGET = "sglang_omni.models.fun_cosyvoice3.streaming_vocoder"


def patch(module):
    scheduler = module.FunCosyVoice3StreamingVocoderScheduler
    hop_batch_with_prefix = scheduler.hop_batch_with_prefix

    def counted(self, participants, items):
        pool = self.vocoder.flow.prefix_pool
        before = {id(state): state.flow_cache is not None for _, state in participants}
        mels = hop_batch_with_prefix(self, participants, items)
        cached = sum(state.flow_cache is not None for _, state in participants)
        dropped = sum(
            before[id(state)] and state.flow_cache is None for _, state in participants
        )
        free = -1 if pool is None else pool.free_frames
        print(
            f"prefix.step t={time.time():.3f} rows={len(participants)} cached={cached} "
            f"plain={len(participants) - cached} dropped={dropped} free_frames={free}",
            file=sys.stderr,
            flush=True,
        )
        return mels

    scheduler.hop_batch_with_prefix = counted


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen = False

    def find_spec(self, name, path, target=None):
        if name != TARGET or self.seen:
            return None
        self.seen = True
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("OMNI_PREFIX_COUNT") == "1":
    sys.meta_path.insert(0, PatchOnImport())
