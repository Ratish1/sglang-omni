# SPDX-License-Identifier: Apache-2.0
"""Pool probe for profile_20261022/PLAN.md 4's memory pass (POOL_PROBE=1, this directory first on PYTHONPATH): the
Flow prefix pool's capacity, allocated and peak frames and its failed allocations (each a row that runs its whole
history that hop, streaming_vocoder.py hop_batch_with_prefix), and each whole utterance HiFT decode's rows, frames
and peak allocated bytes over the memory allocated before it (the device synchronized around the call, so this pass
is not timed). One JSON line a second to POOL_PROBE_OUT.<pid>.jsonl. Classes are wrapped on import, as the trace
marks probe does (cosy_profile_probe/cosy_impl.py PatchOnImport)."""

import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time
import weakref

ENABLED = os.environ.get("POOL_PROBE") == "1"
STATE = {
    "capacity_frames": 0,
    "allocated_frames": 0,
    "peak_frames": 0,
    "failed_allocations": 0,
    "hift_calls": 0,
    "hift_peak_bytes": 0,
    "hift_peak_rows": 0,
    "hift_peak_longest": 0,
    "hift_peak_frames": 0,
}
POOLS = []


def patch_prefix_cache(module) -> None:
    pool_class = module.PrefixKVPool
    original_init = pool_class.__init__
    original_allocate = pool_class.allocate
    original_release = pool_class.release
    block_frames = module.BLOCK_FRAMES

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # note (ratish): weak, so the startup sizing's throwaway pool is freed as without the probe.
        POOLS.append((weakref.ref(self), len(self.free_blocks)))

    def account(self) -> None:
        for pool, total in POOLS:
            if pool() is self:
                STATE["capacity_frames"] = total * block_frames
                STATE["allocated_frames"] = (total - len(self.free_blocks)) * block_frames
                STATE["peak_frames"] = max(STATE["peak_frames"], STATE["allocated_frames"])
            else:
                pass

    def allocate(self, block_count):
        taken = original_allocate(self, block_count)
        if taken is None:
            STATE["failed_allocations"] += 1
        else:
            account(self)
        return taken

    def release(self, blocks):
        original_release(self, blocks)
        account(self)

    pool_class.__init__ = init
    pool_class.allocate = allocate
    pool_class.release = release


def patch_stages(module) -> None:
    import torch

    vocoder_class = module.CosyVoice3Vocoder
    original = vocoder_class.mel2wav_batch

    def mel2wav_batch(self, mels):
        if not mels:
            return original(self, mels)
        else:
            pass
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        wavs = original(self, mels)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() - base
        STATE["hift_calls"] += 1
        if peak > STATE["hift_peak_bytes"]:
            lengths = [int(mel.shape[2]) for mel in mels]
            STATE.update(
                hift_peak_bytes=peak, hift_peak_rows=len(mels), hift_peak_longest=max(lengths),
                hift_peak_frames=sum(lengths),
            )
        else:
            pass
        return wavs

    vocoder_class.mel2wav_batch = mel2wav_batch


PATCHES = {
    "sglang_omni.models.fun_cosyvoice3.prefix_cache": patch_prefix_cache,
    "sglang_omni.models.fun_cosyvoice3.stages": patch_stages,
}


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES or name in getattr(self, "seen", ()):
            return None
        self.seen = (*getattr(self, "seen", ()), name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            try:
                PATCHES[name](module)
            except Exception as exc:
                print(f"pool_probe patch {name} failed: {exc!r}", file=sys.stderr)

        spec.loader.exec_module = exec_and_patch
        return spec


def write_forever(path: str) -> None:
    while True:
        time.sleep(1.0)
        if POOLS or STATE["hift_calls"]:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({"t": round(time.time(), 2), **STATE}) + "\n")
        else:
            pass


if ENABLED:
    sys.meta_path.insert(0, PatchOnImport())
    threading.Thread(
        target=write_forever, args=(f"{os.environ['POOL_PROBE_OUT']}.{os.getpid()}.jsonl",), daemon=True
    ).start()
    print("pool_probe on", file=sys.stderr, flush=True)
