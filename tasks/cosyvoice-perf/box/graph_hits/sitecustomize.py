"""#2631 graph hit counter: one stderr line per whole history solve and per prefix solve with
its rows, its frames per CFG half, the tier that holds it and whether it replayed from a graph.
Put this directory on PYTHONPATH and set OMNI_GRAPH_HITS=1. No runtime code changes."""

import bisect
import importlib.abc
import importlib.util
import os
import sys
import time

WHOLE_HISTORY = "sglang_omni.models.fun_cosyvoice3.whole_history_cuda_graph"
PREFIX = "sglang_omni.models.fun_cosyvoice3.prefix_cuda_graph"


def tier_of(tier_frames, frames):
    index = bisect.bisect_left(tier_frames, frames)
    return tier_frames[index] if index < len(tier_frames) else 0


def patch_whole_history(module):
    cls = module.WholeHistoryCudaGraphRunner
    run = cls.run

    def counted_run(self, **kwargs):
        lengths = kwargs["lengths"]
        frames = sum(lengths)
        generated = run(self, **kwargs)
        print(
            f"graph.whole t={time.time():.3f} hit={int(generated is not None)} "
            f"rows={len(lengths)} frames={frames} "
            f"tier={tier_of(self.tier_frames, frames)} max_row={max(lengths)} "
            f"top={self.tier_frames[-1]} lengths={','.join(map(str, lengths))}",
            file=sys.stderr,
            flush=True,
        )
        return generated

    cls.run = counted_run


def patch_prefix(module):
    cls = module.PrefixCudaGraphRunner
    run = cls.run

    def counted_run(self, **kwargs):
        new_frames = kwargs["new_frames"]
        frames = sum(new_frames)
        committed = [pair[0].committed_frames for pair in kwargs["caches"]]
        first = [int(frames_held == 0) for frames_held in committed]
        generated = run(self, **kwargs)
        print(
            f"graph.prefix t={time.time():.3f} hit={int(generated is not None)} "
            f"rows={len(new_frames)} frames={frames} "
            f"tier={tier_of(self.tier_frames, frames)} max_row={max(new_frames)} "
            f"top={self.tier_frames[-1]} lengths={','.join(map(str, new_frames))} "
            f"first={','.join(map(str, first))} "
            f"committed={','.join(map(str, committed))}",
            file=sys.stderr,
            flush=True,
        )
        return generated

    cls.run = counted_run


def patch_stages(module):
    cls = module.CosyVoice3Vocoder
    hop_batch = cls.hop_batch

    def counted_hop_batch(self, items):
        print(
            f"graph.plain t={time.time():.3f} rows={len(items)}",
            file=sys.stderr,
            flush=True,
        )
        return hop_batch(self, items)

    cls.hop_batch = counted_hop_batch


STAGES = "sglang_omni.models.fun_cosyvoice3.stages"
PATCHES = {WHOLE_HISTORY: patch_whole_history, PREFIX: patch_prefix, STAGES: patch_stages}


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name not in PATCHES or name in self.seen:
            return None
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            PATCHES[name](module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("OMNI_GRAPH_HITS") == "1":
    sys.meta_path.insert(0, PatchOnImport())
