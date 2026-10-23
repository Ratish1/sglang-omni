"""Box only: Qwen3-Omni's processor times each video at its sampled fps rounded to the nearest 2 / k (k whole).

Each stage process imports sitecustomize at start while this directory is on PYTHONPATH (socialomni_pair.py puts it
there for --hook-arms). On a tree that hands the processor each video's sampled fps, 2 / fps seconds a temporal grid
becomes whole, so the rope positions stay whole (13 k a grid) while frames keep about their real time against the
video's own audio, which use_audio_in_video interleaves at 13 positions a second.
"""

import importlib.abc
import importlib.machinery
import sys

PROCESSOR_MODULE = "transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe"
GRID_FRAMES = 2.0


def install(module):
    processor_call = module.Qwen3OmniMoeProcessor.__call__

    def call_with_whole_grid(self, *args, **kwargs):
        videos_kwargs = kwargs.get("videos_kwargs")
        if isinstance(videos_kwargs, dict) and videos_kwargs.get("fps"):
            frames_a_grid = max(1, round(GRID_FRAMES / float(videos_kwargs["fps"])))
            kwargs["videos_kwargs"] = {**videos_kwargs, "fps": GRID_FRAMES / frames_a_grid}
        else:
            pass
        return processor_call(self, *args, **kwargs)

    module.Qwen3OmniMoeProcessor.__call__ = call_with_whole_grid


class InstallAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != PROCESSOR_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            install(module)

        spec.loader.exec_module = exec_and_install
        return spec


sys.meta_path.insert(0, InstallAfterLoad())
