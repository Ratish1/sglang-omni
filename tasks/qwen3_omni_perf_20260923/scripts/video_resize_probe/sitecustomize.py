"""Whether the HF video processor's resize changes frames the loader already resized.

Put this directory on PYTHONPATH. Once transformers.models.qwen2_vl.video_processing_qwen2_vl is
imported, Qwen2VLVideoProcessor.resize is wrapped: each call prints one VIDEORESIZE line to stderr
with the process, the input shape and dtype, the output shape, whether the output equals the input
exactly (same shape only), the largest absolute difference, and the call's wall ms.
"""

import importlib.abc
import importlib.util
import multiprocessing
import sys
import time

TARGET = "transformers.models.qwen2_vl.video_processing_qwen2_vl"


def patch(module):
    processor_cls = module.Qwen2VLVideoProcessor
    resize_original = processor_cls.resize

    def resize(self, videos, *args, **kwargs):
        start = time.perf_counter()
        result = resize_original(self, videos, *args, **kwargs)
        wall_ms = (time.perf_counter() - start) * 1e3
        same_shape = tuple(result.shape) == tuple(videos.shape)
        if same_shape:
            difference = (result.float() - videos.float()).abs().max().item()
            exact = bool(difference == 0)
        else:
            difference = None
            exact = False
        print(
            f"VIDEORESIZE {multiprocessing.current_process().name} in={tuple(videos.shape)} "
            f"{videos.dtype} out={tuple(result.shape)} exact={exact} "
            f"max_abs_diff={difference} wall_ms={wall_ms:.1f}",
            file=sys.stderr,
            flush=True,
        )
        return result

    processor_cls.resize = resize


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        exec_original = spec.loader.exec_module

        def exec_module(module):
            exec_original(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, PatchOnImport())
