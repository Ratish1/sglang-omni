"""Box only: every Qwen3-Omni vision encoder call's grids as served, for vision_interp_probe.py.

Every stage process imports sitecustomize at start while this directory is on PYTHONPATH. With ENCODER_GRIDS_DIR set,
once the image encoder module loads, Qwen3OmniImageEncoder.forward is wrapped: each call appends one line to
<dir>/grids_<pid>.jsonl with the image and video grid_thw rows it was given (each a list of [t, h, w], or null).
"""

import importlib.abc
import importlib.machinery
import json
import os
import sys

TARGET = "sglang_omni.models.qwen3_omni.components.image_encoder"


def install(module):
    encoder = module.Qwen3OmniImageEncoder
    original_forward = encoder.forward
    directory = os.environ["ENCODER_GRIDS_DIR"]
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"grids_{os.getpid()}.jsonl")

    def rows(grid):
        return None if grid is None else [[int(value) for value in row] for row in grid.tolist()]

    def forward(self, *, pixel_values=None, image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None, **kwargs):
        with open(path, "a") as handle:
            handle.write(json.dumps({"image": rows(image_grid_thw), "video": rows(video_grid_thw)}) + "\n")
        return original_forward(
            self,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            **kwargs,
        )

    encoder.forward = forward


class GridsAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        else:
            pass
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            install(module)

        spec.loader.exec_module = exec_and_install
        return spec


if os.environ.get("ENCODER_GRIDS_DIR"):
    sys.meta_path.insert(0, GridsAfterLoad())
else:
    pass
