"""Box only: zeroes the prompt rows the talker's first thinker chunk carries, for talker_rows_value_probe.py.

Every stage process imports sitecustomize at start while this directory is on PYTHONPATH. With TALKER_ZERO_ROWS
set, TalkerPrefillBuilder.build_prompt_prefill is wrapped once its module loads and replaces the first chunk's rows
with zeros before the original builds the prefill. The talker then reads main's inputs at the multimodal rows while
the thinker's capture, store and hand-off run unchanged.
"""

import importlib.abc
import importlib.machinery
import os
import sys

TARGET = "sglang_omni.models.qwen3_omni.components.talker_prefill"


def zero_rows(module):
    import torch

    original = module.TalkerPrefillBuilder.build_prompt_prefill

    def build_prompt_prefill(self, payload, thinker_chunks, *, thinker_done):
        first = thinker_chunks[0]
        if (first.metadata or {}).get("prompt_hidden_positions") is not None:
            first.data = torch.zeros_like(first.data)
        else:
            pass
        return original(self, payload, thinker_chunks, thinker_done=thinker_done)

    module.TalkerPrefillBuilder.build_prompt_prefill = build_prompt_prefill


class ZeroRowsAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def exec_and_zero(module):
            exec_module(module)
            zero_rows(module)

        spec.loader.exec_module = exec_and_zero
        return spec


if os.environ.get("TALKER_ZERO_ROWS"):
    sys.meta_path.insert(0, ZeroRowsAfterLoad())
