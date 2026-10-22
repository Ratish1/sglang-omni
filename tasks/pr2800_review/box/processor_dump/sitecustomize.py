"""Box only: every Qwen3-Omni processor call that carries audio saves what the thinker will receive, for
hf_agreement_queue.py.

Each stage process imports sitecustomize at start while this directory is on PYTHONPATH. Once the processor module
loads, Qwen3OmniMoeProcessor.__call__ is wrapped: the call runs unchanged, then its prompt text, the audio arrays it
was given, and the input_ids, input_features and feature_attention_mask it returned are saved under
PROCESSOR_DUMP_DIR, one file a call.
"""

import importlib.abc
import importlib.machinery
import itertools
import os
import sys
import threading

TARGET = "transformers.models.qwen3_omni_moe.processing_qwen3_omni_moe"


def install(module):
    import numpy as np
    import torch

    original = module.Qwen3OmniMoeProcessor.__call__
    directory = os.environ["PROCESSOR_DUMP_DIR"]
    os.makedirs(directory, exist_ok=True)
    counter = itertools.count()
    lock = threading.Lock()

    def call(self, *args, **kwargs):
        outputs = original(self, *args, **kwargs)
        audio = kwargs.get("audio")
        if audio:
            with lock:
                index = next(counter)
            torch.save(
                {
                    "text": kwargs.get("text"),
                    "audio": [np.asarray(item, dtype=np.float32) for item in audio],
                    "outputs": {
                        name: value.clone()
                        for name, value in outputs.items()
                        if isinstance(value, torch.Tensor)
                    },
                    "input_ids": outputs["input_ids"].clone(),
                    "input_features": outputs["input_features"].clone(),
                    "feature_attention_mask": outputs["feature_attention_mask"].clone(),
                },
                f"{directory}/{os.getpid()}_{index}.pt",
            )
        else:
            pass
        return outputs

    module.Qwen3OmniMoeProcessor.__call__ = call


class DumpAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def exec_and_install(module):
            exec_module(module)
            install(module)

        spec.loader.exec_module = exec_and_install
        return spec


if os.environ.get("PROCESSOR_DUMP_DIR"):
    sys.meta_path.insert(0, DumpAfterLoad())
else:
    pass
