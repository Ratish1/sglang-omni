"""Box only: records the prompt rows the talker receives with its first thinker chunk, for prompt_rows_probe.py.

Every stage process imports sitecustomize at start while this directory is on PYTHONPATH. With PROMPT_ROWS_DUMP set,
TalkerPrefillBuilder.build_prompt_prefill is wrapped once its module loads, and each talker prompt prefill writes one
file there: the request id, the prompt ids, and the positions and rows the first chunk carries (None when it carries
none).
"""

import importlib.abc
import importlib.machinery
import os
import sys
import time

TARGET = "sglang_omni.models.qwen3_omni.components.talker_prefill"


def record(module):
    import torch

    original = module.TalkerPrefillBuilder.build_prompt_prefill

    def build_prompt_prefill(self, payload, thinker_chunks, *, thinker_done):
        first = thinker_chunks[0]
        positions = (first.metadata or {}).get("prompt_hidden_positions")
        prompt_ids = module.Qwen3OmniPipelineState.from_dict(payload.data).prompt["input_ids"]
        torch.save(
            dict(
                request_id=payload.request_id,
                prompt_ids=prompt_ids.reshape(-1).cpu().clone(),
                positions=positions,
                rows=None if positions is None else first.data.cpu().clone(),
            ),
            os.path.join(os.environ["PROMPT_ROWS_DUMP"], f"{time.time_ns()}.pt"),
        )
        return original(self, payload, thinker_chunks, thinker_done=thinker_done)

    module.TalkerPrefillBuilder.build_prompt_prefill = build_prompt_prefill


class RecordAfterLoad(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        exec_module = spec.loader.exec_module

        def exec_and_record(module):
            exec_module(module)
            record(module)

        spec.loader.exec_module = exec_and_record
        return spec


if os.environ.get("PROMPT_ROWS_DUMP"):
    sys.meta_path.insert(0, RecordAfterLoad())
