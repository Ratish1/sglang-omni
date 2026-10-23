"""Box only: what the served thinker computes at each boundary of a prefill, for mmsu_layer_parity.py.

Every stage process imports sitecustomize at start while this directory is on PYTHONPATH. With LAYER_DUMP_DIR set,
once the shared thinker runner's module loads, inject_multimodal_embeds and post_prefill are wrapped: each prefill
chunk writes <rid>_inputs.pt (the prompt ids, the chunk's prefix, the audio embeddings the encoder stage handed the
thinker, the composed input embeddings of the chunk, and the request's M-RoPE positions and delta), and each prefill
writes <rid>_logits.pt (the next-token logits in fp32 and the token sampled). With PROCESSOR_DUMP_DIR set,
processor_dump/sitecustomize.py's processor hook is installed too.
"""

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from pathlib import Path

TARGET = "sglang_omni.model_runner.thinker_model_runner"


def install(module):
    import torch

    runner = module.ThinkerModelRunner
    original_inject = runner.inject_multimodal_embeds
    original_post_prefill = runner.post_prefill
    directory = os.environ["LAYER_DUMP_DIR"]
    os.makedirs(directory, exist_ok=True)

    def inject_multimodal_embeds(self, forward_batch, schedule_batch):
        result = original_inject(self, forward_batch, schedule_batch)
        if result is None or result[0] is None:
            return result
        else:
            pass
        input_embeds = result[0]
        extend_lens = [int(length) for length in forward_batch.extend_seq_lens_cpu]
        prefix_lens = [int(prefix) for prefix in forward_batch.extend_prefix_lens_cpu]
        start = 0
        for req, length, prefix in zip(schedule_batch.reqs, extend_lens, prefix_lens):
            model_inputs = req.omni_model_inputs or {}
            audio_embeds = model_inputs.get("audio_embeds")
            multimodal_inputs = req.multimodal_inputs
            torch.save(
                dict(
                    rid=req.rid,
                    prefix=prefix,
                    origin_input_ids=list(req.origin_input_ids),
                    audio_embeds=None if audio_embeds is None else audio_embeds.detach().cpu().clone(),
                    input_embeds=input_embeds[start : start + length].detach().cpu().clone(),
                    mrope_positions=(
                        None if multimodal_inputs is None else multimodal_inputs.mrope_positions.detach().cpu().clone()
                    ),
                    mrope_position_delta=(
                        None
                        if multimodal_inputs is None
                        else multimodal_inputs.mrope_position_delta.detach().cpu().clone()
                    ),
                ),
                f"{directory}/{req.rid}_inputs.pt",
            )
            start += length
        return result

    def post_prefill(self, result, forward_batch, schedule_batch, requests):
        logits = result.logits_output.next_token_logits
        next_token_ids = getattr(result, "next_token_ids", None)
        if logits is not None:
            for index, req in enumerate(schedule_batch.reqs):
                torch.save(
                    dict(
                        rid=req.rid,
                        logits=logits[index].detach().float().cpu().clone(),
                        next_token_id=None if next_token_ids is None else int(next_token_ids[index]),
                    ),
                    f"{directory}/{req.rid}_logits.pt",
                )
        else:
            pass
        return original_post_prefill(self, result, forward_batch, schedule_batch, requests)

    runner.inject_multimodal_embeds = inject_multimodal_embeds
    runner.post_prefill = post_prefill


class DumpAfterLoad(importlib.abc.MetaPathFinder):
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


if os.environ.get("LAYER_DUMP_DIR"):
    sys.meta_path.insert(0, DumpAfterLoad())
else:
    pass
if os.environ.get("PROCESSOR_DUMP_DIR"):
    processor_hook = Path(__file__).resolve().parent.parent / "processor_dump" / "sitecustomize.py"
    processor_spec = importlib.util.spec_from_file_location("processor_dump_hook", processor_hook)
    processor_spec.loader.exec_module(importlib.util.module_from_spec(processor_spec))
else:
    pass
