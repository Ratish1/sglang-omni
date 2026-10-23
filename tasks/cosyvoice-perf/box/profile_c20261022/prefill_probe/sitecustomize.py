# SPDX-License-Identifier: Apache-2.0
"""L5 forensics (10-23): L5 alone (the AR prefill replayed from full CUDA graphs) read the seeded c1 WER higher than
L4's eager prefill in every draw. Put this directory first on PYTHONPATH with PREFILL_PROBE=<dir>. For every prefill
batch the AR samples from, each row's first token logits (float32, before any sampling transform), its full prompt
embeddings, its radix prefix length and its extend length go to <dir>/<tree>/<card>/<key>.pt, the tree from the
server's working directory, the card from CUDA_VISIBLE_DEVICES, the key a hash of the prompt ids; a prompt seen again
keeps its first record. No runtime code changes."""

import hashlib
import importlib.abc
import importlib.util
import os
import sys

TARGET = "sglang_omni.models.fun_cosyvoice3.model_runner"


def patch(module) -> None:
    import torch

    out = os.path.join(
        os.environ["PREFILL_PROBE"], os.path.basename(os.getcwd()), os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    )
    os.makedirs(out, exist_ok=True)
    sample = module.FunCosyVoice3ModelRunner.sample_next_token_ids

    def sample_recorded(self, logits_output, forward_batch, schedule_batch, requests):
        if forward_batch is not None and forward_batch.forward_mode.is_extend():
            logits = logits_output.next_token_logits.detach().float().cpu()
            for row, sched_req in enumerate(requests):
                req = sched_req.data.req
                key = hashlib.sha1(str(list(req.origin_input_ids)).encode()).hexdigest()[:16]
                path = os.path.join(out, key + ".pt")
                if not os.path.exists(path):
                    torch.save(
                        {
                            "logits": logits[row].clone(),
                            "prompt_input_embeds": sched_req.data.prompt_input_embeds.detach().cpu(),
                            "prefix_len": int(req.prefix_len),
                            "extend_len": int(req.extend_len),
                            "prompt_len": len(req.origin_input_ids),
                        },
                        path,
                    )
                else:
                    pass
        else:
            pass
        return sample(self, logits_output, forward_batch, schedule_batch, requests)

    module.FunCosyVoice3ModelRunner.sample_next_token_ids = sample_recorded
    print(f"prefill probe patched {module.__name__} into {out}", file=sys.stderr, flush=True)


class PatchOnImport(importlib.abc.MetaPathFinder):
    seen: set[str] = set()

    def find_spec(self, name, path, target=None):
        if name != TARGET or name in self.seen:
            return None
        self.seen.add(name)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("PREFILL_PROBE"):
    sys.meta_path.insert(0, PatchOnImport())
