# SPDX-License-Identifier: Apache-2.0
"""#3132 SIM forensics (10-25 evening): the first layer at which two servers' outputs for one request differ. Put
this directory first on PYTHONPATH with CONDITIONING_PROBE=<dir>; each server process appends JSON lines to
<dir>/<tree>_<card>.jsonl, the tree from the working directory, the card from CUDA_VISIBLE_DEVICES:
  prepared  request id, content key (sha1 of the target text and the reference's content key), and sha1 of the AR's
            prompt speech tokens, the Flow's prompt speech tokens, its prompt mel feat, the speaker embedding and the
            AR's prompt ids;
  prefill   request id, radix prefix and extend lengths, sha1 of the first token's float32 logits;
  result    request id, speech code count and sha1 of the codes.
No runtime code changes."""

import hashlib
import importlib.abc
import importlib.util
import json
import os
import sys
import threading

REQUEST_BUILDERS = "sglang_omni.models.fun_cosyvoice3.request_builders"
MODEL_RUNNER = "sglang_omni.models.fun_cosyvoice3.model_runner"
LOCK = threading.Lock()


def out_path():
    directory = os.environ["CONDITIONING_PROBE"]
    os.makedirs(directory, exist_ok=True)
    card = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    return os.path.join(directory, f"{os.path.basename(os.getcwd())}_{card}.jsonl")


def write(record):
    with LOCK, open(out_path(), "a") as probe:
        probe.write(json.dumps(record) + "\n")


def digest(value):
    import torch

    if isinstance(value, torch.Tensor):
        data = value.detach().float().cpu().contiguous().numpy().tobytes()
    else:
        data = repr(value).encode()
    return hashlib.sha1(data).hexdigest()[:16]


def patch_request_builders(module):
    preprocess = module.preprocess_cosyvoice3_payload
    apply_result = module.apply_sglang_cosyvoice3_result

    def preprocess_recorded(payload):
        prepared_payload = preprocess(payload)
        prepared = module._PREPARED_REQUESTS.get(payload.request_id)
        if prepared is not None:
            state = prepared.state
            reference_key = module.cosyvoice3_reference_input_key(state.ref_audio)
            write({
                "kind": "prepared",
                "rid": payload.request_id,
                "key": hashlib.sha1(f"{state.text}|{reference_key}".encode()).hexdigest()[:16],
                "reference": str(reference_key)[:16],
                "llm_prompt_speech_token": digest(prepared.llm_prompt_speech_token),
                "flow_prompt_speech_token": digest(prepared.flow_prompt_speech_token),
                "flow_prompt_speech_feat": digest(prepared.flow_prompt_speech_feat),
                "flow_embedding": digest(prepared.flow_embedding),
                "input_ids": digest(prepared.input_ids_list),
            })
        else:
            pass
        return prepared_payload

    def apply_result_recorded(payload, data):
        import torch

        codes = torch.stack(data.output_codes) if data.output_codes else torch.zeros(0)
        write({"kind": "result", "rid": payload.request_id, "codes": int(codes.numel()), "codes_sha": digest(codes)})
        return apply_result(payload, data)

    module.preprocess_cosyvoice3_payload = preprocess_recorded
    module.apply_sglang_cosyvoice3_result = apply_result_recorded


def patch_model_runner(module):
    sample = module.FunCosyVoice3ModelRunner.sample_next_token_ids

    def sample_recorded(self, logits_output, forward_batch, schedule_batch, requests):
        if forward_batch is not None and forward_batch.forward_mode.is_extend():
            logits = logits_output.next_token_logits.detach().float().cpu()
            for row, scheduler_request in enumerate(requests):
                request = scheduler_request.data.req
                write({"kind": "prefill", "rid": request.rid, "prefix": int(request.prefix_len),
                       "extend": int(request.extend_len), "logits": digest(logits[row])})
        else:
            pass
        return sample(self, logits_output, forward_batch, schedule_batch, requests)

    module.FunCosyVoice3ModelRunner.sample_next_token_ids = sample_recorded


PATCHES = {REQUEST_BUILDERS: patch_request_builders, MODEL_RUNNER: patch_model_runner}


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
            print(f"conditioning probe patched {name} into {out_path()}", file=sys.stderr, flush=True)

        spec.loader.exec_module = exec_and_patch
        return spec


if os.environ.get("CONDITIONING_PROBE"):
    sys.meta_path.insert(0, PatchOnImport())
