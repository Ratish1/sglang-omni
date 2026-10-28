"""Hash of every thinker request's inputs, to compare two arms request by request.

Put this directory on PYTHONPATH. Once sglang_omni.models.qwen3_omni.request_builders is
imported, build_sglang_thinker_request is wrapped: before the build it prints one THINKERIN
line to stderr with the request id, the sha256 of the prompt's input ids (the key that pairs
a request across arms), and for every tensor in the state's thinker inputs its path, shape,
dtype and the sha256 of its bytes, so a value that differs between arms names its field.
build_talker_request_data is wrapped the same way: one TALKERIN line with the prompt ids' hash
and every tensor of the payload the talker receives from the pipeline (not its thinker stream).
In the Qwen3-Omni stages module, run_single_encoder_payload and batch_audio_encoder_payloads are
wrapped: one ENCODE line per encoded request with its batch size and batch members, the hash of
its encoder inputs (taken before the call) and of its encoder outputs.
"""

import hashlib
import importlib.abc
import importlib.util
import sys

TARGET = "sglang_omni.models.qwen3_omni.request_builders"


def digest(tensor):
    import torch

    host = tensor.detach().to("cpu").contiguous().reshape(-1)
    return hashlib.sha256(host.view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


def tensors(value, path, found):
    import torch

    if isinstance(value, torch.Tensor):
        found.append((path, value))
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            tensors(value[key], f"{path}.{key}", found)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            tensors(item, f"{path}[{index}]", found)


def patch(module):
    build_original = module.build_sglang_thinker_request

    def build_sglang_thinker_request(state, *args, **kwargs):
        input_ids = state.prompt["input_ids"]
        found = []
        tensors(state.thinker_inputs or {}, "thinker_inputs", found)
        fields = " ".join(
            f"{path}:{tuple(value.shape)}:{str(value.dtype).removeprefix('torch.')}:{digest(value)}"
            for path, value in found
        )
        print(
            f"THINKERIN rid={kwargs.get('request_id')} key={digest(input_ids)} "
            f"ids={input_ids.numel()} {fields}",
            file=sys.stderr,
            flush=True,
        )
        return build_original(state, *args, **kwargs)

    module.build_sglang_thinker_request = build_sglang_thinker_request

    talker_build_original = module.build_talker_request_data

    def build_talker_request_data(payload, *args, **kwargs):
        found = []
        tensors(payload.data, "data", found)
        input_ids = (payload.data.get("prompt") or {}).get("input_ids")
        key = digest(input_ids) if input_ids is not None else "none"
        fields = " ".join(
            f"{path}:{tuple(value.shape)}:{str(value.dtype).removeprefix('torch.')}:{digest(value)}"
            for path, value in found
        )
        print(
            f"TALKERIN rid={payload.request_id} key={key} {fields}",
            file=sys.stderr,
            flush=True,
        )
        return talker_build_original(payload, *args, **kwargs)

    module.build_talker_request_data = build_talker_request_data


def patch_stages(module):
    """One ENCODE line per encoded request: its batch, input and output hashes."""
    single_original = module.run_single_encoder_payload
    batch_original = module.batch_audio_encoder_payloads

    def input_digests(payloads, stage_name):
        digests = []
        for payload in payloads:
            request = module.build_encoder_request(
                module.load_state(payload), stage_name=stage_name
            )
            found = []
            tensors(request.model_inputs, "in", found)
            digests.append("|".join(digest(value) for _, value in found))
        return digests

    def report(payloads, results, inputs, stage_name, mode):
        batch_rids = ",".join(payload.request_id[:8] for payload in payloads)
        for payload, result, input_digest in zip(payloads, results, inputs):
            found = []
            tensors(
                module.load_state(result).encoder_outs.get(stage_name, {}),
                "out",
                found,
            )
            print(
                f"ENCODE stage={stage_name} mode={mode} rid={payload.request_id} "
                f"batch={len(payloads)} in={input_digest} "
                f"out={'|'.join(digest(value) for _, value in found)} "
                f"batch_rids={batch_rids}",
                file=sys.stderr,
                flush=True,
            )

    def run_single_encoder_payload(payload, *args, stage_name, **kwargs):
        inputs = input_digests([payload], stage_name)
        result = single_original(payload, *args, stage_name=stage_name, **kwargs)
        report([payload], [result], inputs, stage_name, "single")
        return result

    def batch_audio_encoder_payloads(payloads, *args, **kwargs):
        inputs = input_digests(payloads, module.AUDIO_STAGE)
        results = batch_original(payloads, *args, **kwargs)
        report(payloads, results, inputs, module.AUDIO_STAGE, "batch")
        return results

    module.run_single_encoder_payload = run_single_encoder_payload
    module.batch_audio_encoder_payloads = batch_audio_encoder_payloads


PATCHES = {TARGET: patch, "sglang_omni.models.qwen3_omni.stages": patch_stages}


class PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in PATCHES:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        exec_original = spec.loader.exec_module

        def exec_module(module):
            exec_original(module)
            PATCHES[name](module)

        spec.loader.exec_module = exec_module
        return spec


sys.meta_path.insert(0, PatchOnImport())
