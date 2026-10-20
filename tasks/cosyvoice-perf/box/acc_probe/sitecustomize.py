# SPDX-License-Identifier: Apache-2.0
"""Accuracy probe: what each Fun-CosyVoice3 component computed in serving, per request, for a replay through
CosyVoice's own modules (acc_reference_c20261011.py). In the server process, once the modules are imported:
- build_sglang_cosyvoice3_request: the AR's inputs (text token ids, the LM's prompt speech tokens, the
  prompt embeddings in the model dtype, the Flow prompt and speaker embedding, the sampling parameters),
- FunCosyVoice3ModelRunner.sample_next_token_ids: each step's raw logits row (before the sampler's in place
  temperature and softmax) and the sampled id,
- FunCosyVoice3StreamingVocoderScheduler.hift_step: per step and row the Flow's mel, the hop's offsets, the
  padded Flow prompt (first step), the tokens (final step) and HiFT's emitted samples.
AR records go to ACC_OUT/ar_<request>.pt when the AR finishes the request, vocoder records to
ACC_OUT/voc_<request>.pt at the final step. At most ACC_MAX_REQUESTS requests (default 24) are kept.
"""

import os
import sys
import threading
import time

OUT = os.environ.get("ACC_OUT")
MAX_REQUESTS = int(os.environ.get("ACC_MAX_REQUESTS", "24"))
PACKAGE = "sglang_omni.models.fun_cosyvoice3"


def install() -> None:
    import torch

    os.makedirs(OUT, exist_ok=True)
    lock = threading.Lock()
    ar: dict[str, dict] = {}
    voc: dict[str, dict] = {}
    patched: set[str] = set()

    def kept(request_id: str) -> bool:
        return request_id in ar

    def save_ar(request_id: str) -> None:
        with lock:
            record = ar.get(request_id)
            if record is None or "saved" in record:
                return
            record["saved"] = True
            snapshot = {key: value for key, value in record.items() if key != "saved"}
        torch.save(snapshot, os.path.join(OUT, f"ar_{request_id}.pt"))

    def patch_request_builders(module) -> None:
        original = module.build_sglang_cosyvoice3_request

        def build(payload, *, model):
            data = original(payload, model=model)
            request_id = str(payload.request_id)
            with lock:
                if len(ar) < MAX_REQUESTS:
                    params = data.req.sampling_params
                    ar[request_id] = {
                        "text_token_ids": list(data.req._cosyvoice3_text_token_ids),
                        "llm_prompt_speech_token_ids": list(data.req._cosyvoice3_prompt_speech_token_ids),
                        "prompt_input_embeds": data.prompt_input_embeds.detach().cpu().clone(),
                        "flow_prompt_speech_token": data.flow_prompt_speech_token.detach().cpu().clone(),
                        "flow_prompt_speech_feat": data.flow_prompt_speech_feat.detach().cpu().clone(),
                        "flow_embedding": data.flow_embedding.detach().cpu().clone(),
                        "sampling": {
                            "temperature": params.temperature,
                            "top_p": params.top_p,
                            "top_k": params.top_k,
                            "repetition_penalty": params.repetition_penalty,
                            "min_new_tokens": params.min_new_tokens,
                            "max_new_tokens": params.max_new_tokens,
                            "seed": params.sampling_seed,
                        },
                        "logits": [],
                        "sampled": [],
                    }
                    voc[request_id] = {"steps": []}
            return data

        module.build_sglang_cosyvoice3_request = build

    def patch_model_runner(module) -> None:
        runner = module.FunCosyVoice3ModelRunner
        sample = runner.sample_next_token_ids
        finished = runner.on_request_finished

        def sample_next_token_ids(self, logits_output, forward_batch, schedule_batch, requests):
            ids = [str(request.request_id) for request in requests]
            rows = [index for index, request_id in enumerate(ids) if kept(request_id)]
            raw = logits_output.next_token_logits[rows].detach().float().cpu() if rows else None
            next_ids = sample(self, logits_output, forward_batch, schedule_batch, requests)
            if rows:
                sampled = next_ids.reshape(-1).tolist()
                with lock:
                    for position, index in enumerate(rows):
                        record = ar[ids[index]]
                        record["logits"].append(raw[position].clone())
                        record["sampled"].append(int(sampled[index]))
            return next_ids

        def on_request_finished(self, request_id, req_data):
            finished(self, request_id, req_data)
            save_ar(str(request_id))

        runner.sample_next_token_ids = sample_next_token_ids
        runner.on_request_finished = on_request_finished

    def patch_vocoder(module) -> None:
        scheduler = module.FunCosyVoice3StreamingVocoderScheduler
        hift_step = scheduler.hift_step

        def hift_step_probe(self, participants, mels, *, is_final):
            before = []
            for (request_id, state), mel in zip(participants, mels, strict=True):
                before.append(
                    {
                        "request_id": str(request_id),
                        "is_final": bool(is_final),
                        "token_offset": int(state.token_offset),
                        "hop_len": int(state.hop_len),
                        "token_count": len(state.tokens),
                        "speech_offset_before": int(state.speech_offset),
                        "rows_in_step": len(participants),
                        "mel": mel.detach().float().cpu().clone(),
                    }
                )
            results = hift_step(self, participants, mels, is_final=is_final)
            for entry, ((request_id, state), (delta, offset)) in zip(
                before, zip(participants, results, strict=True), strict=True
            ):
                request_id = str(request_id)
                with lock:
                    record = voc.get(request_id)
                if record is None:
                    continue
                entry["delta"] = delta.detach().float().cpu().clone()
                entry["speech_offset_after"] = int(offset)
                if "prompt_token" not in record:
                    record["prompt_token"] = state.prompt_token.detach().cpu().clone()
                    record["prompt_feat"] = state.prompt_feat.detach().float().cpu().clone()
                    record["embedding"] = state.embedding.detach().float().cpu().clone()
                record["steps"].append(entry)
                if is_final:
                    record["tokens"] = list(state.tokens)
                    record["tf32 matmul, cudnn"] = (
                        torch.backends.cuda.matmul.allow_tf32,
                        torch.backends.cudnn.allow_tf32,
                    )
                    torch.save(record, os.path.join(OUT, f"voc_{request_id}.pt"))
                    save_ar(request_id)
            return results

        scheduler.hift_step = hift_step_probe

    patches = {
        f"{PACKAGE}.request_builders": patch_request_builders,
        f"{PACKAGE}.model_runner": patch_model_runner,
        f"{PACKAGE}.streaming_vocoder": patch_vocoder,
    }
    while len(patched) < len(patches):
        for name, patch in patches.items():
            module = sys.modules.get(name)
            if name not in patched and module is not None and hasattr(module, "__file__"):
                # note(ratish): wait for the module body to finish executing before reading its classes.
                spec = getattr(module, "__spec__", None)
                if spec is not None and getattr(spec, "_initializing", False):
                    continue
                patch(module)
                patched.add(name)
                with open(os.path.join(OUT, f"patched.{os.getpid()}"), "a") as handle:
                    handle.write(f"{name} {time.time():.2f}\n")
        time.sleep(0.05)


if OUT:
    threading.Thread(target=install, name="acc-probe", daemon=True).start()
