import argparse
import inspect
import json
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


def inspect_penalty_compilation() -> dict[str, int | float]:
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )
    from torch._dynamo.utils import counters

    from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker

    torch._dynamo.reset()
    counters.clear()
    embedding = nn.Embedding(3072, 2, device="cuda", dtype=torch.bfloat16)
    model = SimpleNamespace(
        model=SimpleNamespace(codec_embedding=embedding),
        config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=3072)),
    )
    Qwen3OmniTalker.precompile_kernels_after_loading(model)
    torch.cuda.synchronize()
    after_startup = counters["stats"]["unique_graphs"]
    samples = {}
    for rows in (1, 2):
        logits = torch.ones(rows, 3072, device="cuda", dtype=torch.float32)
        penalties = torch.full_like(logits, 1.05)
        started = time.perf_counter()
        apply_scaling_penalties(logits, penalties)
        torch.cuda.synchronize()
        samples[f"fp32_rows_{rows}_seconds"] = time.perf_counter() - started
        samples[f"graphs_after_fp32_rows_{rows}"] = counters["stats"]["unique_graphs"]
    fixed_source = textwrap.dedent(
        inspect.getsource(Qwen3OmniTalker.precompile_kernels_after_loading)
    ).replace("dtype=activations.dtype", "dtype=torch.float32")
    fixed_namespace = {
        "torch": torch,
        "apply_scaling_penalties": apply_scaling_penalties,
    }
    exec(compile(fixed_source, "<fp32_warmup_probe>", "exec"), fixed_namespace)
    torch._dynamo.reset()
    counters.clear()
    fixed_namespace["precompile_kernels_after_loading"](model)
    torch.cuda.synchronize()
    samples["fixed_graphs_after_startup"] = counters["stats"]["unique_graphs"]
    for rows in (1, 2):
        logits = torch.ones(rows, 3072, device="cuda", dtype=torch.float32)
        apply_scaling_penalties(logits, torch.full_like(logits, 1.05))
    torch.cuda.synchronize()
    samples["fixed_graphs_after_live_shapes"] = counters["stats"]["unique_graphs"]
    return {"graphs_after_startup": after_startup, **samples}


def inspect_request_boundaries() -> dict[str, object]:
    from sglang.srt.managers.utils import validate_input_length

    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest, SamplingParams
    from sglang_omni.models.qwen3_omni.components.preprocessor import (
        validate_prompt_seq_len,
    )
    from sglang_omni.models.qwen3_omni.components.talker_input import (
        build_prefill_input,
    )
    from sglang_omni.models.qwen3_omni.request_builders import (
        build_sglang_talker_request,
        make_thinker_scheduler_adapters,
    )
    from sglang_omni.proto.request import OmniRequest, StagePayload
    from sglang_omni.serve.openai_api import build_chat_generate_request
    from sglang_omni.serve.protocol import ChatCompletionRequest
    from sglang_omni.serve.realtime.events import SessionObject
    from sglang_omni.serve.realtime.session import RealtimeSession

    builder, _ = make_thinker_scheduler_adapters(tokenizer=None, vocab_size=16)
    prompt = {
        "input_ids": torch.tensor([1, 2, 3]),
        "attention_mask": torch.ones(3, dtype=torch.long),
        "prompt_text": "probe",
    }
    requests = {
        "http_omitted": Client.build_omni_request(
            build_chat_generate_request(
                ChatCompletionRequest(
                    model="probe", messages=[{"role": "user", "content": "hi"}]
                )
            )
        ),
        "http_explicit_0_7": Client.build_omni_request(
            build_chat_generate_request(
                ChatCompletionRequest(
                    model="probe",
                    messages=[{"role": "user", "content": "hi"}],
                    temperature=0.7,
                )
            )
        ),
        "python_explicit_0_7": Client.build_omni_request(
            GenerateRequest(prompt="hi", sampling=SamplingParams(temperature=0.7))
        ),
        "direct_explicit_0_7": OmniRequest(inputs="hi", params={"temperature": 0.7}),
    }
    session = object.__new__(RealtimeSession)
    session.session_object = SessionObject(id="probe", model="probe", temperature=0.7)
    session.model_name = "probe"
    session.conversation = []
    requests["realtime_explicit_0_7"] = Client.build_omni_request(
        session.build_response_request("probe-audio")
    )
    observed = {}
    for name, request in requests.items():
        data = builder(
            StagePayload(request_id=name, request=request, data={"prompt": prompt})
        )
        observed[name] = {
            "requested_temperature": request.params.get("temperature"),
            "explicit_fields": request.metadata.get("explicit_generation_params"),
            "adapter_temperature": data.temperature,
            "normalized_temperature": data.req.sampling_params.temperature,
            "normalized_top_k": data.req.sampling_params.top_k,
        }

    ids = torch.tensor([1, 2, 3] + [4] * 33000 + [1, 5, 3, 6])
    validate_prompt_seq_len(ids, max_seq_len=65536, max_new_tokens=2048)
    embeddings = torch.zeros(ids.numel(), 2)
    prefill = build_prefill_input(
        thinker_embed=embeddings,
        thinker_hidden=embeddings,
        thinker_input_ids=ids,
        multimodal_mask=torch.zeros(ids.numel(), dtype=torch.bool),
        text_projection=nn.Identity(),
        hidden_projection=nn.Identity(),
        codec_embed_fn=nn.Embedding(16, 2),
        tts_bos_embed=torch.zeros(1, 2),
        tts_eos_embed=torch.zeros(1, 2),
        tts_pad_embed=torch.zeros(1, 2),
        im_start_token_id=1,
        system_token_id=7,
        user_token_id=2,
        assistant_token_id=5,
        speaker_id=8,
        codec_nothink_id=9,
        codec_think_bos_id=10,
        codec_think_eos_id=11,
        codec_pad_id=12,
        codec_bos_id=13,
        tts_pad_token_id=14,
    )
    talker = build_sglang_talker_request(
        torch.empty(0),
        tokenizer=None,
        codec_vocab_size=3072,
        talker_input_embeds=prefill["input_embeds"],
        talker_input_ids=prefill["input_ids"],
        input_embeds_are_projected=True,
    )
    error = validate_input_length(talker.req, 32766, allow_auto_truncate=False)
    return {
        "temperatures": observed,
        "long_speech": {
            "thinker_prompt_tokens": ids.numel(),
            "thinker_limit": 65536,
            "talker_prefill_tokens": prefill["input_ids"].numel(),
            "talker_limit": 32768,
            "talker_admission_error": error,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sampling", "defaults"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "sampling":
        result = inspect_penalty_compilation()
    else:
        result = inspect_request_boundaries()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
