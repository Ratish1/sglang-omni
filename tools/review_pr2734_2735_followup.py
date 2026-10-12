import argparse
import hashlib
import inspect
import json
from collections import deque
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import torch
from torch import nn


@torch.no_grad()
def penalty_graphs():
    from sglang.srt.sampling.penaltylib.repetition_penalty import (
        apply_scaling_penalties,
    )
    from torch._dynamo.utils import counters

    from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker

    torch._dynamo.reset()
    counters.clear()
    model = SimpleNamespace(
        model=SimpleNamespace(
            codec_embedding=nn.Embedding(3072, 2, device="cuda", dtype=torch.bfloat16)
        ),
        config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=3072)),
    )
    Qwen3OmniTalker.precompile_kernels_after_loading(model)
    torch.cuda.synchronize()
    counts = {"startup": counters["stats"]["unique_graphs"]}
    for rows in (1, 3, 16, 32):
        logits = torch.ones(rows, 3072, device="cuda", dtype=torch.float32)
        apply_scaling_penalties(logits, torch.full_like(logits, 1.05))
        torch.cuda.synchronize()
        counts[str(rows)] = counters["stats"]["unique_graphs"]
    assert counts["startup"] > 0
    assert len(set(counts.values())) == 1
    return counts


def admission_cases():
    from sglang.srt.runtime_context import get_context

    from sglang_omni.models.qwen3_omni.request_builders import (
        build_sglang_talker_request,
    )
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from tests.unit_test.pipeline.test_scheduler import (
        init_sync_request_build_state,
        new_stage_payload,
    )

    results = []
    cases = [
        (65536, 131072, 62456, 1, True),
        (65536, 8192, 6000, 1, True),
        (4096, 8192, 100, 1, True),
        (4096, 8192, 4093, 1, True),
        (4096, 8192, 4094, 1, False),
        (65536, 4096, 4000, 16, True),
    ]
    for context, pool, prompt, page_size, accepted in cases:
        data = build_sglang_talker_request(
            torch.zeros(prompt, 2),
            tokenizer=None,
            codec_vocab_size=3072,
            max_new_tokens=4096,
        )
        scheduler = object.__new__(OmniScheduler)
        scheduler.outbox = Queue()
        scheduler.waiting_queue = []
        scheduler.pending_stream_ingress = {}
        scheduler.deferred_request_payloads = {}
        scheduler.dirty_deferred_request_ids = set()
        scheduler.aborted_request_ids = set()
        scheduler.aborted_request_id_order = deque()
        scheduler.max_req_len = min(context - 1, pool - 1)
        scheduler.max_req_input_len = scheduler.max_req_len - 1
        scheduler.max_new_tokens_limit = None
        scheduler.page_size = page_size
        scheduler.max_total_num_tokens = pool
        init_sync_request_build_state(scheduler)
        scheduler.request_builder = lambda payload: data
        with get_context().override_server_args():
            if accepted:
                scheduler.process_input_requests([new_stage_payload(data.req.rid)])
                assert scheduler.waiting_queue == [data.req]
                rejection = None
            else:
                rejection = scheduler.prepare_request_limits(data)
                assert rejection is not None
                assert len(data.req.origin_input_ids) == prompt
        item = dict(
            context=context,
            pool=pool,
            prompt=prompt,
            page_size=page_size,
            accepted=accepted,
        )
        if accepted:
            cap = data.req.sampling_params.max_new_tokens
            assert cap == data.max_new_tokens
            assert prompt + cap < scheduler.max_req_len
            for index in range(cap):
                data.req.output_ids.append(1)
                data.req.update_finish_state()
                assert data.req.finished() is (index == cap - 1)
            assert data.req.finished_reason.to_json()["type"] == "length"
            item.update(cap=cap, finished_length=data.req.finished_len)
        else:
            item["error"] = rejection
        results.append(item)
    return results


def producer_temperatures():
    from sglang_omni.client.client import Client
    from sglang_omni.client.types import GenerateRequest, SamplingParams
    from sglang_omni.models.qwen3_omni.request_builders import (
        make_thinker_scheduler_adapters,
    )
    from sglang_omni.proto.request import OmniRequest, StagePayload
    from sglang_omni.serve.openai_api import build_chat_generate_request
    from sglang_omni.serve.protocol import ChatCompletionRequest
    from sglang_omni.serve.realtime.events import SessionObject
    from sglang_omni.serve.realtime.session import RealtimeSession

    builder, _ = make_thinker_scheduler_adapters(tokenizer=None, vocab_size=16)
    http = ChatCompletionRequest(
        model="probe", messages=[{"role": "user", "content": "hi"}]
    )
    explicit = ChatCompletionRequest(
        model="probe", messages=[{"role": "user", "content": "hi"}], temperature=0.7
    )
    requests = {
        "http_omitted": Client.build_omni_request(build_chat_generate_request(http)),
        "http_explicit": Client.build_omni_request(
            build_chat_generate_request(explicit)
        ),
        "python": Client.build_omni_request(
            GenerateRequest(prompt="hi", sampling=SamplingParams(temperature=0.7))
        ),
        "direct": OmniRequest(inputs="hi", params={"temperature": 0.7}),
    }
    session = object.__new__(RealtimeSession)
    session.session_object = SessionObject(id="probe", model="probe", temperature=0.7)
    session.model_name = "probe"
    session.conversation = []
    requests["realtime"] = Client.build_omni_request(
        session.build_response_request("probe-audio")
    )
    output = {}
    for name, request in requests.items():
        data = builder(
            StagePayload(
                request_id=name,
                request=request,
                data={
                    "prompt": {
                        "input_ids": torch.tensor([1, 2, 3]),
                        "attention_mask": torch.ones(3, dtype=torch.long),
                        "prompt_text": "probe",
                    }
                },
            )
        )
        expected = 1.0 if name == "http_omitted" else 0.7
        assert data.req.sampling_params.temperature == expected
        output[name] = data.req.sampling_params.temperature
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("sampling", "defaults"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "sampling":
        result = {"penalty_graphs": penalty_graphs()}
    else:
        result = {
            "temperatures": producer_temperatures(),
            "admission": admission_cases(),
        }
    from sglang.srt.layers import sampler

    result["sampler_sha256"] = hashlib.sha256(
        Path(inspect.getfile(sampler)).read_bytes()
    ).hexdigest()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
