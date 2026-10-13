"""Exercise CosyVoice3 prompt ordering and allocation lifetime on CUDA."""

from __future__ import annotations

import argparse
import faulthandler
import gc
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import triton
import triton.language as tl
from huggingface_hub import hf_hub_download, snapshot_download
from sglang.srt.mem_cache.radix_cache import RadixKey

from sglang_omni.models.fun_cosyvoice3 import request_builders
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CosyVoice3PreparedRequest,
    CosyVoice3ReferenceArtifact,
    CosyVoice3SGLangRequestData,
    build_embedding_cache_key_ids,
    build_sglang_cosyvoice3_request,
    prepare_cosyvoice3_request,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import (
    SOS_ID,
    TASK_ID,
    TOTAL_VOCAB_SIZE,
)
from sglang_omni.models.fun_cosyvoice3.utils import (
    CosyVoice3Tokenizer,
    build_llm_prompt_embeddings,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


@triton.jit
def hold_stream(release_pointer):
    while tl.load(release_pointer, volatile=True) == 0:
        pass


@dataclass(frozen=True, kw_only=True)
class EmbeddingConfig:
    hidden_size: int


class EmbeddingTables(torch.nn.Module):
    def __init__(
        self, text_weights: torch.Tensor, speech_weights: torch.Tensor
    ) -> None:
        super().__init__()
        self.text_embed_tokens = torch.nn.Embedding.from_pretrained(text_weights)
        self.speech_embedding = torch.nn.Embedding.from_pretrained(speech_weights)
        self.config = EmbeddingConfig(hidden_size=text_weights.shape[1])


class Tokenizer:
    def __call__(self, text: str) -> torch.Tensor:
        return torch.tensor([[2, 3, 4] if text == "warm" else [1]], dtype=torch.int32)


def prepare(
    model: EmbeddingTables, producer: torch.cuda.Stream, text: str = "target"
) -> CosyVoice3PreparedRequest:
    return prepare_cosyvoice3_request(
        model=model,
        tokenizer=Tokenizer(),
        state=FunCosyVoice3State(text=text),
        reference_artifact=None,
        stream=producer,
    )


def lower(
    prepared: CosyVoice3PreparedRequest, model: EmbeddingTables
) -> CosyVoice3SGLangRequestData:
    request_id = "stream-probe"
    request_builders._PREPARED_REQUESTS[request_id] = prepared
    return build_sglang_cosyvoice3_request(
        StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs="target"),
            data={"_cosyvoice3_prepared_request": request_id},
        ),
        model=model,
    )


def warm(model: EmbeddingTables, producer: torch.cuda.Stream) -> None:
    for _ in range(16):
        prepared = prepare(model, producer, "warm")
        request = lower(prepared, model)
        snapshot = request.prompt_input_embeds.clone()
        torch.cuda.synchronize()
        del snapshot, request, prepared
    gc.collect()
    torch.cuda.synchronize()


def warm_consumer(consumer: torch.cuda.Stream) -> None:
    with torch.cuda.stream(consumer):
        scratch = torch.empty(65536, device="cuda")
        snapshot = scratch + 0.0
    consumer.synchronize()
    del scratch, snapshot


def check_ordering(model: EmbeddingTables, producer: torch.cuda.Stream) -> None:
    control = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    warm_consumer(consumer)
    release = torch.ones(1, dtype=torch.int32, device="cuda")
    hold_stream[(1,)](release, num_warps=1)
    torch.cuda.synchronize()
    expected = torch.cat(
        [
            model.speech_embedding.weight[[SOS_ID]],
            model.text_embed_tokens.weight[[1, 1]],
            model.speech_embedding.weight[[TASK_ID]],
        ]
    )
    host_snapshot = torch.empty_like(expected, device="cpu", pin_memory=True)
    producer_complete = torch.cuda.Event()
    consumer_complete = torch.cuda.Event()
    release.zero_()
    torch.cuda.synchronize()
    with torch.cuda.stream(producer):
        hold_stream[(1,)](release, num_warps=1)
    try:
        prepared = prepare(model, producer)
        producer_complete = prepared.ready_event
        assert not producer_complete.query()
        with torch.cuda.stream(consumer):
            request = lower(prepared, model)
            snapshot = request.prompt_input_embeds + 0.0
            host_snapshot.copy_(snapshot, non_blocking=True)
            consumer_complete.record()
        for _ in range(20):
            if consumer_complete.query():
                break
            else:
                time.sleep(0.001)
        consumer_finished_early = consumer_complete.query()
    finally:
        with torch.cuda.stream(control):
            release.fill_(1)
        control.synchronize()
        producer.synchronize()
        consumer.synchronize()
    matches = torch.equal(host_snapshot, expected.cpu())
    print(
        json.dumps(
            {
                "check": "ordering",
                "consumer_finished_before_producer_release": consumer_finished_early,
                "rows_match": matches,
            }
        ),
        flush=True,
    )
    assert not consumer_finished_early and matches


def check_lifetime(model: EmbeddingTables, producer: torch.cuda.Stream) -> None:
    consumer = torch.cuda.Stream()
    control = torch.cuda.Stream()
    warm_consumer(consumer)
    release = torch.ones(1, dtype=torch.int32, device="cuda")
    hold_stream[(1,)](release, num_warps=1)
    torch.cuda.synchronize()
    prepared = prepare(model, producer)
    producer.synchronize()
    expected = prepared.prompt_input_embeds.cpu().clone()
    shape = prepared.prompt_input_embeds.shape
    source_pointer = prepared.prompt_input_embeds.data_ptr()
    host_snapshot = torch.empty_like(expected, pin_memory=True)
    with torch.cuda.stream(producer):
        reserve = [
            torch.empty(shape, dtype=expected.dtype, device="cuda") for _ in range(256)
        ]
        for allocation in reserve:
            allocation.fill_(12345)
    producer.synchronize()
    del reserve
    allocations: list[torch.Tensor] = []
    release.zero_()
    torch.cuda.synchronize()
    with torch.cuda.stream(consumer):
        hold_stream[(1,)](release, num_warps=1)
    try:
        print("lifetime: lower", flush=True)
        with torch.cuda.stream(consumer):
            request = lower(prepared, model)
            snapshot = request.prompt_input_embeds + 0.0
            host_snapshot.copy_(snapshot, non_blocking=True)
        del request, prepared
        print("lifetime: collect", flush=True)
        gc.collect()
        print("lifetime: allocate", flush=True)
        with torch.cuda.stream(producer):
            for _ in range(128):
                allocation = torch.empty(shape, dtype=expected.dtype, device="cuda")
                allocation.fill_(12345)
                allocations.append(allocation)
        producer.synchronize()
        print("lifetime: producer complete", flush=True)
        reused = any(tensor.data_ptr() == source_pointer for tensor in allocations)
    finally:
        with torch.cuda.stream(control):
            release.fill_(1)
        control.synchronize()
        consumer.synchronize()
    matches = torch.equal(host_snapshot, expected)
    print(
        json.dumps(
            {"check": "lifetime", "source_reused": reused, "rows_match": matches}
        ),
        flush=True,
    )
    assert matches


def check_checkpoint(model_id: str) -> None:
    weights_path = hf_hub_download(model_id, "llm.pt")
    weights = torch.load(weights_path, map_location="cpu", weights_only=True, mmap=True)
    model = EmbeddingTables(
        weights["llm.model.model.embed_tokens.weight"].to("cuda", torch.bfloat16),
        weights["speech_embedding.weight"].to("cuda", torch.bfloat16),
    )
    del weights
    checkpoint = snapshot_download(
        model_id,
        allow_patterns=["CosyVoice-BlankEN/*.json", "CosyVoice-BlankEN/*.txt"],
    )
    tokenizer = CosyVoice3Tokenizer(str(Path(checkpoint) / "CosyVoice-BlankEN"))
    producer = torch.cuda.Stream()
    reference = CosyVoice3ReferenceArtifact(
        llm_prompt_speech_token=torch.tensor([[0, 40, 6560]], dtype=torch.int32),
        flow_prompt_speech_token=torch.tensor([[0, 40, 6560]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 6, 80),
        flow_embedding=torch.zeros(1, 192),
    )
    prompt_keys: list[tuple[RadixKey, RadixKey]] = []
    for text in ("Hello world.", "The rain falls softly.", "请慢慢地说这句话。"):
        for prompt in (None, "A reference sentence.", "speak warmly"):
            state = FunCosyVoice3State(
                text=text,
                ref_audio="encoded-reference",
                ref_text=prompt if prompt == "A reference sentence." else None,
                instructions=prompt if prompt == "speak warmly" else None,
            )
            prepared = prepare_cosyvoice3_request(
                model=model,
                tokenizer=tokenizer,
                state=state,
                reference_artifact=reference,
                stream=producer,
            )
            producer.synchronize()
            text_ids = torch.tensor([prepared.text_token_ids], device="cuda")
            expected = build_llm_prompt_embeddings(
                text_token=text_ids,
                text_embed=model.text_embed_tokens(text_ids),
                prompt_speech_token=prepared.llm_prompt_speech_token.to("cuda"),
                speech_embed=model.speech_embedding,
                embedding=torch.zeros(0, 192),
                sos_id=SOS_ID,
                task_id=TASK_ID,
                hidden_size=model.config.hidden_size,
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            ).squeeze(0)
            assert torch.equal(prepared.prompt_input_embeds, expected)
            prompt_keys.append(
                (
                    RadixKey(prepared.input_ids_list),
                    RadixKey(build_embedding_cache_key_ids(expected)),
                )
            )
    for new_left, old_left in prompt_keys:
        for new_right, old_right in prompt_keys:
            for page_size in (1, 16):
                assert new_left.match(new_right, page_size) == old_left.match(
                    old_right, page_size
                )
    print(
        json.dumps(
            {
                "check": "checkpoint_rows_and_radix_prefixes",
                "cases": 9,
                "ordered_pairs": 81,
                "page_sizes": [1, 16],
                "passed": True,
            }
        )
    )


if __name__ == "__main__":
    faulthandler.register(signal.SIGUSR1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--only", choices=("ordering", "lifetime", "checkpoint"))
    arguments = parser.parse_args()
    text_weights = torch.arange(16 * 32, dtype=torch.float32).reshape(16, 32).cuda()
    speech_weights = (
        torch.arange(TOTAL_VOCAB_SIZE * 32, dtype=torch.float32)
        .reshape(TOTAL_VOCAB_SIZE, 32)
        .cuda()
    )
    tables = EmbeddingTables(text_weights, speech_weights)
    preprocessing_stream = torch.cuda.Stream()
    warm(tables, preprocessing_stream)
    if arguments.only in (None, "ordering"):
        check_ordering(tables, preprocessing_stream)
    else:
        pass
    if arguments.only in (None, "lifetime"):
        check_lifetime(tables, preprocessing_stream)
    else:
        pass
    if arguments.checkpoint is not None:
        check_checkpoint(arguments.checkpoint)
    else:
        pass
