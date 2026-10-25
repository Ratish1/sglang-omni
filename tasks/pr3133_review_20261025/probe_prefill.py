"""Compare real CosyVoice prefills through the scheduler, before sampling."""

import json
from pathlib import Path

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.fun_cosyvoice3.engine_builder import FunCosyVoice3EngineBuilder
from sglang_omni.models.fun_cosyvoice3.request_builders import (
    CONTROL_TOKEN_IDS,
    TOTAL_VOCAB_SIZE,
    CosyVoice3SGLangRequestData,
    _COSYVOICE3_NULL_TOKENIZER,
)
from sglang_omni.proto.request import OmniRequest, StagePayload


def main() -> None:
    torch.manual_seed(3133)
    scheduler = FunCosyVoice3EngineBuilder().build(
        "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        device="cuda",
        gpu_id=0,
        server_args_overrides={
            "max_running_requests": 4,
            "cuda_graph_max_bs": 4,
            "max_total_tokens": 8192,
            "chunked_prefill_size": 64,
            "cuda_graph_max_bs_prefill": 64,
            "attention_backend": "fa3",
        },
    )
    runner = scheduler.model_runner
    worker = runner.tp_worker
    graph_runner = worker.model_runner.prefill_cuda_graph_runner
    assert graph_runner is not None
    assert 1 in graph_runner.capture_num_tokens
    assert graph_runner.buffer_registry.has_slot("input_embeds")
    original_forward = worker.forward_batch_generation
    records = []

    def checked_forward(forward_batch, **kwargs):
        if not forward_batch.forward_mode.is_extend():
            return original_forward(forward_batch, **kwargs)
        sidecar = get_omni_prefill_inputs(forward_batch)
        assert sidecar is not None
        assert forward_batch.input_embeds is None
        reference = runner.forward_with_input_embeds(
            forward_batch, sidecar.input_embeds
        ).logits_output.next_token_logits.clone()
        result = original_forward(forward_batch, **kwargs)
        actual = result.logits_output.next_token_logits.clone()
        assert actual.shape == reference.shape
        assert torch.isfinite(actual).all()
        log_reference = torch.log_softmax(reference.float(), dim=-1)
        log_actual = torch.log_softmax(actual.float(), dim=-1)
        divergence = (log_reference.exp() * (log_reference - log_actual)).sum(-1)
        assert divergence.max().item() < 0.001
        records.append({
            "rids": forward_batch.rids,
            "tokens": len(forward_batch.input_ids),
            "extend_lengths": forward_batch.extend_seq_lens_cpu,
            "prefix_lengths": forward_batch.extend_prefix_lens_cpu,
            "graph": bool(result.can_run_cuda_graph),
            "max_logit_error": (actual-reference).abs().max().item(),
            "max_kl": divergence.max().item(),
            "top1_equal": bool(torch.equal(actual.argmax(-1), reference.argmax(-1))),
        })
        print(json.dumps(records[-1]), flush=True)
        return result

    worker.forward_batch_generation = checked_forward
    prompt_ids = {}
    prompt_embeds = {}

    def enqueue(name, length, source=None):
        if source is None:
            # Keys deliberately lie outside the text/speech vocabulary. They must
            # never reach the captured token lookup instead of composed rows.
            keys = [10_000_000 + len(prompt_ids) * 10000 + i for i in range(length)]
            text_ids = torch.randint(0, 1000, (length,), device="cuda")
            with torch.no_grad():
                embeds = runner.model.text_embed_tokens(text_ids).clone()
            prompt_ids[name], prompt_embeds[name] = keys, embeds
        else:
            keys, embeds = prompt_ids[source], prompt_embeds[source]
        parameters = SamplingParams(
            max_new_tokens=3, min_new_tokens=3, temperature=0.0,
            stop_token_ids=list(CONTROL_TOKEN_IDS), sampling_seed=3133,
        )
        parameters.normalize(_COSYVOICE3_NULL_TOKENIZER)
        parameters.verify(TOTAL_VOCAB_SIZE)
        req = Req(rid=name, origin_input_text="", origin_input_ids=keys,
                  sampling_params=parameters, vocab_size=TOTAL_VOCAB_SIZE)
        req.tokenizer = _COSYVOICE3_NULL_TOKENIZER
        payload = StagePayload(name, OmniRequest(inputs="probe"), {})
        data = CosyVoice3SGLangRequestData(
            req=req, input_ids=torch.tensor(keys), prompt_input_embeds=embeds,
            output_ids=req.output_ids, max_new_tokens=3, stage_payload=payload,
        )
        scheduler.enqueue_built_request(payload, False, data)

    def drain():
        for _ in range(100):
            batch = scheduler.get_next_batch_to_run()
            if batch is None:
                break
            result = scheduler._run_batch(batch)
            scheduler.process_batch_result(batch, result)
            scheduler.last_batch = batch
        else:
            raise AssertionError("scheduler failed to drain")
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            assert message.type != "error", message.data

    try:
        enqueue("cold31", 31)
        enqueue("cold17", 17)
        drain()
        enqueue("cached31", 31, "cold31")
        drain()
        enqueue("chunked129", 129)
        drain()
        enqueue("cold2", 2)
        drain()
        enqueue("cached129", 129, "chunked129")
        drain()
        assert any(record["graph"] and record["tokens"] == 1 for record in records)
        assert any(record["graph"] and len(record["rids"]) > 1 for record in records)
        assert any(record["graph"] and max(record["prefix_lengths"]) >= 64 for record in records)
        assert all(record["graph"] for record in records)
        Path("/data/review3133oct25/prefill.json").write_text(json.dumps(records, indent=2))
    finally:
        scheduler.shutdown_request_build_executor()
        scheduler.shutdown_resources()


if __name__ == "__main__":
    main()
