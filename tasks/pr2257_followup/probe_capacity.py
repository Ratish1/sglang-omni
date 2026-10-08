import argparse
import json
from array import array
from types import SimpleNamespace

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.llada2_uni.stages import create_sglang_dllm_thinker_executor_from_config
from sglang_omni.scheduling.message import IncomingMessage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-tokens", type=int, required=True)
    arguments = parser.parse_args()
    scheduler = create_sglang_dllm_thinker_executor_from_config(
        "inclusionAI/LLaDA2.0-Uni",
        device="cuda",
        gpu_id=0,
        max_seq_len=512,
        dllm_algorithm="LowConfidenceCFG",
        server_args_overrides={
            "enable_torch_compile": False,
            "max_running_requests": 3,
            "max_prefill_tokens": 512,
            "max_total_tokens": arguments.pool_tokens,
        },
    )
    allocator = scheduler.token_to_kv_pool_allocator
    pool = scheduler.req_to_token_pool
    initial_tokens = allocator.available_size()
    initial_rows = pool.available_size()
    scheduler.result_adapter = lambda result: result.output_ids
    scheduler.request_builder = lambda request: SimpleNamespace(req=request)
    print(json.dumps({"pool_tokens": initial_tokens, "page_size": allocator.page_size}), flush=True)

    for name, branches in (("group", 3), ("following-single", 1)):
        sampling = SamplingParams(max_new_tokens=64, temperature=0.0)
        sampling.normalize(None)
        conditional = Req(
            name, "", array("q", [1] * 96), sampling,
            vocab_size=scheduler.model_config.vocab_size,
            dllm_config=scheduler.dllm_config,
        )
        if branches == 3:
            conditional._uncond_input_ids = [scheduler.dllm_config.mask_id] * 40 + [1] * 56
            conditional._uncond_left_pad_len = 40
            conditional._uncond_img_input_ids = [scheduler.dllm_config.mask_id] * 80 + [1] * 16
            conditional._uncond_img_left_pad_len = 80
        else:
            pass
        scheduler.inbox.put(IncomingMessage(request_id=name, type="new_request", data=conditional))
        scheduler.drain_and_purge()
        should_reject = branches * 160 > initial_tokens
        if should_reject:
            message = scheduler.outbox.get_nowait()
            assert message.type == "error" and message.request_id == name
            assert "480 KV tokens" in message.data
            assert not scheduler.waiting_queue
            print(json.dumps({"case": name, "rejected": message.data}), flush=True)
        else:
            for step in range(5):
                batch = scheduler.schedule_next_batch()
                assert batch is not None, (name, step, allocator.available_size())
                scheduler.apply_results(batch, SimpleNamespace(
                    next_token_ids=[[] if request.is_dllm_prefill() else [2] * 32 for request in batch.reqs],
                    accept_length_per_req_cpu=None, dllm_algo_state=None,
                ))
                scheduler.post_step(batch)
            assert conditional.finished()
            message = scheduler.outbox.get_nowait()
            assert message.type == "result" and message.request_id == name
            assert len(message.data) == 64
            print(json.dumps({"case": name, "completed": True}), flush=True)
        assert not scheduler.waiting_queue and not scheduler.staging_queue
        assert not scheduler.cond_to_unconds and not scheduler.uncond_to_cond
        assert not scheduler.uncond_rids and not scheduler.rid_to_req_data
        assert allocator.available_size() == initial_tokens
        assert pool.available_size() == initial_rows
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
