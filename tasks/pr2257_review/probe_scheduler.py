import argparse
import json
from array import array
from types import SimpleNamespace

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.llada2_uni.stages import (
    create_sglang_dllm_thinker_executor_from_config,
)
from sglang_omni.scheduling.dllm_scheduler import DllmForwardBatch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-tokens", type=int, default=512)
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
    initial_rows = len(pool.free_slots)
    print(
        json.dumps({"pool_tokens": initial_tokens, "page_size": allocator.page_size}),
        flush=True,
    )
    scheduler.result_adapter = lambda result: result.output_ids

    def admit(name: str, branches: int) -> Req:
        sampling = SamplingParams(max_new_tokens=64, temperature=0.0)
        sampling.normalize(None)
        conditional = Req(
            name,
            "",
            array("q", [1] * 96),
            sampling,
            vocab_size=scheduler.model_config.vocab_size,
            dllm_config=scheduler.dllm_config,
        )
        scheduler.waiting_queue.append(conditional)
        for branch in range(1, branches):
            padding = 40 if branch == 1 else 80
            prompt = [scheduler.dllm_config.mask_id] * padding + [1] * (96 - padding)
            scheduler.create_uncond_companion(
                conditional, prompt, padding, f"-branch-{branch}", branch == 2
            )
        return conditional

    def step(name: str, number: int) -> bool:
        batch = scheduler.schedule_next_batch()
        if batch is None:
            print(
                json.dumps(
                    {
                        "case": name,
                        "step": number,
                        "batch": None,
                        "available_tokens": allocator.available_size(),
                        "staged_prefixes": [
                            len(request.prefix_indices)
                            for request in scheduler.staging_queue
                        ],
                    }
                ),
                flush=True,
            )
            return False
        forward = DllmForwardBatch.init_new(
            batch,
            scheduler.tp_worker.model_runner,
            return_hidden_states_before_norm=False,
        )
        forward.reqs = batch.reqs
        scheduler.apply_cfg_padding_metadata(forward, batch)
        rows = []
        for request in batch.reqs:
            rows.append([] if request.is_dllm_prefill() else [2] * 32)
        print(
            json.dumps(
                {
                    "case": name,
                    "step": number,
                    "ranges": [
                        [request.extend_range.start, request.extend_range.end]
                        for request in batch.reqs
                    ],
                    "pads": forward.dllm_left_pad_lens_cpu,
                    "available_tokens": allocator.available_size(),
                    "rows": [request.kv.req_pool_idx for request in batch.reqs],
                }
            ),
            flush=True,
        )
        scheduler.apply_results(
            batch,
            SimpleNamespace(
                next_token_ids=rows,
                accept_length_per_req_cpu=None,
                dllm_algo_state=None,
            ),
        )
        scheduler.post_step(batch)
        return True

    for branches in (1, 2, 3) if initial_tokens >= 512 else (1,):
        name = f"complete-{branches}"
        conditional = admit(name, branches)
        for iteration in range(5):
            assert step(name, iteration)
        assert conditional.finished()
        assert not scheduler.staging_queue
        assert allocator.available_size() == initial_tokens
        assert len(pool.free_slots) == initial_rows
        assert not scheduler.cond_to_unconds
        assert not scheduler.uncond_rids
        print(json.dumps({"case": name, "completion_and_recovery": "pass"}), flush=True)

    if initial_tokens >= 384:
        name = "abort-companion"
        conditional = admit(name, 3)
        assert step(name, 0)
        scheduler.abort(f"{name}-branch-1")
        scheduler.drain_and_purge()
        assert not scheduler.waiting_queue and not scheduler.staging_queue
        assert allocator.available_size() == initial_tokens
        assert len(pool.free_slots) == initial_rows
        print(json.dumps({"case": name, "recovery": "pass"}), flush=True)

    name = f"three-branches-{initial_tokens}-tokens"
    conditional = admit(name, 3)
    try:
        for iteration in range(7):
            step(name, iteration)
            if conditional.finished():
                break
    except (RuntimeError, AssertionError) as error:
        print(
            json.dumps(
                {"case": name, "error": type(error).__name__, "message": str(error)}
            ),
            flush=True,
        )
    scheduler.abort(conditional.rid)
    scheduler.drain_and_purge()
    assert allocator.available_size() == initial_tokens
    assert len(pool.free_slots) == initial_rows
    print(json.dumps({"case": name, "cleanup": "pass"}), flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
