"""Focused review probes. Run in the pinned Linux/H100 environment, from repo root.

PYTHONPATH=. python /path/to/remote_probes.py

These exercise production helpers with small fixtures, without loading a checkpoint.
They report storage retention and graph-key coverage; they do not benchmark serving.
This file has not been executed locally.
"""

from __future__ import annotations

import json
from collections import deque
from types import SimpleNamespace

import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner
from sglang_omni.models.qwen3_tts.sglang_model import Qwen3TTSTalker
from sglang_omni.scheduling.generation_batch_policy import build_default_cuda_graph_bs
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


def graph_capacity() -> None:
    """Execute the real capture planner, replacing only GPU graph construction."""
    for maximum in (16, 64, 128):
        talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
        torch.nn.Module.__init__(talker)
        talker.config = SimpleNamespace(
            code_predictor_config=SimpleNamespace(vocab_size=2048)
        )
        talker._predictor_graph_enabled = True
        talker._predictor_graphs = {}
        talker._predictor_graph_capture_count = 0
        talker._predictor_graph_startup_count = 0
        talker._predictor_graph_batch_sizes = tuple(
            build_default_cuda_graph_bs(maximum)
        )
        talker._capture_predictor_graph = lambda bucket, signature: SimpleNamespace(
            bucket=bucket, signature=signature
        )
        talker.capture_predictor_graphs(do_sample=True, top_k=50, top_p=1.0)
        missing = [
            bucket
            for bucket in talker._predictor_graph_batch_sizes
            if bucket >= 2
            and (bucket, "sampled", 50, False, False, True)
            not in talker._predictor_graphs
        ]
        print(
            json.dumps(
                {
                    "probe": "startup_key_coverage",
                    "maximum": maximum,
                    "bucket_count": len(talker._predictor_graph_batch_sizes),
                    "captured": len(talker._predictor_graphs),
                    "missing_reachable_mixed_buckets": missing,
                }
            )
        )


def history_storage() -> None:
    """Keep one request alive while its peers finish, as terminal cleanup does."""
    batch_size, hidden_size, steps = 16, 2048, 1024
    device = torch.device("cuda")
    dtype = torch.bfloat16
    target = torch.nn.Embedding(batch_size, hidden_size, device=device, dtype=dtype)
    target.weight.requires_grad_(False)
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = SimpleNamespace(_decode_feedback_embedding=target)
    survivor = SGLangARRequestData()
    pad = torch.zeros(hidden_size, device=device, dtype=dtype)
    survivor.tts_pad_embed = pad
    expected_rows = []

    with torch.inference_mode():
        for step in range(steps):
            # Rotate the surviving request through batch slots to check row identity.
            survivor_slot = step % batch_size
            requests = []
            for slot in range(batch_size):
                data = survivor if slot == survivor_slot else SGLangARRequestData()
                feedback = torch.full(
                    (hidden_size,),
                    float((step + slot) % 31),
                    device=device,
                    dtype=dtype,
                )
                text = torch.full((hidden_size,), 0.5, device=device, dtype=dtype)
                data.pending_feedback_queue = deque([feedback])
                data.pending_text_queue = deque([text])
                data.tts_pad_embed = pad
                requests.append(SimpleNamespace(data=data))
                if data is survivor:
                    # Independent per-row arithmetic; CPU storage avoids affecting
                    # the retained-device-storage count below.
                    expected_rows.append((feedback + text).cpu())
            forward_batch = SimpleNamespace(
                input_ids=torch.zeros(batch_size, dtype=torch.long, device=device)
            )
            runner._write_feedback_buffers(forward_batch, requests)
            for request in requests:
                if request.data is not survivor:
                    request.data.decode_input_embeds = None

        target.weight.fill_(-100)
        actual = torch.stack(survivor.decode_input_embeds).cpu()
        assert torch.equal(
            actual, torch.stack(expected_rows)
        ), "history was overwritten or reordered"
        storage_sizes = {
            row.untyped_storage().data_ptr(): row.untyped_storage().nbytes()
            for row in survivor.decode_input_embeds
        }
        logical_bytes = sum(
            row.numel() * row.element_size() for row in survivor.decode_input_embeds
        )
        retained_bytes = sum(storage_sizes.values())
        print(
            json.dumps(
                {
                    "probe": "survivor_history",
                    "batch_size": batch_size,
                    "steps": steps,
                    "logical_bytes": logical_bytes,
                    "retained_storage_bytes": retained_bytes,
                    "amplification": retained_bytes / logical_bytes,
                    "history_bits_and_row_order": "PASS",
                }
            )
        )


def temperature_staging() -> None:
    """Compare the relocated floor against FP32 staging followed by device clamp."""
    device = torch.device("cuda")
    values = [0.0, 1e-12, 9.999999e-6, 1e-5, 1.0000001e-5, 0.9]
    talker = Qwen3TTSTalker.__new__(Qwen3TTSTalker)
    torch.nn.Module.__init__(talker)
    talker.config = SimpleNamespace(
        code_predictor_config=SimpleNamespace(vocab_size=2048)
    )
    for field in ("_sub_temperature_tensor", "_sub_top_p_tensor"):
        setattr(
            talker, field, torch.empty(len(values), device=device, dtype=torch.float32)
        )
    for field in (
        "_sub_top_k_tensor",
        "_semantic_sampling_seed_tensor",
        "_sub_sampling_seed_tensor",
    ):
        setattr(
            talker, field, torch.empty(len(values), device=device, dtype=torch.long)
        )
    talker._sub_do_sample_tensor = torch.empty(
        len(values), device=device, dtype=torch.bool
    )
    requests = [
        SimpleNamespace(
            data=SimpleNamespace(
                semantic_sampling_seed=1,
                subtalker_dosample=True,
                subtalker_temperature=value,
                subtalker_top_p=1.0,
                subtalker_top_k=50,
                subtalker_sampling_seed=2,
            )
        )
        for value in values
    ]
    talker.prepare_decode_buffers(requests)
    previous = torch.tensor(values, device=device, dtype=torch.float32).clamp_min(1e-5)
    assert torch.equal(
        talker._sub_temperature_tensor.view(torch.int32), previous.view(torch.int32)
    )
    print(json.dumps({"probe": "temperature_staging_bits", "status": "PASS"}))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("Run this probe in the pinned Linux/CUDA environment.")
    graph_capacity()
    temperature_staging()
    history_storage()
