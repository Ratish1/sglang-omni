"""Run the actual codec exchange with graph scratch indices or distinct control slots."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from sglang_omni.kernels.codec_state import history_exchange, overlap_exchange
from sglang_omni.models.qwen3_tts.incremental_codec_cuda_graph import (
    IncrementalCodecGraphPool,
    Qwen3TTSIncrementalCodecCudaGraphRunner,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("history", "overlap"))
    parser.add_argument("slots", choices=("scratch", "distinct"))
    arguments = parser.parse_args()
    device = torch.device("cuda", 0)
    runner = Qwen3TTSIncrementalCodecCudaGraphRunner(
        SimpleNamespace(),
        device=device,
        dtype=torch.bfloat16,
        num_quantizers=16,
        mode="warm",
        fresh_frames=(1,),
        batch_sizes=(4,),
        arena=SimpleNamespace(scratch_slot=0),
        graph_pool=IncrementalCodecGraphPool(stream_priority=0),
        enabled=False,
    )
    if arguments.slots == "scratch":
        slot_index = runner.scratch_index(4)
    else:
        slot_index = torch.arange(4, device=device)
    print(
        json.dumps({"operation": arguments.operation, "indices": slot_index.tolist()}),
        flush=True,
    )
    if arguments.operation == "history":
        history_pool = torch.zeros(4, 54, 64, device=device, dtype=torch.bfloat16)
        padded = torch.ones(4, 55, 64, device=device, dtype=torch.bfloat16)
        history_exchange(padded, history_pool, slot_index, new_samples=1, is_cold=False)
    else:
        overlap_pool = torch.zeros(4, 4, 64, device=device, dtype=torch.bfloat16)
        output = torch.ones(4, 8, 64, device=device, dtype=torch.bfloat16)
        overlap_exchange(
            output, overlap_pool, slot_index, emitted_samples=4, is_cold=False
        )
    torch.cuda.synchronize()
    print("exchange finished without a sanitizer error", flush=True)


if __name__ == "__main__":
    main()
