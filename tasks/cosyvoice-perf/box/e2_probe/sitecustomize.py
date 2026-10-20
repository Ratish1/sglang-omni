# SPDX-License-Identifier: Apache-2.0
"""E2 probe: the torch allocator's segments per stream and pool, and the memory outside torch, sampled
every E2_STREAM_PROBE_PERIOD seconds (default 2) in every process that has initialized CUDA, appended as
JSON lines to E2_STREAM_PROBE_OUT.<pid>.jsonl. The kit's sitecustomize (E2_KIT_SITECUSTOMIZE) is loaded
first, so the count probe keeps working.

Per sample: free and used device memory (cudaMemGetInfo), torch reserved and allocated, the memory
outside torch (used minus torch reserved), and per (stream, segment pool, small or large pool): mapped
segment bytes and allocated bytes.
"""

import importlib.util
import json
import os
import threading
import time

kit_site = os.environ.get("E2_KIT_SITECUSTOMIZE")
if kit_site:
    spec = importlib.util.spec_from_file_location("kit_sitecustomize", kit_site)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def sample_forever(out_prefix: str, period: float) -> None:
    import torch

    while not torch.cuda.is_initialized():
        time.sleep(1.0)
    path = f"{out_prefix}.{os.getpid()}.jsonl"
    while True:
        time.sleep(period)
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        per: dict[str, list[int]] = {}
        for segment in torch.cuda.memory_snapshot():
            key = (
                f"{segment['stream']}|{tuple(segment['segment_pool_id'])}|"
                f"{segment['segment_type']}"
            )
            totals = per.setdefault(key, [0, 0])
            totals[0] += int(segment["total_size"])
            totals[1] += int(segment["allocated_size"])
        line = {
            "t": round(time.time(), 2),
            "free": free,
            "used": total - free,
            "torch_reserved": reserved,
            "torch_allocated": torch.cuda.memory_allocated(),
            "non_torch": total - free - reserved,
            "per": per,
        }
        with open(path, "a") as handle:
            handle.write(json.dumps(line) + "\n")


if os.environ.get("E2_STREAM_PROBE") == "1":
    threading.Thread(
        target=sample_forever,
        args=(
            os.environ["E2_STREAM_PROBE_OUT"],
            float(os.environ.get("E2_STREAM_PROBE_PERIOD", "2")),
        ),
        daemon=True,
        name="e2-stream-probe",
    ).start()
