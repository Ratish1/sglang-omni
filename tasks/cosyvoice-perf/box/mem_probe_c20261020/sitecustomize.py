# SPDX-License-Identifier: Apache-2.0
"""Memory probe for item 3's worst step check (10-20): every MEM_PROBE_PERIOD seconds (default 1) in every process
that has initialized CUDA, one JSON line to MEM_PROBE_OUT.<pid>.jsonl with free and used device memory, torch
reserved and allocated, the memory outside torch, and the allocator's num_alloc_retries (the out of memory retry that
frees cached blocks and synchronizes) and num_ooms. Put this directory first on PYTHONPATH and set MEM_PROBE=1."""

import json
import os
import sys
import threading
import time


def sample_forever(out_prefix: str, period: float) -> None:
    # note (ratish): never import torch here: this file loads in every process on PYTHONPATH, inductor's compile
    # workers too, and a torch import on this thread beside the worker's own made the pool exit uncleanly on torch
    # 2.14 (10-22, every worst step run: "SubprocPool unclean exit", then "Attempting to use a closed pool").
    while True:
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_initialized():
                break
            else:
                pass
        except AttributeError:
            pass
        time.sleep(1.0)
    path = f"{out_prefix}.{os.getpid()}.jsonl"
    while True:
        time.sleep(period)
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        reserved = int(stats.get("reserved_bytes.all.current", 0))
        line = {
            "t": round(time.time(), 2),
            "free": free,
            "used": total - free,
            "torch_reserved": reserved,
            "torch_allocated": int(stats.get("allocated_bytes.all.current", 0)),
            "non_torch": total - free - reserved,
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
        }
        with open(path, "a") as handle:
            handle.write(json.dumps(line) + "\n")


if os.environ.get("MEM_PROBE") == "1":
    threading.Thread(
        target=sample_forever,
        args=(os.environ["MEM_PROBE_OUT"], float(os.environ.get("MEM_PROBE_PERIOD", "1"))),
        daemon=True,
        name="mem-probe",
    ).start()
