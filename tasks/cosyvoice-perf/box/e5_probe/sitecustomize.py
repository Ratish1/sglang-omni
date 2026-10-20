# SPDX-License-Identifier: Apache-2.0
"""E5 probe, in every process that initializes CUDA:
- every E5_PERIOD seconds (default 2) a JSON line in E5_OUT.<pid>.jsonl: free and used device memory, torch
  reserved and allocated, memory outside torch, the allocator's num_alloc_retries and num_ooms, and mapped
  and allocated bytes per (stream, segment pool, small or large pool),
- with E5_HISTORY=1, the allocator's history (context all, Python stacks, E5_MAX_ENTRIES entries) recorded
  from the moment http://127.0.0.1:E5_PORT/health answers 200, and a snapshot dumped to
  E5_OUT.<pid>.<n>.pickle each time E5_PROGRESS gains a "cell" line.
The kit's sitecustomize (E5_KIT_SITECUSTOMIZE) is loaded first.
"""

import importlib.util
import json
import os
import threading
import time
import urllib.request

kit_site = os.environ.get("E5_KIT_SITECUSTOMIZE")
if kit_site:
    spec = importlib.util.spec_from_file_location("kit_sitecustomize", kit_site)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def healthy(port: str) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def cells_done(progress: str) -> list[str]:
    try:
        return [line.split()[1] for line in open(progress) if line.startswith("cell ")]
    except OSError:
        return []


def run() -> None:
    import torch

    while not torch.cuda.is_initialized():
        time.sleep(1.0)
    out = f"{os.environ['E5_OUT']}.{os.getpid()}"
    period = float(os.environ.get("E5_PERIOD", "2"))
    history = os.environ.get("E5_HISTORY") == "1"
    port = os.environ.get("E5_PORT", "")
    progress = os.environ.get("E5_PROGRESS", "")
    recording = False
    dumped = 0
    while True:
        time.sleep(period)
        if history and not recording and port and healthy(port):
            torch.cuda.memory._record_memory_history(
                max_entries=int(os.environ.get("E5_MAX_ENTRIES", "1000000")),
                context="all",
                stacks="python",
            )
            recording = True
            with open(f"{out}.events", "a") as handle:
                handle.write(f"history started {time.time():.2f}\n")
        if recording:
            done = cells_done(progress)
            if len(done) > dumped:
                dumped = len(done)
                path = f"{out}.{dumped:02d}_{done[-1]}.pickle"
                torch.cuda.memory._dump_snapshot(path)
                with open(f"{out}.events", "a") as handle:
                    handle.write(f"dumped {path} {time.time():.2f}\n")
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        stats = torch.cuda.memory_stats()
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
            "num_alloc_retries": int(stats.get("num_alloc_retries", 0)),
            "num_ooms": int(stats.get("num_ooms", 0)),
            "per": per,
        }
        with open(f"{out}.jsonl", "a") as handle:
            handle.write(json.dumps(line) + "\n")


if os.environ.get("E5_OUT"):
    threading.Thread(target=run, daemon=True, name="e5-probe").start()
