"""The base runner's seed install on CUDA with about 1 ms of GPU work still queued, as a decode step's forward is when
the sampler installs its seeds: host time of one install, median of 200, at 1, 16 and 32 seeded rows. Run once from
each tree (PYTHONPATH picks main's or A3's base runner).
usage (in the container, one card): PYTHONPATH=<tree> python3 seed_upload_profile_c20261017.py <label>"""

import statistics
import sys
import time
from types import SimpleNamespace

import torch

from sglang_omni.model_runner.base import ModelRunner

QUEUED_CYCLES = 2_000_000
CALLS = 200


def request(seed, request_id):
    return SimpleNamespace(request_id=request_id,
                           data=SimpleNamespace(req=SimpleNamespace(sampling_params=SimpleNamespace(sampling_seed=seed))))


def forward_batch():
    return SimpleNamespace(sampling_info=SimpleNamespace(device="cuda", sampling_seed=None, need_top_p_sampling=True,
                                                         need_top_k_sampling=True, need_min_p_sampling=False))


runner = object.__new__(ModelRunner)
for rows in (1, 16, 32):
    requests = [request(1000 + row, f"r{row}") for row in range(rows)]
    seconds = []
    for _ in range(CALLS):
        batch = forward_batch()
        torch.cuda._sleep(QUEUED_CYCLES)  # noqa: leading-underscore  # upstream name
        start = time.perf_counter()
        runner.install_sampling_seeds(batch, requests)
        seconds.append(time.perf_counter() - start)
        torch.cuda.synchronize()
    print(f"| {sys.argv[1]} | {rows} | {1e6 * statistics.median(seconds):.1f} |")
