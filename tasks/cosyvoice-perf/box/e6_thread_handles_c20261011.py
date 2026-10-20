# SPDX-License-Identifier: Apache-2.0
"""E6: device memory outside torch per new host thread that runs CUDA work. E5 measured 0.16 GiB growing
outside torch while serving, the same with lazy and eager module loading and with no input dependence.
PyTorch keeps cuBLAS and cuDNN handles per thread (CublasHandlePool.cpp DeviceThreadHandles), created on a
thread's first call. Here: after one warm thread, K new threads each run a matmul and a 1-D convolution on
the default stream, then on a new stream of their own, joined one at a time; the memory outside torch
(cudaMemGetInfo used minus torch reserved) is read after each.

    python e6_thread_handles_c20261011.py
"""

from __future__ import annotations

import threading

import torch

THREADS = 8


def outside_torch_mib() -> float:
    free, total = torch.cuda.mem_get_info()
    return (total - free - torch.cuda.memory_reserved()) / 2**20


def work(new_stream: bool) -> None:
    stream = torch.cuda.Stream() if new_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        a = torch.randn(512, 512, device="cuda")
        (a @ a).sum().item()
        x = torch.randn(4, 80, 400, device="cuda")
        w = torch.randn(256, 80, 3, device="cuda")
        torch.nn.functional.conv1d(x, w, padding=1).sum().item()
    torch.cuda.synchronize()


def main() -> None:
    torch.cuda.set_device(0)
    work(new_stream=False)
    print(f"after the main thread's warm work: outside torch {outside_torch_mib():.1f} MiB")
    for new_stream in (False, True):
        for index in range(THREADS):
            before = outside_torch_mib()
            thread = threading.Thread(target=work, args=(new_stream,))
            thread.start()
            thread.join()
            print(
                f"thread {index} new_stream={new_stream}: outside torch "
                f"{outside_torch_mib() - before:+.1f} MiB, torch reserved "
                f"{torch.cuda.memory_reserved() / 2**20:.1f} MiB"
            )


if __name__ == "__main__":
    main()
