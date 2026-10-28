"""Block 6 component check: what a new shape costs torch's cuDNN SDPA against the efficient
backend, on the speech tokenizer decoder's attention as its eager whole-utterance decode calls it.

The decoder attention is 16 heads of 64 over a 72-frame sliding window, so a chunk longer than 72
frames reaches SDPA with a (B, 1, L, L) boolean mask; the eager decode's chunks are 1 to 325 frames
and its batches 1 to 8 requests.

1. time: each backend's first call at a new (B, L) and its repeat, over a grid of shapes.
2. memory: device memory taken outside PyTorch's allocator by the cuDNN shapes of part 1.
3. full device: one shape warmed per backend, PyTorch's allocator then holds all but a few MiB
   of the device (a cached block keeps room for outputs), then one new shape per backend.

python3 cudnn_sdpa_new_shape.py --device cuda:0
"""

import argparse
import statistics
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HEADS = 16
HEAD_DIM = 64
WINDOW = 72
MIB = 1024**2


def inputs(
    batch: int, frames: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    query, key, value = (
        torch.randn(batch, HEADS, frames, HEAD_DIM, device=device, dtype=torch.bfloat16)
        for _ in range(3)
    )
    positions = torch.arange(frames, device=device)
    distance = positions.view(-1, 1) - positions.view(1, -1)
    mask = ((distance >= 0) & (distance < WINDOW)).expand(batch, 1, frames, frames)
    return query, key, value, mask


def timed_call(
    backend: SDPBackend,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    torch.cuda.synchronize()
    start_seconds = time.perf_counter()
    with sdpa_kernel([backend]):
        F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    torch.cuda.synchronize()
    return (time.perf_counter() - start_seconds) * 1000


def outside_allocator_bytes(device: torch.device) -> int:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return total_bytes - free_bytes - torch.cuda.memory_reserved(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headroom-mib", type=int, default=8)
    arguments = parser.parse_args()
    device = torch.device(arguments.device)
    torch.cuda.set_device(device)
    print(
        f"torch {torch.__version__} cudnn {torch.backends.cudnn.version()} "
        f"{torch.cuda.get_device_name(device)}"
    )

    shapes = [
        (batch, frames)
        for batch in (1, 2, 4, 8)
        for frames in (97, 131, 173, 211, 257, 293, 311, 325)
    ]
    for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        timed_call(backend, *inputs(1, 80, device))
        outside_before = outside_allocator_bytes(device)
        first_ms, repeat_ms = [], []
        for batch, frames in shapes:
            tensors = inputs(batch, frames, device)
            first_ms.append(timed_call(backend, *tensors))
            repeat_ms.append(timed_call(backend, *tensors))
        outside_after = outside_allocator_bytes(device)
        print(
            f"{backend.name}: {len(shapes)} new shapes, first call ms median "
            f"{statistics.median(first_ms):.2f} max {max(first_ms):.2f}, repeat ms median "
            f"{statistics.median(repeat_ms):.3f} max {max(repeat_ms):.3f}, outside the "
            f"allocator +{(outside_after - outside_before) / MIB:.1f} MiB "
            f"({(outside_after - outside_before) / MIB / len(shapes):.2f} MiB a shape)"
        )

    room = torch.empty(512 * MIB, dtype=torch.uint8, device=device)
    del room
    free_bytes, _ = torch.cuda.mem_get_info(device)
    filler = torch.empty(
        free_bytes - arguments.headroom_mib * MIB, dtype=torch.uint8, device=device
    )
    free_bytes, _ = torch.cuda.mem_get_info(device)
    print(
        f"full device: reserved {torch.cuda.memory_reserved(device) / MIB:.0f} MiB, "
        f"device free {free_bytes / MIB:.1f} MiB"
    )
    for backend, frames in (
        (SDPBackend.EFFICIENT_ATTENTION, 149),
        (SDPBackend.CUDNN_ATTENTION, 151),
    ):
        try:
            elapsed_ms = timed_call(backend, *inputs(3, frames, device))
        except RuntimeError as error:
            print(f"{backend.name} new shape (3, {frames}): failed: {str(error)[:300]}")
        else:
            print(f"{backend.name} new shape (3, {frames}): ran in {elapsed_ms:.2f} ms")
    del filler


if __name__ == "__main__":
    main()
