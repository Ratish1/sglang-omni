# SPDX-License-Identifier: Apache-2.0
"""Capture-safe GPU radix-key hash for MOSS-TTS Local generated frames.

The scheduler appends one radix-cache token id per generated frame to a
request's KV chain, and the radix tree keys on those ids. The text channel
alone is the same assistant-slot id for every continuing frame, so the key
must hash the full multi-channel row (text + RVQ codes) to keep a radix match
implying identical audio content.

Prompt rows are hashed once, off the decode hot path, by
``moss_tts.request_builders.build_row_cache_key_ids`` (host-side blake2b); that
call is fine to keep -- it never runs inside a CUDA-graph capture region. The
*generated*-row key, by contrast, is computed every decode step on a tensor
that the local-frame decode just produced on device. Hashing it host-side
forces a GPU->CPU sync (``.cpu()``/``numpy``) every frame, which blocks
CUDA-graph capture and the async-decode lookahead (#734/#736). This module
hashes the row tensor with a fixed-coefficient polynomial on device, with a
torch reference path and an exact fused CUDA path.

See ``docs/design/gpu_radix_hash.md`` for the capture-safety argument, the
collision analysis, and the two-layer verification rubric.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

# <|endoftext|> = 151643 opens the special/control id band. Generated radix
# keys fold strictly below it; the scheduler finishes any request whose
# generated id crosses this boundary (``Req._check_vocab_boundary_finish``), so
# a real (continuing) audio frame must never land in or above the band.
RADIX_HASH_SPACE = 151643

# Polynomial-hash constants.
#
# _MOD is the Mersenne prime 2**31 - 1. With the accumulator and every channel
# value reduced below _MOD (< 2**31) and _BASE < _MOD, each Horner step
# ``acc * _BASE + v`` stays below 2**31 * 2**31 = 2**62, comfortably inside
# signed int64 (max 2**63 - 1). So the int64 ops never overflow and the result
# is bit-reproducible on CPU and GPU -- no implementation-defined wraparound.
#
# _BASE is a large prime well below _MOD. Folding each channel in as a power of
# _BASE (Horner) makes the hash order-sensitive (a channel permutation changes
# the key) and spreads neighbours (a single-channel +/-1 changes the key by a
# power of _BASE mod _MOD). Both constants are arbitrary fixed primes chosen
# only for these size/spread properties; the generated-row key space is private
# to the radix cache, so the exact values carry no on-disk/ABI contract.
_MOD = 2147483647  # 2**31 - 1, Mersenne prime M31
_BASE = 1000000007  # 1e9 + 7, prime, < _MOD
_FUSED_RADIX_HASH_ENV = "MOSS_TTS_LOCAL_FUSED_RADIX_HASH"
_FUSED_RADIX_HASH_BLOCK = 16
_ENV_FALSE_VALUES = frozenset({"0", "false", "off", "no"})


if triton is not None and tl is not None:

    @triton.jit
    def _radix_row_hash_kernel(
        rows_ptr,
        next_text_ptr,
        out_ptr,
        n_rows,
        row_stride,
        col_stride,
        next_text_stride,
        end_id: tl.constexpr,
        hash_space: tl.constexpr,
        mod: tl.constexpr,
        base: tl.constexpr,
        n_cols: tl.constexpr,
        block: tl.constexpr,
    ):
        offsets = tl.program_id(0) * block + tl.arange(0, block)
        mask = offsets < n_rows
        acc = tl.zeros((block,), dtype=tl.int64)
        for col in tl.static_range(0, n_cols):
            values = tl.load(
                rows_ptr + offsets * row_stride + col * col_stride,
                mask=mask,
                other=0,
            ).to(tl.int64)
            values = values % mod
            acc = (acc * base + values) % mod
        text = tl.load(
            next_text_ptr + offsets * next_text_stride,
            mask=mask,
            other=0,
        ).to(tl.int64)
        folded = acc % hash_space
        keys = tl.where(text == end_id, text, folded)
        tl.store(out_ptr + offsets, keys, mask=mask)

else:
    _radix_row_hash_kernel = None


def fused_radix_hash_available() -> bool:
    """Return whether the fused generated-row hash can be launched."""
    return _radix_row_hash_kernel is not None


@lru_cache(maxsize=1)
def _fused_radix_hash_enabled_by_default() -> bool:
    value = os.environ.get(_FUSED_RADIX_HASH_ENV)
    if value is None:
        return True
    return value.strip().lower() not in _ENV_FALSE_VALUES


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _fused_gpu_radix_row_hash(
    rows: torch.Tensor,
    next_text: torch.Tensor,
    end_id: int,
    hash_space: int,
) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D [B, C], got shape {tuple(rows.shape)}")
    if next_text.ndim != 1:
        raise ValueError(
            f"next_text must be 1-D [B], got shape {tuple(next_text.shape)}"
        )
    if rows.shape[0] != next_text.shape[0]:
        raise ValueError(
            "rows and next_text batch sizes differ: "
            f"{rows.shape[0]} != {next_text.shape[0]}"
        )
    n_rows = rows.shape[0]
    if n_rows == 0:
        return torch.empty((0,), dtype=torch.int64, device=rows.device)

    out = torch.empty((n_rows,), dtype=torch.int64, device=rows.device)
    block = _FUSED_RADIX_HASH_BLOCK
    if n_rows > block:
        block = min(1024, _next_power_of_2(n_rows))
    grid = ((n_rows + block - 1) // block,)
    _radix_row_hash_kernel[grid](
        rows,
        next_text,
        out,
        n_rows,
        rows.stride(0),
        rows.stride(1),
        next_text.stride(0),
        end_id,
        hash_space,
        _MOD,
        _BASE,
        rows.shape[1],
        block,
    )
    return out


def poly_row_hash(rows: torch.Tensor) -> torch.Tensor:
    """Fixed-coefficient polynomial hash of each row, in ``[0, _MOD)``.

    ``rows`` is ``[B, C]`` integer. Returns ``[B]`` int64 on ``rows.device``.
    Pure elementwise int64 torch ops (mul / add / remainder) over a static
    channel count -- no host sync, CUDA-graph capturable.
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be 2-D [B, C], got shape {tuple(rows.shape)}")
    work = rows.to(torch.int64)
    acc = torch.zeros(work.shape[0], dtype=torch.int64, device=work.device)
    # Static trip count (one frame = a fixed number of channels): the loop
    # unrolls into a fixed op sequence at capture time.
    for channel in range(work.shape[1]):
        # Reduce defensively in case a caller passes a raw id >= _MOD.
        value = torch.remainder(work[:, channel], _MOD)
        acc = torch.remainder(acc * _BASE + value, _MOD)
    return acc


def gpu_radix_row_hash(
    rows: torch.Tensor,
    next_text: torch.Tensor,
    end_id: int,
    *,
    hash_space: int = RADIX_HASH_SPACE,
    use_fused: bool | None = None,
) -> torch.Tensor:
    """Capture-safe radix token ids for a batch of generated frames.

    ``rows`` is ``[B, C]`` int64 (text channel + RVQ codes); ``next_text`` is
    ``[B]`` (the text-channel id, ``end_id`` for a stop frame). Continuing
    frames get a key in ``[0, hash_space)``; EOS rows keep the raw ``end_id``
    so the existing eos detection still fires. device/dtype follow ``rows``.
    """
    if use_fused is None:
        use_fused = _fused_radix_hash_enabled_by_default()
    if (
        use_fused
        and rows.is_cuda
        and next_text.is_cuda
        and fused_radix_hash_available()
    ):
        return _fused_gpu_radix_row_hash(rows, next_text, end_id, hash_space)
    folded = torch.remainder(poly_row_hash(rows), hash_space)
    return torch.where(next_text == end_id, next_text.to(torch.int64), folded)
