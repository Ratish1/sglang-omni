# SPDX-License-Identifier: Apache-2.0
"""Step 1's packed DiT with RoPE applied out of place in the compiled forward.

rotate_in_place writes the rotary dims of query and key in place; compiled, that
mutation becomes a full copy of each before FA3 (packed_fa3_select_view, two
launches per block per step). Rotating out of place with a cat lets Inductor
write the rotated tensor in one kernel. The rotated part is cast to the input
dtype before the cat, the same rounding as copy_. Eager keeps the in-place form.

    python c12_rope_out_of_place.py test          GPU parity test (torch.equal)
    python c12_rope_out_of_place.py sweep ARGS    c2_hop_cost.py with ARGS
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

from sglang_omni.models.fun_cosyvoice3 import packed_dit


def rotated(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # One pointwise expression over every channel, so Inductor writes the
    # result in one kernel: the rotary channels as rotate_in_place computes
    # them, the rest x itself (a cat realized the rotary part, then copied).
    rotary_dims, width = cos.shape[-1], x.shape[-1]
    cos = torch.nn.functional.pad(cos, (0, width - rotary_dims))
    sin = torch.nn.functional.pad(sin, (0, width - rotary_dims))
    half = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
    turned = (x * cos + half * sin).to(x.dtype)
    is_rotary = torch.arange(width, device=x.device) < rotary_dims
    return torch.where(is_rotary, turned, x)


def attend(attn, x, rope, attention):
    x = x.to(attn.to_q.weight.dtype)
    query = attn.to_q(x)
    key = attn.to_k(x)
    value = attn.to_v(x)
    if torch.compiler.is_compiling():
        query = rotated(query, *rope)
        key = rotated(key, *rope)
    else:
        packed_dit.rotate_in_place(query, *rope)
        packed_dit.rotate_in_place(key, *rope)
    out = attention(query, key, value).to(query.dtype)
    return attn.to_out[1](attn.to_out[0](out))


packed_dit.PackedDiT.attend = staticmethod(attend)

if __name__ == "__main__":
    # runpy replaces __main__'s globals, which Dynamo's guards look the patch
    # up in; installed from an importable module instead.
    sys.path.insert(0, str(Path(__file__).parent))
    import c12_rope_out_of_place  # noqa: F401

    mode, rest = sys.argv[1], sys.argv[2:]
    if mode == "test":
        import pytest

        sys.exit(
            pytest.main(
                ["-q", "tests/unit_test/fun_cosyvoice3/test_dit_compile_gpu.py", *rest]
            )
        )
    else:
        sys.argv = [str(Path(__file__).with_name("c2_hop_cost.py")), *rest]
        runpy.run_path(sys.argv[0], run_name="__main__")
