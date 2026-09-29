# SPDX-License-Identifier: Apache-2.0
"""The native DiT compile with the chunk mask compiled, RoPE without
x_transformers' autocast regions (c9_rope_exact.py), and automatic dynamic
instead of dynamic=True: the batch and frame dims are hinted dynamic before
every call, so floats such as the LayerNorm eps and the head counts stay static.

    python c11_native_auto_dynamic.py --out startup.json   (c1_compile_startup's args)

install() is also used by c3_native_compile.py --variant auto_dynamic.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch
import torch._dynamo as dynamo

sys.path.insert(0, str(Path(__file__).parent))


def install() -> None:
    from c9_rope_exact import apply_rotary_pos_emb, rotary_forward
    from cosyvoice.flow.DiT import modules as dit_modules
    from x_transformers.x_transformers import RotaryEmbedding

    from sglang_omni.models.fun_cosyvoice3 import stages

    stages.CHUNK_MASK_COMPILE_DISABLED = True
    RotaryEmbedding.forward = rotary_forward
    dit_modules.apply_rotary_pos_emb = apply_rotary_pos_emb
    original_compile = torch.compile

    def compile(function, **kwargs):
        if kwargs.pop("dynamic", None) is not True:
            return original_compile(function, **kwargs)
        else:
            pass
        compiled = original_compile(function, **kwargs)

        def forward(x, mask, mu, t, spks=None, cond=None, streaming=False):
            for tensor in (x, mask, mu, cond):
                dynamo.maybe_mark_dynamic(tensor, (0, 2))
            dynamo.maybe_mark_dynamic(spks, 0)
            dynamo.maybe_mark_dynamic(t, 0)
            return compiled(x, mask, mu, t, spks, cond, streaming=streaming)

        return forward

    torch.compile = compile


if __name__ == "__main__":
    install()
    sys.argv[0] = str(Path(__file__).with_name("c1_compile_startup.py"))
    runpy.run_path(sys.argv[0], run_name="__main__")
