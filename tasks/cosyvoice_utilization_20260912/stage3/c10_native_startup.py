# SPDX-License-Identifier: Apache-2.0
"""c1_compile_startup.py with the native DiT's chunk mask compiled and
x_transformers' RoPE autocast regions replaced by c9_rope_exact.py's bit
identical versions, so the native graphs can use the AOTAutograd cache.

    python c10_native_startup.py --out startup.json   (c1_compile_startup's args)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from x_transformers.x_transformers import RotaryEmbedding

from sglang_omni.models.fun_cosyvoice3 import stages

sys.path.insert(0, str(Path(__file__).parent))
from c9_rope_exact import apply_rotary_pos_emb, rotary_forward  # noqa: E402
from cosyvoice.flow.DiT import modules as dit_modules  # noqa: E402

stages.CHUNK_MASK_COMPILE_DISABLED = True
RotaryEmbedding.forward = rotary_forward
dit_modules.apply_rotary_pos_emb = apply_rotary_pos_emb
sys.argv[0] = str(Path(__file__).with_name("c1_compile_startup.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
