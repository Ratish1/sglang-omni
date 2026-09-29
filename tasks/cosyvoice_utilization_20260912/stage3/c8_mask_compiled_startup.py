# SPDX-License-Identifier: Apache-2.0
"""c1_compile_startup.py with the native DiT's chunk mask compiled: the tree's
compile_dit_backbone skips its torch.compiler.disable wrap, nothing else changes.

    python c8_mask_compiled_startup.py --out startup.json   (c1_compile_startup's args)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from sglang_omni.models.fun_cosyvoice3 import stages

stages.CHUNK_MASK_COMPILE_DISABLED = True
sys.argv[0] = str(Path(__file__).with_name("c1_compile_startup.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
