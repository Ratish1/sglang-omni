# SPDX-License-Identifier: Apache-2.0
"""c2_hop_cost.py with Inductor's pointwise autotuning off: each 1D pointwise
kernel keeps its heuristic config (no first call benchmark), so every boot picks
the same launch configs.

    python c13_no_pointwise_autotune.py ARGS    (c2_hop_cost.py's args)
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch._inductor.config as inductor_config

inductor_config.triton.autotune_pointwise = False
sys.argv[0] = str(Path(__file__).with_name("c2_hop_cost.py"))
runpy.run_path(sys.argv[0], run_name="__main__")
