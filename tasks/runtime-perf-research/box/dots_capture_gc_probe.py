"""Does destroying dead CUDA graphs during a dots.tts vocoder capture invalidate it?

usage (cwd and PYTHONPATH: the sglang-omni tree under test):
  python3 dots_capture_gc_probe.py <mode>

modes:
  dead_graphs_collect_during     a graph pool from an earlier test is dead in a cycle; a collection runs inside the capture
  dead_graphs_collect_before     the same dead pool; the collection runs before the capture
  no_dead_graphs_collect_during  no dead pool; a collection runs inside the capture
Automatic collection is off throughout, so only the explicit collection can free the cycle.
"""

import gc
import json
import sys

import torch

from sglang_omni.models.dots_tts.vocoder_slot_pool import DotsVocoderSlotPool
from tests.unit_test.dots_tts.test_vocoder_cuda_graph import (
    ALL_KEYS,
    make_pool,
    make_streaming_vocoder,
)

mode = sys.argv[1]
device = torch.device("cuda")
gc.disable()
if mode.startswith("dead_graphs"):
    dead_pool = make_pool(device, graph_keys=ALL_KEYS)
    dead_graph_count = len(dead_pool.graph_runner.graphs)
    del dead_pool
else:
    dead_graph_count = 0
if mode == "dead_graphs_collect_before":
    gc.collect()
else:
    pass
collected_during_capture = []
original_forward = DotsVocoderSlotPool.forward


def forward(self, *inputs):
    if (
        mode.endswith("collect_during")
        and not collected_during_capture
        and torch.cuda.is_current_stream_capturing()
    ):
        collected_during_capture.append(gc.collect())
    else:
        pass
    return original_forward(self, *inputs)


DotsVocoderSlotPool.forward = forward
vocoder = make_streaming_vocoder(device)
try:
    pool = vocoder.ensure_slot_pool()
    outcome = {"captured_keys": len(pool.graph_runner.captured_keys)}
except Exception as exc:
    outcome = {"error": repr(exc)[:400]}
print(
    json.dumps(
        {
            "mode": mode,
            "torch": torch.__version__,
            "dead_graphs": dead_graph_count,
            "collected_during_capture": collected_during_capture,
            **outcome,
        }
    ),
    flush=True,
)
