"""Whether a garbage collection during the audio layer graph capture invalidates it, on the tree it runs from.

Run on the box from a tree (cwd and PYTHONPATH the tree): python3 capture_gc_probe.py
Leaves four reference cycles that each hold a captured CUDA graph (what a finished runner in a reference cycle
leaves behind), lowers the collection threshold so a collection runs inside the capture, then captures the audio
layer graphs of the unit tests' small tower with the platform's packed attention. Prints whether every bucket
captured, and the disabled reason when one failed.
"""

from __future__ import annotations

import gc

import torch

from sglang_omni.models.qwen3_omni.components.audio_layer_graph import AudioLayerGraphRunner
from sglang_omni.models.qwen3_omni.components.packed_attention import resolve_packed_attention
from tests.unit_test.qwen3_omni.test_audio_layer_graph import WINDOW, real_tower


class GraphCycle:
    """A captured CUDA graph held by an object that refers to itself."""

    def __init__(self, device: torch.device) -> None:
        self.cycle = self
        self.inputs = torch.zeros(8, device=device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = self.inputs + 1


def main() -> None:
    device = torch.device("cuda", 0)
    runner = AudioLayerGraphRunner(
        real_tower(device), device=device, window=WINDOW,
        packed_attention=resolve_packed_attention(device, torch.bfloat16),
    )
    gc.collect()
    for _ in range(4):
        GraphCycle(device)
    gc.set_threshold(1)
    runner.capture_all()
    gc.set_threshold(700)
    print(f"has_graphs {runner.has_graphs}; disabled_reason {runner.disabled_reason}; buckets {sorted(runner.graphs)}",
          flush=True)


if __name__ == "__main__":
    main()
