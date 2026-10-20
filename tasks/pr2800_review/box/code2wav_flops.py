"""FLOPs of one code2wav decode of 10 new frames, by module, from the checkpoint's own architecture.

Run on the box from a tree: cd <tree> && PYTHONPATH=<tree> python3 /workspace/box/tasks/pr2800_review/box/code2wav_flops.py
HF's Qwen3OmniMoeCode2Wav built from the checkpoint's code2wav config (random weights, bf16, card 0) runs its whole
forward on 10 frames for 1 and 8 rows under torch's FlopCounterMode; a slot decode computes the same outputs for its
new frames, so these are the decode's matmul and conv FLOPs. Prints the total and the top modules.
"""

from __future__ import annotations

import torch
from torch.utils.flop_counter import FlopCounterMode
from transformers import AutoConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True).code2wav_config
    model = Qwen3OmniMoeCode2Wav._from_config(config).to(device="cuda", dtype=torch.bfloat16).eval()
    for rows in (1, 8):
        codes = torch.randint(0, int(config.codebook_size), (rows, int(config.num_quantizers), 10), device="cuda")
        counter = FlopCounterMode(display=False, depth=3)
        with torch.inference_mode(), counter:
            waveform = model(codes)
        flops = counter.get_flop_counts()
        total = counter.get_total_flops()
        print(f"rows {rows}: samples {waveform.shape[-1]}, total {total / 1e9:.2f} GFLOP")
        modules = sorted(
            ((name, sum(ops.values())) for name, ops in flops.items() if name.count(".") <= 2 and name != "Global"),
            key=lambda item: -item[1],
        )
        for name, value in modules[:14]:
            print(f"  {value / 1e9:8.2f} GFLOP  {name}")


if __name__ == "__main__":
    main()
