# #2631 review F1: the whole history RoPE table sized from the largest tier holds the same angles
# as the one sized from the noise buffer, for every position the smaller one keeps. Uses the DiT's
# own rotary embedding constructor and its head size from the checkpoint config.
import re
import sys

import torch
from huggingface_hub import snapshot_download
from x_transformers.x_transformers import RotaryEmbedding

config = open(f"{snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512')}/cosyvoice3.yaml").read()
dim = int(re.search(r"\n\s+dim: (\d+)", config).group(1))
heads = int(re.search(r"\n\s+heads: (\d+)", config).group(1))
dim_head = int(re.search(r"\n\s+dim_head: (\d+)", config).group(1))
print(f"DiT dim {dim} heads {heads} dim_head {dim_head}")
rotary = RotaryEmbedding(dim_head).cuda()
large, large_scale = rotary.forward_from_seq_len(15000)
for tier in (int(arg) for arg in sys.argv[1:]):
    small, small_scale = rotary.forward_from_seq_len(tier)
    print(
        f"tier {tier}: identical={torch.equal(large[:, :tier], small)} "
        f"shapes {tuple(large.shape)} {tuple(small.shape)} scales {large_scale} {small_scale}"
    )
