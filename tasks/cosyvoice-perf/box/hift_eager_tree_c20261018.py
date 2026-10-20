"""One tree's eager whole utterance HiFT call (the buffered path, real checkpoint, 400 frames, 1 row): the median
wall of 200 calls after a warm up, printed as one line. usage: PYTHONPATH=<tree> python3 hift_eager_tree_c20261018.py"""

import statistics
import time

import torch
from huggingface_hub import snapshot_download
from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

from sglang_omni.models.fun_cosyvoice3 import hift_kernels, stages

device = torch.device("cuda")
checkpoint_dir = snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_files_only=True)
_, hift = stages.load_cosyvoice3_flow_hift_lightweight(checkpoint_dir, device="cuda")
stages.keep_hift_constants_on_device(hift, "cuda")
stages.patch_causal_conv_cache()
for submodule in list(hift.modules()):
    while is_parametrized(submodule):
        name = next(iter(submodule.parametrizations.keys()))
        remove_parametrizations(submodule, name, leave_parametrized=True)
stages.patch_hift_snake(hift)
stages.patch_hift_spectra(hift, device)
mel = (torch.randn(1, 80, 400, generator=torch.Generator().manual_seed(0)) * 2 - 6).to(device)
walls = []
with torch.inference_mode():
    for _ in range(20):
        hift.inference(speech_feat=mel, finalize=True)
    torch.cuda.synchronize()
    for _ in range(200):
        start = time.perf_counter()
        hift.inference(speech_feat=mel, finalize=True)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - start) * 1000)
print(f"{hift_kernels.__file__} eager median {statistics.median(walls):.3f} ms p10 "
      f"{statistics.quantiles(walls, n=10)[0]:.3f} ms")
