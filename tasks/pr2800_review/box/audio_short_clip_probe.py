"""Whether the shortest MMSU clips' distance from fp32 is the model's own bf16 behavior: Transformers' audio tower in
bf16 against run311's fp32 reference.

Run on the box from main's tree (cwd and PYTHONPATH the tree): python3 audio_short_clip_probe.py --probe /data/pr2800/run311/probe
Builds Transformers' Qwen3OmniMoeAudioEncoder from the checkpoint (build_audio_tower, bf16, card 0) without omni's
attention patches, encodes each of the 2000 clips alone through it with the chunk inputs omni's encoder hands it, and
prints the relative L2 to reference.pt next to main's served graph path (main.json): median, clips over 0.1 and 0.05,
and the farthest clips on main with their frame count.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as hf_modeling

from sglang_omni.models.qwen3_omni.components.audio_encoder import build_audio_tower
from sglang_omni.models.qwen3_omni.components.common import load_thinker_config

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def encode(tower: torch.nn.Module, features: torch.Tensor, frames: int) -> torch.Tensor:
    device = next(tower.parameters()).device
    lengths_cpu = torch.tensor([frames], dtype=torch.long)
    input_features = features[0, :, :frames].to(device=device, dtype=torch.bfloat16).contiguous()
    padded_feature, chunk_lengths = hf_modeling.chunk_and_pad_features(input_features, lengths_cpu, tower.n_window)
    valid_indices = hf_modeling.get_valid_indices(chunk_lengths, tower.n_window)
    cu_seqlens = hf_modeling.get_audio_cu_seqlens(chunk_lengths, lengths_cpu, tower.n_window_infer, tower.n_window)
    return tower(
        input_features,
        feature_lens=lengths_cpu.to(device),
        padded_feature=padded_feature,
        chunk_lengths=chunk_lengths,
        valid_indices=valid_indices.to(device),
        cu_seqlens=cu_seqlens.to(device),
        max_seqlen=int((cu_seqlens[1:] - cu_seqlens[:-1]).max()),
    ).last_hidden_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    path = snapshot_download(MODEL)
    tower = build_audio_tower(
        path, thinker_cfg=load_thinker_config(path), torch_dtype=torch.bfloat16, device="cuda:0"
    ).eval()
    print(f"attention {tower.config._attn_implementation}")  # noqa: leading-underscore  # Transformers config field
    inputs = torch.load(args.probe / "inputs.pt")
    truth = torch.load(args.probe / "reference.pt")
    served = json.loads((args.probe / "main.json").read_text())["graph"]["distance"]
    native = []
    with torch.no_grad():
        for (features, mask), expected in zip(inputs, truth):
            rows = encode(tower, features, int(mask.sum())).float()
            native.append(float((rows - expected.cuda()).norm() / expected.norm().clamp_min(1e-30)))
    for name, distances in (("main served graph", served), ("Transformers bf16", native)):
        print(f"{name}: median {statistics.median(distances):.4f}, over 0.1 {sum(d > 0.1 for d in distances)}, "
              f"over 0.05 {sum(d > 0.05 for d in distances)}, largest {max(distances):.4f}")
    for index in sorted(range(len(served)), key=lambda i: -served[i])[:15]:
        print(f"clip {index} frames {int(inputs[index][1].sum())}: served {served[index]:.4f} "
              f"Transformers bf16 {native[index]:.4f}")


if __name__ == "__main__":
    main()
