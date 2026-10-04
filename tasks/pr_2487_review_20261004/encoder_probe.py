"""Measure preprocessing differences through the real weighted vision encoder."""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
from sglang_omni.models.minicpm_o.components.image_processing import process_images
from tasks.pr_2487_review_20261004.probe import image_fixture


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint = snapshot_download(
        "openbmb/MiniCPM-o-4_5",
        revision="503e754207c94da6bb26850b4469f367c9ea3582",
        allow_patterns=[
            "*.py",
            "*.json",
            "*.txt",
            "*.jinja",
            "*.model",
            "model-00004-of-00004.safetensors",
        ],
    )
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    encoder = MiniCPMOImageEncoder(checkpoint, device="cuda:0", dtype="bfloat16")
    report = []
    with torch.inference_mode():
        for width, height, pattern in [
            (448, 448, "ramp"),
            (641, 479, "noise"),
            (641, 479, "edges"),
            (1920, 1080, "edges"),
        ]:
            image = image_fixture(width, height, pattern)
            baseline = processor.process_image([[image]], max_slice_nums=9)
            optimized = process_images(processor.image_processor, [[image]])
            original_embedding = encoder(
                pixel_values=baseline["pixel_values"][0],
                tgt_sizes=baseline["tgt_sizes"][0],
            )["image_embeds"].float()
            modified_embedding = encoder(
                pixel_values=optimized["pixel_values"][0],
                tgt_sizes=optimized["tgt_sizes"][0],
            )["image_embeds"].float()
            difference = modified_embedding - original_embedding
            cosine = torch.nn.functional.cosine_similarity(
                original_embedding, modified_embedding, dim=-1
            )
            row = {
                "size": [width, height],
                "pattern": pattern,
                "shape": list(original_embedding.shape),
                "max_abs_difference": difference.abs().max().item(),
                "mean_abs_difference": difference.abs().mean().item(),
                "relative_l2": (difference.norm() / original_embedding.norm()).item(),
                "min_token_cosine": cosine.min().item(),
                "mean_token_cosine": cosine.mean().item(),
            }
            report.append(row)
            print(json.dumps(row), flush=True)
    (arguments.output / "encoder.json").write_text(json.dumps(report, indent=2))
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
