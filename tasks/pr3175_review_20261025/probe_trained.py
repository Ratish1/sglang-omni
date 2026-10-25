"""Compare trained vision outputs on every image in the MMMU CI set."""

import base64
import io
import json
import time
import types
from pathlib import Path

import torch
from base_vision_compat import Qwen3OmniMoeVisionEncoderCompat as BaseEncoder
from PIL import Image
from transformers import AutoProcessor

from benchmarks.dataset.mmmu import load_mmmu_samples
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder


@torch.inference_mode()
def main() -> None:
    model_id = "marksverdhei/Qwen3-Omni-30B-A3B-FP8"
    torch.manual_seed(3175)
    model = Qwen3OmniImageEncoder(model_id, device="cuda:0", dtype="bf16")
    visual = model.visual
    processor = AutoProcessor.from_pretrained(model_id)
    native_method = visual.interpolate_position_embeddings

    def legacy(self, grid):
        return BaseEncoder.legacy_pos_embed_interpolate(self, torch.tensor(grid))

    legacy_method = types.MethodType(legacy, visual)
    samples = load_mmmu_samples(repo_id="zhaochenyang20/mmmu-ci-50")
    inputs = []
    for sample in samples:
        for image_index, uri in enumerate(sample.image_data_uris):
            image = Image.open(
                io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))
            ).convert("RGB")
            encoded = processor.image_processor(images=[image], return_tensors="pt")
            inputs.append((sample.sample_id, image_index, encoded))
    # The full image set supplies real aspect ratios and serving patch grids.
    records = []
    for sample_id, image_index, encoded in inputs:
        pixels = encoded["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
        grid = encoded["image_grid_thw"]
        row = {
            "sample_id": sample_id,
            "image_index": image_index,
            "grid": grid.tolist(),
        }
        outputs = []
        for name, method in (("base", legacy_method), ("current", native_method)):
            visual.interpolate_position_embeddings = method
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = visual(pixels, grid_thw=grid, return_dict=True)
            torch.cuda.synchronize()
            row[f"{name}_wall_ms"] = (time.perf_counter() - started) * 1000
            outputs.append(
                (
                    output.pooler_output.cpu().float(),
                    [x.cpu().float() for x in output.deepstack_features],
                )
            )
        row["base_current_embedding_relative_l2"] = (
            (outputs[0][0] - outputs[1][0]).norm() / outputs[0][0].norm()
        ).item()
        row["base_current_deepstack_relative_l2"] = [
            ((a - b).norm() / a.norm()).item()
            for a, b in zip(outputs[0][1], outputs[1][1], strict=True)
        ]
        row["bf16_outputs"] = outputs
        records.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "bf16_outputs"}),
            flush=True,
        )
    visual.interpolate_position_embeddings = native_method
    visual.float()
    # Joint RoPE's CUDA entry supports the model's FP32 tensors as well.
    for row, (_, _, encoded) in zip(records, inputs, strict=True):
        output = visual(
            encoded["pixel_values"].to(device="cuda", dtype=torch.float32),
            grid_thw=encoded["image_grid_thw"],
            return_dict=True,
        )
        reference = output.pooler_output.cpu().float()
        reference_stack = [x.cpu().float() for x in output.deepstack_features]
        for name, (embedding, stack) in zip(
            ("base", "current"), row.pop("bf16_outputs"), strict=True
        ):
            row[f"{name}_embedding_relative_l2_to_fp32"] = (
                (embedding - reference).norm() / reference.norm()
            ).item()
            row[f"{name}_deepstack_relative_l2_to_fp32"] = [
                ((x - y).norm() / y.norm()).item()
                for x, y in zip(stack, reference_stack, strict=True)
            ]
        assert torch.isfinite(output.pooler_output).all()
    result = {
        "model": model_id,
        "samples": len(samples),
        "images": len(inputs),
        "config": visual.config.to_dict(),
        "records": records,
    }
    Path("/data/review3175oct25/trained.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"completed_images": len(inputs)}), flush=True)


if __name__ == "__main__":
    main()
